# Security

What is automated, what is deliberately manual, and what is out of scope. This
is a portfolio project serving a public demo on free hosting; the threat model
is "a stranger on the internet can reach `/predict`, and the repository is
public", not a payments company's.

## The supply chain

The service loads a pickle it fetches over the network, so most of the risk
here is supply-chain rather than application logic.

| link | what protects it |
|---|---|
| `model.joblib` (a pickle: loading it executes it) | its sha256 is checked against the `champion-<sha12>` in `FRAUD_CHAMPION_URL` **before** it is written or deserialized; an unpinned remote fetch is refused (`fraud.serve.app.expected_champion_digest`, `docs/deployment.md`) |
| which champion is live | `release.py` refuses to tag an unpublished champion; `deploy.yml` fails the release unless live `/health` reports the champion the tag was cut for (`docs/promotion.md`) |
| which code is live | the deploy hook is pinned with `?ref=<the tag's commit>`, and the workflow fails unless `/health` reports that commit back |
| the block/review thresholds | `bands` are policy (ADR 0006) and are never taken from a fetched manifest, which is not digest-pinned. They come from `configs/serving.yaml`, which ships in the image from a reviewed commit, and a champion whose manifest disagrees with it fails startup rather than serving a policy nobody approved |
| Python dependencies | `uv.lock` pins every version and hash; `security.yml` audits it weekly, and Dependabot security updates are enabled |
| GitHub Actions | pinned to releases — never a floating branch — and updated by Dependabot |
| base images | Dependabot watches both Dockerfiles |
| the host's `FRAUD_CHAMPION_URL` | set by `deploy.yml` from the champion in the tag annotation, then read back and checked, so the value that carries the digest pin is written by the pipeline rather than by hand. `RENDER_API_KEY` is an **account-wide** Render credential held as a repository secret; it is the broadest secret in CI and is the reason the write is limited to one key and verified |

The manifest and the frozen golden cannot attest the artifact's provenance:
they are fetched from the same store as the artifact, so anything able to
substitute one can substitute the others. Only the URL, which is deployer
configuration, is outside the store. That is why it is the anchor.

## Automation

`.github/dependabot.yml` — weekly PRs for `uv`, `github-actions` and both
Dockerfiles. Minor and patch Python bumps are grouped into one PR; majors
arrive alone so a break is reviewed by itself. Dependabot does not merge:
`ci.yml` runs on its PRs like any other and the owner reviews the diff.

`.github/workflows/security.yml` — on push, on PR, and **weekly**:

- `pip-audit` against the runtime dependencies (what the image ships) and
  against the full set including training and tooling. It audits what
  `uv.lock` pins rather than a fresh resolution, because the lock is what CI
  installs and what the image builds from.
- CodeQL (`security-extended`) for Python, into the Security tab.

The weekly schedule is the part that matters. A vulnerability disclosed
against a dependency this repository has not touched in a month still needs
to surface, and a push-triggered scan would never see it.

It is separate from `ci.yml` on purpose. Both jobs need the network, and
`ci.yml` is what `deploy.yml` runs before a release, so folding them in would
let a PyPI outage block a deployment. **The consequence is that a known
vulnerability fails merges, not releases.** Making it block a release is a
one-line change (call `security.yml` from `deploy.yml`) and a deliberate
trade of release availability for strictness; it has not been made.

## Application surface

The public demo runs with no API key by design — a dashboard that needs a key
is not a demo. What limits it instead: per-client application rate limits, an
upload cap, a 5,000-row CSV ceiling, a 60 s request time budget, and admin
endpoints (`POST /outcomes`, `GET /audit/recent`, `GET /audit/monitor`) closed
unless `FRAUD_ADMIN_API_KEY` is set, which it is not (`docs/deployment.md`).

`GET /audit/monitor` returns aggregates only — counts, shares, PSIs, latency
quantiles, metrics — never a transaction, an identifier or a score, and it
counts a window before loading it (`monitor_max_rows`) so a wide window is
refused rather than served at the cost of the 0.1-CPU instance. The scheduled
monitoring job (ADR 0011) publishes those aggregates into a GitHub issue in a
public repository; that is the intended exposure and the reason the report
carries no row-level data.

Setting `FRAUD_API_KEY` closes scoring to key holders; the service reports
which mode it is in through `/health` (`auth`, `admin`).

## Not covered

Stated rather than implied:

- **The current champion's release assets are still mutable.** Release
  immutability is **enabled** on this repository (2026-09-18), so releases created
  from now on have locked assets, protected tags and a release attestation. It is
  not retroactive: every release published before that date — including
  `champion-7af85ec92813`, the one the live service actually fetches — remains
  replaceable by anyone with write access. The next promoted champion will be
  covered. **Decided (2026-09-18): this one is not being re-published to close
  it.** Doing so means deleting a release the running service fetches its weights
  from, which opens a window in which a cold start cannot start at all — spending
  a real chance of an outage to remove a risk the digest pin has already reduced
  to a failed startup, and that needs write access to this repository to reach.
  The next promotion closes it for nothing.
  The REST API exposes neither the repository flag nor an `immutable` field on
  existing releases, so the setting was verifiable only by publishing: `v0.8.0`
  came back `immutable=true`, `v0.7.4` did not. Either way
  the startup digest pin means a swapped asset fails startup rather than being
  loaded, so the residual risk is availability, not code execution.
- **The champion's sidecar files are not digest-pinned.** Only `model.joblib` is
  verified against the URL's `champion-<sha12>`. The manifest and the frozen
  golden are checked *for consistency with it*: a wrong `artifact_sha256`, wrong
  probabilities or a substituted model all fail startup. What they cannot do is
  change behaviour silently — see the bands row above; everything else they carry
  is reporting (`model_version`, `model_info`), so tampering with it is visible in
  `/model-info` and `/health` rather than acted on.
- **No secrets scanning configured by this repository.** GitHub's native push
  protection and secret scanning are on for public repositories; nothing here
  adds to them. No secret has ever been committed — the weights, the data and
  the tokens have always lived outside git.
- **No SBOM published** with releases, and no artifact signing of this project's
  own doing (Sigstore, `actions/attest-build-provenance`). Releases created since
  immutability was enabled carry GitHub's automatic *release* attestation, which
  records the tag, commit and assets — but nothing here produces or verifies one;
  the champion is verified by digest instead, and the image is built by the host
  from a public commit.
- **No container image scanning** (Trivy, Grype). Dependabot covers the base
  image tag and `pip-audit` covers the Python layer; what falls between them —
  OS packages in the base image — is unwatched.
- **No penetration testing, no fuzzing** of the request schemas beyond the
  validation tests.
- **Actions are pinned to tags, not commit SHAs.** A tag can be moved. This is
  a deliberate stop: the exposure that mattered was a third-party action on a
  floating branch in a job holding a deploy token, and that is fixed.

## Reporting

[`SECURITY.md`](../SECURITY.md) is the policy: GitHub **private vulnerability
reporting** is enabled (*Security* tab → *Report a vulnerability*) and is the
channel for anything exploitable. Public issues are right for everything else,
including the accepted limitations listed above. Nothing here processes real
payment data.
