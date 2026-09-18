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
| Python dependencies | `uv.lock` pins every version and hash; `security.yml` audits it weekly |
| GitHub Actions | pinned to releases — never a floating branch — and updated by Dependabot |
| base images | Dependabot watches both Dockerfiles |

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
endpoints (`POST /outcomes`, `GET /audit/recent`) closed unless
`FRAUD_ADMIN_API_KEY` is set, which it is not (`docs/deployment.md`).

Setting `FRAUD_API_KEY` closes scoring to key holders; the service reports
which mode it is in through `/health` (`auth`, `admin`).

## Not covered

Stated rather than implied:

- **No secrets scanning configured by this repository.** GitHub's native push
  protection and secret scanning are on for public repositories; nothing here
  adds to them. No secret has ever been committed — the weights, the data and
  the tokens have always lived outside git.
- **No SBOM published** with releases, and no artifact signing (Sigstore,
  attestations). The image is built by the host from a public commit; the
  champion is verified by digest instead.
- **No container image scanning** (Trivy, Grype). Dependabot covers the base
  image tag and `pip-audit` covers the Python layer; what falls between them —
  OS packages in the base image — is unwatched.
- **No penetration testing, no fuzzing** of the request schemas beyond the
  validation tests.
- **Actions are pinned to tags, not commit SHAs.** A tag can be moved. This is
  a deliberate stop: the exposure that mattered was a third-party action on a
  floating branch in a job holding a deploy token, and that is fixed.

## Reporting

Open an issue on the repository. There is no private disclosure channel and
nothing here processes real payment data.
