# Security policy

This is a portfolio project: a fraud risk scoring service trained on the public
Kaggle IEEE-CIS dataset, with a free public demo. It processes **no real payment
data and no personal data** — the dataset is anonymised by its provider, and the
demo scores whatever a visitor submits. Nothing here is a production payment
system, and the public deployment is documented as a demo rather than a service.

The threat model, what protects each link in the supply chain, and what is
deliberately not covered are in [`docs/security.md`](docs/security.md).

## Supported versions

Only the latest release. The public demo tracks it, and fixes are made on `main`
and shipped in the next tag rather than backported.

| version | supported |
|---|---|
| latest release | yes |
| anything earlier | no |

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting**: the *Security* tab →
*Report a vulnerability*. That opens a private advisory visible only to the
maintainer, which is the right channel for anything exploitable.

Please do not open a public issue for an exploitable finding. Public issues are
the right place for everything else, including the accepted limitations below.

Expect a reply within a week. This is a personal project maintained in spare
time, so there is no guaranteed response window beyond that.

## What is in scope

- The scoring service (`src/fraud/serve/`) and its public deployment.
- The model supply chain: how the champion artifact is fetched, verified and
  loaded (`model.joblib` is a pickle, so this is the sharp edge — it is
  digest-pinned before deserialization).
- The release and deployment pipeline (`scripts/release.py`,
  `.github/workflows/`).

## What is already known and accepted

These are documented decisions, not undiscovered problems. Reports of them are
welcome as issues, but they are not vulnerabilities:

- **The demo runs with no API key.** A demo that needs a key is not a demo.
  Rate limits, an upload cap, a row ceiling and a request time budget bound it;
  the admin endpoints are closed.
- **The demo's state is ephemeral.** Free hosting resets the audit trail and the
  review-budget history on restart. Durable storage is out of scope.
- **Release assets published before 2026-09-18 are mutable.** Immutability is
  enabled but is not retroactive. The digest pin means a swapped artifact fails
  startup rather than executing.
- **No SBOM, artifact signing or container image scanning.** Listed with the
  rest in [`docs/security.md`](docs/security.md).
