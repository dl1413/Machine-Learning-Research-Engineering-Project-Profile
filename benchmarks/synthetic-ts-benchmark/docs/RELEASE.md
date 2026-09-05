# Research-prototype release checklist

## Included

- Python package, pinned numerical dependency, two JSON configurations.
- Four comparison arms including a deterministic tool agent and optional local LLM controller.
- Unit/contract/end-to-end tests and Windows/Linux CI configuration.
- Seeded observations, separate evaluator labels, traces, metrics, and provenance produced by the runner.
- Explicit limitations and local verification record. Personal portfolio reconciliation and interview notes remain outside the release archive.

## Before sharing publicly

1. Independently rerun the tests and reference configuration in a clean environment.
2. Inspect the generated report and representative failure traces; make no favorable-result assumptions.
3. Confirm author/contributor wording and select a license. No license has been chosen for the user.
4. If claiming LLM results, run a real locally installed model and archive its digest, version, seeds, and hardware. Mock tests are insufficient.
5. Choose the target repository and review all public-facing claims. Keep old résumé reports unchanged until their underlying evidence is reconciled.
6. Publish a branch or release only after the target and distribution scope are agreed. Hosted CI is not verified until it actually runs.

## Current limits

The package is a research prototype prepared for a dedicated GitHub branch.
Real local-model inference, real-data validation, an open-source license, and
peer-reviewed acceptance remain uncompleted. Hosted CI status must be checked
on the pushed commit; local tests do not establish hosted success.

## Distribution and rollback

Only benchmark sources, configurations, tests, synthetic evidence and supporting
documentation are included. Private interview and portfolio-audit notes are
excluded. No production service or database is deployed. The default branch
is unchanged by publishing this feature branch. If a later merge causes a
regression or evidence-integrity failure, revert that merge with a normal commit;
do not rewrite published history.
