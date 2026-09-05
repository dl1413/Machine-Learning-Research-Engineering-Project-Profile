# Local verification, 2026-09-05

The v0.1.1 reference configuration was run twice on Windows, Python 3.12.14,
NumPy 2.3.5. Dataset hashes, source hashes, calibration parameters and semantic
result hashes matched. Timing is deliberately excluded from the semantic hash.

Semantic result SHA256:
`f54a070bf918217971dab5a419bb52f3cdc16d98f1b1987dd42e1b8feb482545`

All 42 unit, contract and end-to-end tests passed locally. Windows/Linux CI configuration is supplied but
hosted CI has not been run. Tests of the local LLM interface use mocks; the
reference experiment uses a scripted adaptive agent, not a language model.

The archive includes the earlier pilot with its original source snapshot for
provenance. The revised experiment is exploratory: changing seeds does not undo
prior inspection of the same scenario families. It is not confirmatory evidence
about previously unseen families or evidence of LLM performance.

The packaging script excludes local interview preparation and portfolio-audit
notes. No license has been selected. Public distribution is limited to the
benchmark sources and evidence; private preparation materials are excluded.
