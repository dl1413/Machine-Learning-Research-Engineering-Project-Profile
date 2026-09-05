# Evaluation protocol and test plan

## Research question

Does adaptive numerical tool selection improve four synthetic time-series classification tasks over a fixed workflow, and does a fixed verification pass improve the *same* agent proposal? Does any gain remain when compared with a fixed full-tool control?

This reference implementation tests the harness and a deterministic adaptive policy. A separate local-model experiment is required to address LLM reasoning. A negative comparison is an acceptable result.

## Frozen choices

- Primary outcome: paired difference in mean per-scenario question accuracy.
- Comparisons: agent minus fixed; verified agent minus agent; verified agent minus fixed full tools.
- Secondary outcomes: task accuracy, exact match, raw/calibrated Brier, log loss, ECE, failure rate, confidence-threshold coverage, tool cost, model tokens, and latency.
- Default design: 72 scenarios per family, two development/calibration families and three held-out test families. There are 216 test scenarios and 864 test questions per arm.
- Per-series variation: trend sign, seasonal presence, spike injection, missingness mechanism, amplitudes, phases, periods, noise, and missing indices.
- Fixed numerical tool budget: at least seven unique tools for all arms. Same tool definitions and nominal budget, different actual usage. The full-tool arm consumes all seven.
- Scripted-controller thresholds are heuristics specified in source, not fitted estimators. Probability outputs are scores before temperature scaling.
- Temperature fit is global per arm, not per task or test family. This intentionally simple calibrator can leave task-specific calibration errors.
- Bootstrap: 1,000 resamples within each family. Reuse paired scenarios. Do not interpret the interval as generalization to all possible generators.

## Data flow and leakage boundary

```text
seed and generator configuration
  ├─ observations ──> restricted numerical tools ──> controller ──> probabilities
  └─ labels and family metadata ─────────────────────────────────> evaluator
calibration probabilities and labels ──> frozen temperatures
test probabilities and labels + frozen temperatures ──> metrics
```

The API boundary prevents accidental label injection into the supplied controller. The same local process owns both sides, so a malicious custom controller could inspect source or files. For an adversarial evaluation, put inference in a separate process/container with only observations mounted and keep labels on a different service. That isolation is not implemented here.

## Verification definition

Complete quality/basic diagnostics and robust trend, robust residual outliers, and split-half spectral checks. Derive a second numerical prediction. On disagreement, replace the task distribution only if the evidence's top probability is at least 0.85 and the original proposal's top probability is below 0.8. Otherwise retain the original top label but average its distribution with uniform probabilities to express uncertainty. On agreement, average the original and evidence distributions. Archive both distributions, disagreement names, and changed task names. The gates use raw probabilities before post-hoc temperature scaling; their reliability is itself a limitation.

This is a deliberate, inspectable correction rule, not a formal validator. Its main falsification test is whether it underperforms the original answer or the full-tool fixed workflow. Because it shares numerical assumptions, agreement does not prove validity.

## Protocol history

Version 0.1.0 was run locally as a pilot on seed 20260905. It replaced every disagreement with the full-tool evidence label, making verified-agent and full-tool-control classification identical by construction. It therefore could not meaningfully test whether selective verification adds value beyond that fixed workflow.

Version 0.1.1 corrects that design with the explicit confidence gates above. The gates were selected as design constants, not optimized by a parameter search. A fresh seed, 20260906, gives new realizations, but uses the same already-inspected family definitions. The revised result remains exploratory, not an untouched confirmatory evaluation. The pilot outputs and exact sources are preserved under `evidence/reference`; the corrected result is under `evidence/reference_v0_1_1`.

Future changes informed by these results need another version and newly reserved evaluation families or an external evaluation set before confirmatory claims.

## Test coverage

| Area | Tests | Release expectation |
|---|---|---|
| Generator | Seed repeatability, independent streams, factorial controls, mask/injection targets | Deterministic fixtures and finite JSON |
| Leakage | Public schema excludes labels/family; test-family overlap rejected | No metadata sent through controller interface |
| Numerical tools | Known trend/period, missing interpolation, constant/all-missing cases | Finite or explicit failure, never silent NaN |
| Tool boundary | Allowlist, budget, caching, mutation protection | No arbitrary code or filesystem tool |
| Controller | All arms, exact base proposal reuse, invalid actions, step/timeout failures | Valid schema or recorded failure |
| Calibration | Calibration-only fit, probability validity, ranking, empty calibration | No fitting on test records |
| Scoring | Known Brier, perfect predictions, explicit failure penalties, pair matching | Independently checkable definitions |
| End to end | Artifact inventory, no overwrite, repeated semantic hashes | Same-platform deterministic reproduction |

## Known testing gaps

Real local-LLM execution requires an installed model and has not been demonstrated merely by mocked tests. Hosted cross-platform CI must run after repository publication. Large-series scaling, adversarial process isolation, multiple model seeds, and real financial transfer are future tests. Unit tests are not empirical validation of a research hypothesis.
