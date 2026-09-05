# Synthetic time-series agent benchmark

A runnable research prototype comparing fixed statistical analysis, adaptive tool selection, and numerical verification on controlled synthetic time series. It produces observations, separate labels, traces, calibrated probability scores, costs, failures, paired intervals, and provenance hashes.

**The default agent is a deterministic adaptive tool policy, not an LLM.** An optional local Ollama controller is included. Default results cannot establish a benefit from LLM reasoning. This is not a reproduction of TS-Agent and does not claim publication novelty.

## Quick start

Python 3.11 or newer is required. From this directory:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m tsbench.cli --config configs/smoke.json --out runs/smoke
python -m tsbench.cli --config configs/benchmark.json --out runs/reference
```

Open `runs/reference/REPORT.md`. Each output directory must be new; existing results are never overwritten. The pinned dependency is NumPy 2.3.5. You can also install this package with `pip install .` and use the `tsbench` command.

The included current result is [evidence/reference_v0_1_1/REPORT.md](evidence/reference_v0_1_1/REPORT.md). The earlier `evidence/reference` directory preserves a v0.1.0 pilot and its sources. See [protocol history](docs/PROTOCOL.md#protocol-history) before comparing them.

## Comparison arms

| Arm | Controller | Numerical evidence |
|---|---|---|
| `fixed` | Fixed sequence | Quality, OLS trend, spectrum, residual outliers |
| `agent` | Adaptive scripted policy or local LLM | Same allowlist; adaptively requests diagnostics |
| `agent_verified` | Same exact initial agent result, then a fixed verifier | Completes robust diagnostics; selectively revises conflicts and averages agreeing probabilities |
| `fixed_full_tools` | Fixed sequence, full tool allowance | All seven numerical tools; controls for added evidence and computation |

The verified arm **reuses** the unverified arm's initial proposal and trace. It does not generate a fresh answer. Cached requests are recorded but not charged twice. Report the full-tool control alongside verification: an improvement over the agent alone may merely reflect better statistics.

## Tasks and controlled scenarios

Each series has four questions:

1. **Trend:** down, flat, or up according to the net change in the latent deterministic trend component.
2. **Seasonality:** whether a periodic or quasi-periodic component was generated.
3. **Anomaly:** whether at least one deliberately injected point spike remains observed. Natural heavy-tailed noise is not labeled as a deliberate injection.
4. **Missing block:** whether the observed missingness mask contains at least eight consecutive nulls. This is a deliberately easy sanity task.

The 36-cell factorial design crosses three trend directions, seasonality on/off, injected spikes on/off, and none/random/block missingness. `samples_per_family=72` repeats the factorial twice with independent noise and continuous parameters. Length is 128 by default. Random missingness has probability 0.15; block missingness removes 20 samples. Random missingness can occasionally create a long run, so the missing-block label is computed from the actual mask, not the generator category.

Development and calibration families use linear trends, Gaussian noise, and single/dual sinusoidal components. Test-only families introduce quadratic trends with chirps, piecewise trends with heavy-tailed noise, and amplitude-modulated seasonal signals with AR noise. This tests mechanism-family shift, not merely a fresh random split.

Generator constants and target semantics are versioned in `tsbench/generate.py`. Length, family membership, sample count, seed, backend, budgets, and bootstrap count are configurable. Changing generator constants constitutes a new benchmark version, not an invisible tuning step.

## Split and calibration protocol

- IDs and seeds incorporate split, family, and replicate. Development and calibration never share a generated realization.
- The runner rejects test-family overlap with either development or calibration.
- Controllers receive only the `Observation(values)` object. Family names, labels, latent parameters, IDs, and split names are not sent to the controller.
- One scalar temperature per arm is selected on calibration predictions using a fixed 49-point grid from 0.25 to 4. No threshold, model, or temperature is fitted on the test split.
- Source/config hashes are written before test evaluation. Calibration parameters are also written and hashed before test inference.
- Generated test labels are saved locally for audit. This is procedural isolation, **not** a secure evaluation server or hidden-label competition. Do not tune on the saved test outcomes.
- Development examples are generated for inspection; the reference run does not fit its heuristic thresholds on them. Thresholds are declared implementation choices.

## Outputs and metric definitions

`*_observations.jsonl` contains only ID and values. `*_labels.jsonl` contains evaluator-only family, split, labels, and generating parameters. Prediction files are evaluator artifacts and include labels, per-tool evidence, model actions, failures, and latency.

- **Accuracy:** mean correctness across four questions per scenario. Also report task-specific accuracy and all-four exact match.
- **Brier:** sum of squared class probability errors, averaged across questions (range 0–2; not divided by class count).
- **Log loss:** mean negative natural log probability assigned to the truth, clipped at 1e-12.
- **ECE:** ten equal-width bins of top-class confidence, conditional on valid predictions. Small-sample ECE is unstable; it is not a calibration guarantee.
- **Failures:** schema errors, insufficient observations, unknown tools, exhausted budgets, or local endpoint errors. Failures receive zero accuracy, Brier 2, and capped worst-case log loss. They are not silently dropped. ECE explicitly excludes them.
- **Coverage and selective accuracy:** accept questions at confidence >=0.8. Failures reduce coverage.
- **Cost:** unique numerical calls, declared relative tool weights, model request/token counts, and measured wall time. Latency includes bookkeeping and depends on hardware. Zero paid-API dollars does not mean zero local compute cost.
- **Paired intervals:** bootstrap whole scenarios within each named test family. Four dependent questions stay together. Intervals are exploratory, not multiplicity-adjusted, and conditional on those families.

`summary.json` includes raw/calibrated scores and per-family slices; `scores.csv` is a compact export; `manifest.json` records source, config, dataset, calibration, and semantic result hashes. The semantic hash excludes latency; floating-point results across different platforms/BLAS builds are not promised bitwise identical.

## Optional local LLM experiment

Install and run Ollama yourself with a suitable locally available model. This benchmark does not download models, start services, send private files, or call paid APIs.

```sh
python -m tsbench.cli --config configs/smoke.json --backend ollama --model YOUR_INSTALLED_MODEL_TAG --out runs/ollama_smoke
```

Only a loopback `http://localhost:11434/api/chat` endpoint is allowed. Redirects and environment proxies are disabled. The controller receives synthetic numeric observations, task definitions, and allowlisted tool outputs. It cannot execute model-generated Python or access files. Invalid JSON is a recorded failure, not automatically repaired. Requests are bounded by step count, token output, and timeout; keep the first run small.

The LLM uses JSON actions (`{"tool":"trend"}` or `{"prediction":...}`). Temperature zero and seeds reduce variation but do not guarantee bitwise model determinism. Archive exact model tag/digest, Ollama version, hardware, and prompts before making LLM comparisons. Local API integration is mock-tested; see the release notes for whether a real model was actually run.

## Limitations

- Only five generator families and four classification questions; no forecasting, anomaly localization, causal inference, multivariate data, or natural-language explanation grading.
- Latent trend/seasonality/injection labels can be difficult or impossible to infer from a finite noisy realization. This is an identifiability limitation, not necessarily agent failure in a broader sense.
- Linear interpolation can distort spectra and mask anomalies around long gaps. Robust fits use only observed residuals but operate on partly interpolated series for some diagnostics.
- The verifier and baselines share assumptions. Their errors are correlated, and verification can make answers worse. It is not an independent oracle or a proof.
- The full-tool control uses the same numerical evidence but does not match an LLM's inference FLOPs or wall time. Report tokens and latency in addition to tool units.
- Calibration data come from development families. Good in-distribution calibration need not transfer to held-out mechanisms.
- Synthetic series do not establish utility, privacy, fairness, or safety in real financial applications. Real-data transfer needs separate permission, evaluation, and governance.
- This package was implemented with AI assistance. Research ownership requires understanding, checking, and extending the implementation. Do not describe it as independently completed or peer-reviewed work.

## Research materials

- [Protocol and test coverage](docs/PROTOCOL.md)
- [Release checklist](docs/RELEASE.md)
- [Local verification](docs/VERIFICATION.md)

Personal portfolio reconciliation and interview notes are kept in the local
workspace only and are excluded from the benchmark release archive.

The related-paper connection is conceptual: [TS-Agent](https://arxiv.org/abs/2510.07432) uses analytical tools and evidence verification. Read its full method and benchmarks before making reproduction or novelty claims.

## Contributing

Add a family with explicit target semantics, then test generation, leakage isolation, and edge cases. Use a new config/version for changes motivated by published test results. Submit runnable evidence, not only a headline metric. CI is provided for Windows and Linux on Python 3.11 and 3.12; local execution does not prove hosted CI has run.

This is a research-prototype source distribution, not a validated production release.
No license has been selected; public source visibility does not itself grant an
open-source license. Repository-level CI is configured in
`.github/workflows/synthetic-ts-benchmark.yml` at the repository root. The nested
workflow is retained as a template for standalone use.
