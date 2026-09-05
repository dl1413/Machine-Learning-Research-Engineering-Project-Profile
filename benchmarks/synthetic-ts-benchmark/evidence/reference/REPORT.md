# Synthetic time-series benchmark results

Backend: **scripted**. Version 0.1.0.

This is a local reference experiment, not a TS-Agent reproduction or evidence of LLM superiority.

## Held-out family results

All four tasks count equally. Failures count as incorrect and receive worst-case Brier/log-loss penalties.

| Method | Accuracy | Exact match | Brier calibrated | ECE successful only | Failure rate | Mean tool units |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0.896 | 0.681 | 0.146 | 0.045 | 0.000 | 9.00 |
| agent | 0.922 | 0.764 | 0.133 | 0.059 | 0.000 | 16.43 |
| agent_verified | 0.958 | 0.843 | 0.080 | 0.037 | 0.000 | 30.00 |
| fixed_full_tools | 0.958 | 0.843 | 0.075 | 0.032 | 0.000 | 30.00 |

## Paired accuracy differences

Intervals resample whole scenarios within each held-out family; they do not estimate uncertainty over new families.

- agent minus fixed: +0.027, 95% interval [+0.014, +0.039].
- agent_verified minus agent: +0.036, 95% interval [+0.023, +0.049].
- agent_verified minus fixed_full_tools: +0.000, 95% interval [+0.000, +0.000].

## Per task accuracy

| Method | Trend | Seasonal | Injected anomaly | Missing block |
|---|---:|---:|---:|---:|
| fixed | 1.000 | 0.829 | 0.755 | 1.000 |
| agent | 1.000 | 0.829 | 0.861 | 1.000 |
| agent_verified | 1.000 | 0.926 | 0.907 | 1.000 |
| fixed_full_tools | 1.000 | 0.926 | 0.907 | 1.000 |

## Interpretation boundaries

- The default agent is a deterministic adaptive policy, not an LLM. The local LLM backend must be run separately.
- Verification is a fixed numerical cross-check, not a correctness certificate. It can reduce accuracy.
- Compare agent_verified against fixed_full_tools as well as agent. The full-tool control shares the available numerical evidence.
- Temperature scaling uses only calibration families. It is not guaranteed to remain calibrated under shift.
- Synthetic labels concern latent mechanisms. Natural heavy-tailed noise may be observationally indistinguishable from an injected spike.
- Missing-block detection is an easy sanity task; inspect the other task scores instead of headline accuracy alone.
- No real financial data, forecasting task, publication acceptance, human evaluation, or broad safety validation is included.
- See summary.json for raw and calibrated metrics, family slices, failures, latency, tokens, and selective accuracy.
- Tool units are declared relative weights, not FLOPs or dollars. Zero paid-API cost does not imply free local compute.

## Provenance

- Config SHA256: `252a79eb472f842a5a416171494c896c702b89d473afc3e6ad3c48a94883cba1`
- Source SHA256: `2ec385220b911fb288a96523c114b9df487e24c1fd0fbf31688cfc32af7ad3ba`
- Semantic result SHA256: `31c2e665866bfdb37f4f76dc0e7d80b1817433773ea7d1f8ef939f42b9a86e23`

See manifest.json for Python/NumPy versions and all generated dataset hashes.
