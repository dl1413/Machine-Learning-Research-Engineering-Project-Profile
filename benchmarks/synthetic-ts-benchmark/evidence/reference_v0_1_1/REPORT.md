# Synthetic time-series benchmark results

Backend: **scripted**. Version 0.1.1.

This is a local reference experiment, not a TS-Agent reproduction or evidence of LLM superiority.

## Held-out family results

All four tasks count equally. Failures count as incorrect and receive worst-case Brier/log-loss penalties.

| Method | Accuracy | Exact match | Brier calibrated | ECE successful only | Failure rate | Mean tool units |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0.895 | 0.681 | 0.136 | 0.038 | 0.000 | 9.00 |
| agent | 0.924 | 0.741 | 0.114 | 0.053 | 0.000 | 16.45 |
| agent_verified | 0.924 | 0.741 | 0.101 | 0.048 | 0.000 | 30.00 |
| fixed_full_tools | 0.957 | 0.829 | 0.073 | 0.023 | 0.000 | 30.00 |

## Paired accuracy differences

Intervals resample whole scenarios within each held-out family; they do not estimate uncertainty over new families.

- agent minus fixed: +0.029, 95% interval [+0.016, +0.043].
- agent_verified minus agent: +0.000, 95% interval [+0.000, +0.000].
- agent_verified minus fixed_full_tools: -0.034, 95% interval [-0.047, -0.020].

## Per task accuracy

| Method | Trend | Seasonal | Injected anomaly | Missing block |
|---|---:|---:|---:|---:|
| fixed | 0.995 | 0.815 | 0.769 | 1.000 |
| agent | 0.995 | 0.815 | 0.884 | 1.000 |
| agent_verified | 0.995 | 0.815 | 0.884 | 1.000 |
| fixed_full_tools | 0.995 | 0.921 | 0.912 | 1.000 |

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

- Config SHA256: `20cc7f83f22d531f27e405d6795b71ab8c00dca91ccd972a5f848a18ae4ab4f7`
- Source SHA256: `450e2fd43ffd3dd5c8e34e3bfa168b15b8826cf83bf587fb60075018be1dde19`
- Semantic result SHA256: `f54a070bf918217971dab5a419bb52f3cdc16d98f1b1987dd42e1b8feb482545`

See manifest.json for Python/NumPy versions and all generated dataset hashes.
