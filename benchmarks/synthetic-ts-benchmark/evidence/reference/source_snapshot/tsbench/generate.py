"""Controlled generators with independent seeded streams for each split/family."""
import hashlib
import json
import numpy as np
from .schema import Observation, longest_run

FAMILIES = {"linear_sine", "linear_dual", "curved_chirp", "piecewise_heavytail", "amplitude_ar"}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()

def validate_config(cfg):
    for key in ("seed", "length", "samples_per_family", "bootstrap_replicates"):
        if type(cfg.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    if not 64 <= cfg["length"] <= 512 or cfg["samples_per_family"] < 3:
        raise ValueError("length must be 64..512 and samples_per_family >=3")
    if not 20 <= cfg["bootstrap_replicates"] <= 10000:
        raise ValueError("bootstrap_replicates must be 20..10000")
    splits = cfg["splits"]
    if set(splits) != {"development", "calibration", "test"}:
        raise ValueError("exactly development, calibration, test splits required")
    for fs in splits.values():
        if not fs or len(fs) != len(set(fs)) or not set(fs) <= FAMILIES:
            raise ValueError("invalid or duplicate family")
    if set(splits["test"]) & (set(splits["development"]) | set(splits["calibration"])):
        raise ValueError("test families overlap development/calibration")
    if not set(splits["calibration"]) <= set(splits["development"]):
        raise ValueError("calibration must use development families")
    agent = cfg["agent"]
    if agent["backend"] not in ("scripted", "ollama"):
        raise ValueError("unknown agent backend")
    if agent["backend"] == "ollama" and not agent.get("model"):
        raise ValueError("an explicit locally installed model is required")
    if type(agent["max_tools"]) is not int or not 7 <= agent["max_tools"] <= 20:
        raise ValueError("max_tools must be 7..20 to support all comparison arms")
    if type(agent["max_steps"]) is not int or not 1 <= agent["max_steps"] <= 20:
        raise ValueError("max_steps must be 1..20")
    if not 1 <= agent["timeout_seconds"] <= 60:
        raise ValueError("timeout_seconds must be 1..60")
    return cfg

def generate_case(seed, split, family, index, length):
    key = f"{seed}:{split}:{family}:{index}"
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    t = np.linspace(0, 1, length)
    sign = (-1, 0, 1)[index % 3]
    magnitude = float(rng.uniform(2.5, 6))
    trend = sign * magnitude * t
    periodic = bool((index // 3) % 2)
    injected = bool((index // 6) % 2)
    missing_mode = ("none", "random", "block")[(index // 12) % 3]
    period = float(rng.uniform(10, 28))
    phase = float(rng.uniform(0, 2*np.pi))
    angle = 2*np.pi*np.arange(length)/period + phase
    amplitude = float(rng.uniform(1.2, 2.4))
    seasonal = amplitude*np.sin(angle)
    noise = rng.normal(0, .35, length)
    if family == "linear_dual":
        seasonal += .5*amplitude*np.sin(2*angle+.7)
    elif family == "curved_chirp":
        trend = sign*magnitude*t**2
        seasonal = amplitude*np.sin(angle + 2*np.pi*2*t**2)
    elif family == "piecewise_heavytail":
        trend = sign*magnitude*np.maximum(0, (t-.4)/.6)
        noise = rng.standard_t(3, length)*.35/np.sqrt(3)
    elif family == "amplitude_ar":
        seasonal *= .3+1.4*t
        for j in range(1, length):
            noise[j] += .7*noise[j-1]
    elif family != "linear_sine":
        raise ValueError("unknown family")
    y = trend + (seasonal if periodic else 0) + noise
    anomaly_indices = []
    if injected:
        anomaly_indices = sorted(rng.choice(np.arange(4,length-4), size=2, replace=False).tolist())
        for i in anomaly_indices:
            y[i] += float(rng.choice([-1, 1])*rng.uniform(4,7))
    mask = np.zeros(length, dtype=bool)
    if missing_mode == "random":
        mask = rng.random(length) < .15
    elif missing_mode == "block":
        start = int(rng.integers(4,length-25)); mask[start:start+20] = True
    # Anomaly label concerns observed injected events, not unobservable hidden spikes.
    observed_injections = [i for i in anomaly_indices if not mask[i]]
    values = tuple(None if mask[i] else float(y[i]) for i in range(length))
    observation = Observation(values)
    labels = {"trend": { -1:"down", 0:"flat", 1:"up"}[sign],
              "seasonal": "yes" if periodic else "no",
              "anomaly": "yes" if observed_injections else "no",
              "missing_block": "yes" if longest_run(mask) >= 8 else "no"}
    return {"id": hashlib.sha256(key.encode()).hexdigest()[:20],
            "split": split, "family": family, "observation": observation,
            "labels": labels, "metadata": {"period":period, "amplitude":amplitude,
            "trend_magnitude":magnitude if sign else 0, "missing_mode":missing_mode,
            "injected_indices":anomaly_indices, "observed_injections":observed_injections}}

def generate_split(cfg, split):
    return [generate_case(cfg["seed"],split,f,i,cfg["length"])
            for f in cfg["splits"][split] for i in range(cfg["samples_per_family"])]

def public_record(case):
    return {"id":case["id"], "values":case["observation"].values}

def private_record(case):
    return {k:v for k,v in case.items() if k != "observation"}
