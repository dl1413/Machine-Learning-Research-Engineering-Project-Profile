"""Public task schema: controllers receive observations, never generator labels."""
from dataclasses import dataclass
import math

CLASSES = {
    "trend": ("down", "flat", "up"),
    "seasonal": ("no", "yes"),
    "anomaly": ("no", "yes"),
    "missing_block": ("no", "yes"),
}
TASK_DESCRIPTION = (
    "Classify the underlying trend component's net change (down/flat/up); "
    "presence of a periodic or quasi-periodic component (seasonal no/yes); "
    "presence of deliberately injected observed point spikes (anomaly no/yes); "
    "and whether the observed mask has a run of at least 8 missing values "
    "(missing_block no/yes). Heavy-tailed natural noise is NOT an injected spike. "
    "Return a probability for each class, summing to one per task. "
    "Missing samples are null, not zero. Generator metadata is unavailable."
)

@dataclass(frozen=True)
class Observation:
    values: tuple

    def __post_init__(self):
        if len(self.values) < 32 or len(self.values) > 1024:
            raise ValueError("length must be in [32, 1024]")
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v)) for v in self.values):
            raise ValueError("observations require finite numbers or null")

def validate_prediction(pred):
    if not isinstance(pred, dict) or set(pred) != set(CLASSES):
        raise ValueError("prediction task keys do not match schema")
    for task, classes in CLASSES.items():
        probs = pred[task]
        if not isinstance(probs, dict) or set(probs) != set(classes):
            raise ValueError("prediction class keys do not match schema")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
            raise ValueError("invalid probability")
        if abs(sum(probs.values()) - 1) > 1e-6:
            raise ValueError("probabilities must sum to one")
    return pred

def uniform_prediction():
    return {t: {c: 1 / len(cs) for c in cs} for t, cs in CLASSES.items()}

def longest_run(mask):
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0
        best = max(best, run)
    return best
