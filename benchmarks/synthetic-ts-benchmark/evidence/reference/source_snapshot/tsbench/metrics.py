"""Calibration on a separate split and scenario-clustered paired bootstrap."""
import math
import numpy as np
from .schema import CLASSES, validate_prediction

def temperature_scale(pred, temperature):
    if not math.isfinite(temperature) or temperature <= 0: raise ValueError("temperature must be positive")
    validate_prediction(pred)
    result={}
    for task,classes in CLASSES.items():
        z=np.log(np.clip([pred[task][c] for c in classes],1e-12,1))/temperature
        weights=np.exp(z-z.max());weights/=weights.sum()
        result[task]={c:float(v) for c,v in zip(classes,weights)}
    return result

def nll(pred, labels):
    return float(np.mean([-math.log(max(pred[t][labels[t]],1e-12)) for t in CLASSES]))

def fit_temperature(records):
    if any(r["split"]!="calibration" for r in records):
        raise ValueError("only calibration records may fit temperature")
    valid=[r for r in records if r["prediction"] is not None and not r["failure"]]
    if not valid: return {"temperature":1.,"n_valid":0,"status":"no_valid_calibration"}
    grid=np.exp(np.linspace(np.log(.25),np.log(4),49))
    losses=[np.mean([nll(temperature_scale(r["prediction"],t),r["labels"]) for r in valid]) for t in grid]
    best=int(np.argmin(losses))
    return {"temperature":float(grid[best]),"n_valid":len(valid),"status":"fit",
            "calibration_nll":float(losses[best]),"grid_min":.25,"grid_max":4.}

def task_scores(record,temperature=1):
    failed=record["failure"] is not None or record["prediction"] is None
    pred=None if failed else temperature_scale(record["prediction"],temperature)
    scores=[]
    for task,classes in CLASSES.items():
        if failed:
            # Maximum multiclass Brier and capped log loss; failures never get lucky labels.
            scores.append({"task":task,"correct":0.,"brier":2.,"nll":-math.log(1e-12),
                           "confidence":None,"predicted":None})
        else:
            probs=pred[task];chosen=max(classes,key=lambda c:probs[c]);truth=record["labels"][task]
            scores.append({"task":task,"correct":float(chosen==truth),
                           "brier":sum((probs[c]-float(c==truth))**2 for c in classes),
                           "nll":-math.log(max(probs[truth],1e-12)),
                           "confidence":probs[chosen],"predicted":chosen})
    return scores

def ece(scores,bins=10):
    valid=[s for s in scores if s["confidence"] is not None]
    if not valid:return None
    groups=[[] for _ in range(bins)]
    for s in valid:groups[min(bins-1,int(s["confidence"]*bins))].append(s)
    return sum(len(g)/len(valid)*abs(np.mean([s["confidence"] for s in g])-
                np.mean([s["correct"] for s in g])) for g in groups if g)

def summarize(records,temperature=1):
    if not records:raise ValueError("cannot score empty records")
    scored=[task_scores(r,temperature) for r in records];flat=[s for row in scored for s in row]
    resource_calls=[sum(not t["cached"] for t in r["trace"]) for r in records]
    resource_units=[sum(t["units"] for t in r["trace"]) for r in records]
    elapsed=[r.get("elapsed_ms",0) for r in records]
    valid=[s for s in flat if s["confidence"] is not None]
    accepted=[s for s in valid if s["confidence"]>=.8]
    failures={}
    for r in records:
        if r["failure"]:
            key=r["failure"]["reason"];failures[key]=failures.get(key,0)+1
    result={"n_scenarios":len(records),"n_questions":len(flat),
            "accuracy":float(np.mean([s["correct"] for s in flat])),
            "exact_match":float(np.mean([all(s["correct"] for s in row) for row in scored])),
            "brier":float(np.mean([s["brier"] for s in flat])),
            "log_loss":float(np.mean([s["nll"] for s in flat])),
            "ece_successful_only":float(ece(flat)) if valid else None,
            "failure_rate":sum(bool(r["failure"]) for r in records)/len(records),
            "failure_reasons":failures,
            "coverage_at_0_8":len(accepted)/len(flat),
            "selective_accuracy_at_0_8":float(np.mean([s["correct"] for s in accepted])) if accepted else None,
            "tool_calls_mean":float(np.mean(resource_calls)),"tool_calls_median":float(np.median(resource_calls)),
            "tool_units_mean":float(np.mean(resource_units)),"tool_units_median":float(np.median(resource_units)),
            "elapsed_ms_mean":float(np.mean(elapsed)),"elapsed_ms_median":float(np.median(elapsed)),
            "elapsed_ms_p95":float(np.percentile(elapsed,95)),
            "input_tokens":sum(r["input_tokens"] for r in records),
            "output_tokens":sum(r["output_tokens"] for r in records),
            "model_requests":sum(len(r["model_calls"]) for r in records),
            "api_dollars":0.,"local_compute_dollars":None,"tasks":{}}
    for task in CLASSES:
        rows=[s for s in flat if s["task"]==task]
        result["tasks"][task]={"accuracy":float(np.mean([s["correct"] for s in rows])),
                                "brier":float(np.mean([s["brier"] for s in rows])),
                                "ece_successful_only":ece(rows)}
    return result

def paired_bootstrap(left,right,replicates=1000,seed=0):
    # Stratify by named family; resample whole scenarios, not their four dependent tasks.
    left={r["id"]:r for r in left};right={r["id"]:r for r in right}
    if not left or left.keys()!=right.keys():raise ValueError("paired IDs must match")
    groups={}
    for key,a in left.items():
        b=right[key]
        if a["labels"]!=b["labels"] or a["family"]!=b["family"]:raise ValueError("pair metadata mismatch")
        delta=np.mean([s["correct"] for s in task_scores(a)])-np.mean([s["correct"] for s in task_scores(b)])
        groups.setdefault(a["family"],[]).append(delta)
    rng=np.random.default_rng(seed);samples=np.zeros(replicates)
    for values in groups.values():
        v=np.asarray(values);samples+=rng.choice(v,size=(replicates,len(v)),replace=True).sum(axis=1)/len(left)
    all_values=[x for v in groups.values() for x in v]
    return {"mean_accuracy_difference":float(np.mean(all_values)),
            "ci95":[float(v) for v in np.quantile(samples,[.025,.975])],
            "n_pairs":len(left),"replicates":replicates,
            "scope":"scenario uncertainty conditional on these named families; exploratory, not multiplicity adjusted"}
