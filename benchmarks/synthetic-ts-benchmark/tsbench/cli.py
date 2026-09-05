"""Single-command local reproduction with split artifacts and source provenance."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
from pathlib import Path
import sys
import time
import numpy as np
from . import __version__
from .agents import METHODS, run_method, verified_from
from .generate import validate_config,generate_split,public_record,private_record,digest
from .metrics import fit_temperature,summarize,paired_bootstrap

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

def write_jsonl(path,records):
    with path.open("w",encoding="utf-8") as stream:
        for record in records: stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")

def evaluate_cases(cases,cfg):
    records=[]
    for case in cases:
        agent_base=None;agent_elapsed=0
        for method in METHODS:
            start=time.perf_counter()
            if method=="agent_verified":
                result=verified_from(agent_base,case["observation"],cfg["agent"])
                elapsed=agent_elapsed+(time.perf_counter()-start)*1000
            else:
                result=run_method(case["observation"],method,cfg["agent"],int(case["id"][:8],16)%2**31)
                elapsed=(time.perf_counter()-start)*1000
            if method=="agent":agent_base=copy.deepcopy(result);agent_elapsed=elapsed
            records.append({"id":case["id"],"family":case["family"],"split":case["split"],
                            "labels":case["labels"],"method":method,"elapsed_ms":elapsed,**result})
    return records

def semantic_records(records):
    result=copy.deepcopy(records)
    for r in result:
        r.pop("elapsed_ms",None)
        for event in r["trace"]:event.pop("latency_ms",None)
    return result

def render_report(out,summary,manifest):
    lines=["# Synthetic time-series benchmark results","",
           f"Backend: **{manifest['backend']}**. Version {__version__}.","",
           "This is a local reference experiment, not a TS-Agent reproduction or evidence of LLM superiority.","",
           "## Held-out family results","",
           "All four tasks count equally. Failures count as incorrect and receive worst-case Brier/log-loss penalties.","",
           "| Method | Accuracy | Exact match | Brier calibrated | ECE successful only | Failure rate | Mean tool units |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        s=summary["methods"][method]["calibrated"]
        e="NA" if s["ece_successful_only"] is None else f"{s['ece_successful_only']:.3f}"
        lines.append(f"| {method} | {s['accuracy']:.3f} | {s['exact_match']:.3f} | {s['brier']:.3f} | {e} | {s['failure_rate']:.3f} | {s['tool_units_mean']:.2f} |")
    lines += ["","## Paired accuracy differences","",
              "Intervals resample whole scenarios within each held-out family; they do not estimate uncertainty over new families.",""]
    for name,s in summary["comparisons"].items():
        lines.append(f"- {name}: {s['mean_accuracy_difference']:+.3f}, 95% interval [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}].")
    lines += ["","## Per task accuracy","","| Method | Trend | Seasonal | Injected anomaly | Missing block |",
              "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        tasks=summary["methods"][method]["raw"]["tasks"]
        lines.append("| "+method+" | "+" | ".join(f"{tasks[t]['accuracy']:.3f}" for t in tasks)+" |")
    lines += ["","## Interpretation boundaries","",
              "- The default agent is a deterministic adaptive policy, not an LLM. The local LLM backend must be run separately.",
              "- Verification is a fixed numerical cross-check, not a correctness certificate. It can reduce accuracy.",
              "- Compare agent_verified against fixed_full_tools as well as agent. The full-tool control shares the available numerical evidence.",
              "- Temperature scaling uses only calibration families. It is not guaranteed to remain calibrated under shift.",
              "- Synthetic labels concern latent mechanisms. Natural heavy-tailed noise may be observationally indistinguishable from an injected spike.",
              "- Missing-block detection is an easy sanity task; inspect the other task scores instead of headline accuracy alone.",
              "- No real financial data, forecasting task, publication acceptance, human evaluation, or broad safety validation is included.",
              "- See summary.json for raw and calibrated metrics, family slices, failures, latency, tokens, and selective accuracy.",
              "- Tool units are declared relative weights, not FLOPs or dollars. Zero paid-API cost does not imply free local compute.","",
              "## Provenance","",f"- Config SHA256: `{manifest['config_sha256']}`",
              f"- Source SHA256: `{manifest['source_sha256']}`",
              f"- Semantic result SHA256: `{manifest['semantic_result_sha256']}`","",
              "See manifest.json for Python/NumPy versions and all generated dataset hashes."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def run(cfg,out):
    validate_config(cfg)
    out=Path(out)
    if out.exists():raise FileExistsError("output already exists; choose a new run directory")
    out.mkdir(parents=True)
    source_dir=Path(__file__).parent
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob("*.py"))}
    manifest={"version":__version__,"python":platform.python_version(),"numpy":np.__version__,
              "platform":platform.platform(),"backend":cfg["agent"]["backend"],"model":cfg["agent"]["model"],
              "config_sha256":digest(cfg),"source_sha256":digest(sources),"source_files":sources,
              "status":"started","dataset_hashes":{},"protocol":"v1.1 selective verifier frozen before test evaluation"}
    shutil.copytree(source_dir,out/"source_snapshot"/"tsbench",ignore=shutil.ignore_patterns("__pycache__"))
    write_json(out/"config.json",cfg);write_json(out/"manifest.json",manifest)
    datasets={split:generate_split(cfg,split) for split in cfg["splits"]}
    seen=set()
    for split,cases in datasets.items():
        ids={c["id"] for c in cases}
        if len(ids)!=len(cases) or seen & ids:raise ValueError("duplicate scenario IDs")
        seen |= ids
        public=[public_record(c) for c in cases];labels=[private_record(c) for c in cases]
        write_jsonl(out/f"{split}_observations.jsonl",public)
        write_jsonl(out/f"{split}_labels.jsonl",labels)
        manifest["dataset_hashes"][split]={"observations":digest(public),"labels":digest(labels),"count":len(cases)}
    calibration=evaluate_cases(datasets["calibration"],cfg)
    write_jsonl(out/"calibration_predictions.jsonl",calibration)
    temperatures={m:fit_temperature([r for r in calibration if r["method"]==m]) for m in METHODS}
    write_json(out/"calibration.json",temperatures)
    manifest["calibration_sha256"]=digest(temperatures)
    manifest["status"]="calibration_frozen_before_test";write_json(out/"manifest.json",manifest)
    test=evaluate_cases(datasets["test"],cfg);write_jsonl(out/"test_predictions.jsonl",test)
    summary={"methods":{},"comparisons":{}}
    for method in METHODS:
        rs=[r for r in test if r["method"]==method];temp=temperatures[method]["temperature"]
        summary["methods"][method]={"calibration":temperatures[method],"raw":summarize(rs),
             "calibrated":summarize(rs,temp),"families":{f:summarize([r for r in rs if r["family"]==f],temp)
             for f in cfg["splits"]["test"]}}
    for left,right in (("agent","fixed"),("agent_verified","agent"),("agent_verified","fixed_full_tools")):
        summary["comparisons"][f"{left} minus {right}"]=paired_bootstrap(
            [r for r in test if r["method"]==left],[r for r in test if r["method"]==right],
            cfg["bootstrap_replicates"],cfg["seed"])
    write_json(out/"summary.json",summary)
    with (out/"scores.csv").open("w",newline="",encoding="utf-8") as stream:
        fields=["method","accuracy","exact_match","brier","log_loss","failure_rate","tool_units_mean"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for method in METHODS:
            s=summary["methods"][method]["calibrated"]
            writer.writerow({"method":method,**{k:s[k] for k in fields if k!="method"}})
    manifest["semantic_result_sha256"]=digest(semantic_records(test))
    manifest["status"]="complete";write_json(out/"manifest.json",manifest)
    render_report(out,summary,manifest)
    return summary

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/benchmark.json")
    parser.add_argument("--out",required=True)
    parser.add_argument("--backend",choices=["scripted","ollama"])
    parser.add_argument("--model",help="exact local Ollama model tag; no download is performed")
    args=parser.parse_args(argv)
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.backend:cfg["agent"]["backend"]=args.backend
    if args.model:cfg["agent"]["model"]=args.model
    try:
        summary=run(cfg,args.out)
    except (ValueError,FileExistsError,KeyError) as exc:
        parser.error(str(exc))
    for method,result in summary["methods"].items():
        s=result["calibrated"]
        print(f"{method}: accuracy={s['accuracy']:.3f}, brier={s['brier']:.3f}, failures={s['failure_rate']:.3f}")
    print(f"Report: {Path(args.out)/'REPORT.md'}")

if __name__=="__main__":main()
