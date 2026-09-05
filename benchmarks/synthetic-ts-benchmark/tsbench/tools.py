"""Allowlisted read-only numerical tools. No eval, code execution, files, or labels."""
import time
import numpy as np
from .schema import Observation, longest_run

TOOL_COSTS = {"quality":1, "trend":2, "spectrum":4, "outliers":2,
              "robust_trend":8, "robust_outliers":5, "spectral_stability":8}

class ToolError(RuntimeError):
    pass

def filled(obs):
    a = np.array([np.nan if v is None else v for v in obs.values], dtype=float)
    ok = np.isfinite(a)
    if ok.sum() < 16:
        raise ToolError("insufficient_observations")
    x = np.arange(len(a))
    return np.interp(x,x[ok],a[ok]), ok

def trend_stats(a, ok, robust=False):
    x = np.linspace(0,1,len(a))[ok]; y=a[ok]
    if robust:
        ii,jj=np.triu_indices(len(y),1)
        slope=float(np.median((y[jj]-y[ii])/(x[jj]-x[ii])))
        intercept=float(np.median(y-slope*x))
    else:
        slope,intercept=np.polyfit(x,y,1)
    resid=y-(slope*x+intercept)
    scale=max(.05,float(1.4826*np.median(np.abs(resid-np.median(resid)))))
    return {"net_change":float(slope),"residual_scale":scale}

def spectrum_stats(a):
    x=np.linspace(0,1,len(a)); detr=a-np.polyval(np.polyfit(x,a,1),x)
    power=np.abs(np.fft.rfft(detr))**2
    frequencies=np.fft.rfftfreq(len(a))
    eligible=np.where((frequencies>=1/40)&(frequencies<=1/8))[0]
    best=int(eligible[np.argmax(power[eligible])])
    return {"peak_share":float(power[best]/max(float(power[1:].sum()),1e-9)),
            "period":float(1/frequencies[best])}

def robust_residual(a, ok):
    # Fits trend + a single harmonic by trimmed least squares, then scores only observed points.
    n=len(a);x=np.linspace(0,1,n);best=None
    for period in np.linspace(8,40,49):
        angle=2*np.pi*np.arange(n)/period
        design=np.column_stack([np.ones(n),x,np.sin(angle),np.cos(angle)])
        inliers=ok.copy()
        for _ in range(3):
            coef=np.linalg.lstsq(design[inliers],a[inliers],rcond=None)[0]
            residual=a-design@coef
            scale=max(.05,1.4826*np.median(np.abs(residual[ok]-np.median(residual[ok]))))
            candidate=ok & (np.abs(residual-np.median(residual[ok])) < 3*scale)
            if candidate.sum() >= 16: inliers=candidate
        loss=float(np.median(np.abs(residual[ok])))
        if best is None or loss < best[0]: best=(loss,residual.copy())
    r=best[1][ok];scale=max(.08,float(1.4826*np.median(np.abs(r-np.median(r)))))
    return {"max_z":float(np.max(np.abs(r-np.median(r)))/scale),"residual_scale":scale}

class ToolSession:
    def __init__(self, observation, budget=7):
        if not isinstance(observation, Observation):
            raise TypeError("ToolSession accepts only an Observation")
        self.observation=observation;self.budget=budget;self.cache={};self.trace=[]

    def call(self,name):
        if name not in TOOL_COSTS: raise ToolError("unknown_tool")
        if name in self.cache:
            self.trace.append({"tool":name,"cached":True,"units":0,"latency_ms":0,
                               "result":dict(self.cache[name])})
            return dict(self.cache[name])
        if len(self.cache) >= self.budget: raise ToolError("tool_budget")
        start=time.perf_counter()
        if name == "quality":
            mask=[v is None for v in self.observation.values]
            result={"missing_fraction":sum(mask)/len(mask),"longest_missing_run":longest_run(mask),
                    "observed":len(mask)-sum(mask)}
        else:
            a,ok=filled(self.observation)
            if name in ("trend","robust_trend"):
                result=trend_stats(a,ok,name=="robust_trend")
            elif name == "spectrum": result=spectrum_stats(a)
            elif name == "outliers":
                x=np.linspace(0,1,len(a));r=(a-np.polyval(np.polyfit(x[ok],a[ok],1),x))[ok]
                scale=max(.08,float(1.4826*np.median(np.abs(r-np.median(r)))))
                result={"max_z":float(np.max(np.abs(r-np.median(r)))/scale)}
            elif name == "robust_outliers": result=robust_residual(a,ok)
            else:
                halves=[spectrum_stats(v) for v in np.array_split(a,2)]
                result={"min_peak_share":min(v["peak_share"] for v in halves),
                        "period_ratio":max(v["period"] for v in halves)/min(v["period"] for v in halves)}
        self.cache[name]=result
        self.trace.append({"tool":name,"cached":False,"units":TOOL_COSTS[name],
                           "latency_ms":(time.perf_counter()-start)*1000,"result":result})
        return dict(result)
