"""Fixed workflows, deterministic adaptive controller, optional local LLM controller."""
import json
import math
import copy
import urllib.request
from urllib.parse import urlparse
from .schema import CLASSES, TASK_DESCRIPTION, validate_prediction, uniform_prediction
from .tools import ToolSession, TOOL_COSTS, ToolError

BASE_TOOLS=("quality","trend","spectrum","outliers")
VERIFY_TOOLS=("robust_trend","robust_outliers","spectral_stability")
METHODS=("fixed","agent","agent_verified","fixed_full_tools")

def sigmoid(x): return 1/(1+math.exp(-max(-30,min(30,x))))
def binary(p): return {"no":1-p,"yes":p}

def from_evidence(evidence, robust=False):
    q=evidence.get("quality",{})
    tr=evidence.get("robust_trend" if robust else "trend", evidence.get("trend",{}))
    out=evidence.get("robust_outliers" if robust else "outliers", evidence.get("outliers",{}))
    sp=evidence.get("spectrum",{})
    pred=uniform_prediction()
    if tr:
        slope=tr["net_change"];scale=max(.25,tr["residual_scale"]*.6)
        weights=[math.exp(-((slope+3)/max(1,scale))**2/2),
                 math.exp(-(slope/max(.5,scale))**2/2),
                 math.exp(-((slope-3)/max(1,scale))**2/2)]
        # Clamp before normalizing to retain strictly positive class mass.
        weights=[max(v,1e-8) for v in weights];total=sum(weights)
        pred["trend"]={c:w/total for c,w in zip(CLASSES["trend"],weights)}
    if sp: pred["seasonal"]=binary(sigmoid(12*(sp["peak_share"]-.32)))
    if robust and "spectral_stability" in evidence:
        st=evidence["spectral_stability"]
        p2=sigmoid(12*(st["min_peak_share"]-.3))
        pred["seasonal"]=binary((pred["seasonal"]["yes"]+p2)/2)
    if out: pred["anomaly"]=binary(sigmoid(1.2*(out["max_z"]-5)))
    if q: pred["missing_block"]=binary(.999 if q["longest_missing_run"]>=8 else .001)
    return validate_prediction(pred)

def scripted_agent(session):
    # A deterministic tool policy, explicitly NOT an LLM.
    for name in BASE_TOOLS: session.call(name)
    first=from_evidence(session.cache)
    if max(first["trend"].values()) < .85 or session.cache["quality"]["missing_fraction"]>.1:
        session.call("robust_trend")
    if .15 < first["anomaly"]["yes"] < .85 or first["seasonal"]["yes"] > .5:
        session.call("robust_outliers")
    robust=dict(session.cache)
    # Adaptive diagnostics replace only the queried basic estimators.
    for basic,extra in (("trend","robust_trend"),("outliers","robust_outliers")):
        if extra in robust: robust[basic]=robust[extra]
    return from_evidence(robust)

def verify(pred, session):
    before=json.loads(json.dumps(pred))
    for name in BASE_TOOLS+VERIFY_TOOLS: session.call(name)
    # Fixed numerical verifier; does not claim independent human or formal verification.
    evidence=from_evidence(session.cache,robust=True)
    changes=[]
    result={}
    for task in CLASSES:
        old=max(pred[task],key=pred[task].get);new=max(evidence[task],key=evidence[task].get)
        if old != new:
            result[task]=evidence[task]
            changes.append(task)
        else:
            result[task]={c:(pred[task][c]+evidence[task][c])/2 for c in CLASSES[task]}
    return validate_prediction(result),{"changed_tasks":changes,"before":before,"evidence_prediction":evidence}

class OllamaController:
    def __init__(self,config,seed):
        self.config=config;self.seed=seed;self.calls=[];self.input_tokens=0;self.output_tokens=0
        endpoint=urlparse(config["endpoint"])
        if endpoint.scheme != "http" or endpoint.hostname not in ("localhost","127.0.0.1","::1") or endpoint.path!="/api/chat" or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
            raise ValueError("only an explicit loopback Ollama /api/chat endpoint is allowed")

    def request(self,messages):
        payload={"model":self.config["model"],"messages":messages,"stream":False,"format":"json",
                 "options":{"temperature":0,"seed":self.seed,"num_predict":512}}
        req=urllib.request.Request(self.config["endpoint"],data=json.dumps(payload).encode(),
                                   headers={"Content-Type":"application/json"},method="POST")
        # Do not follow redirects off loopback and do not route through environment proxies.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs): return None
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        with opener.open(req,timeout=self.config["timeout_seconds"]) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000: raise ValueError("response_too_large")
        data=json.loads(raw)
        self.input_tokens+=int(data.get("prompt_eval_count",0));self.output_tokens+=int(data.get("eval_count",0))
        self.calls.append({"response":data.get("message",{}).get("content",""),
                           "input_tokens":data.get("prompt_eval_count",0),"output_tokens":data.get("eval_count",0),
                           "model":data.get("model"),"done_reason":data.get("done_reason")})
        return json.loads(data["message"]["content"])

    def run(self,session):
        system=TASK_DESCRIPTION+" Available tools: "+", ".join(TOOL_COSTS)+(
            '. Respond with exactly {"tool":"NAME"} to inspect evidence, or '
            '{"prediction":{"trend":{"down":0.2,"flat":0.6,"up":0.2},'
            '"seasonal":{"no":0.5,"yes":0.5},"anomaly":{"no":0.5,"yes":0.5},'
            '"missing_block":{"no":0.5,"yes":0.5}}} to finish. '
            'Only listed tools exist. Do not return code, markdown, or prose. '
            'trend gives a linear net change; robust_trend gives median pairwise slope. '
            'spectrum gives a detrended peak share; outliers gives residual max robust z. '
            'robust_outliers fits a trimmed trend plus sinusoid. spectral_stability compares halves. '
            'quality reports missingness. These are heuristics, not truth.')
        messages=[{"role":"system","content":system},
                  {"role":"user","content":json.dumps({"values":session.observation.values})}]
        for step in range(self.config["max_steps"]):
            action=self.request(messages)
            if isinstance(action,dict) and set(action)=={"prediction"}:
                return validate_prediction(action["prediction"])
            if not isinstance(action,dict) or set(action)!={"tool"} or not isinstance(action["tool"],str):
                raise ValueError("invalid_action_schema")
            result=session.call(action["tool"])
            messages.extend([{"role":"assistant","content":json.dumps(action)},
                             {"role":"user","content":json.dumps({"tool":action["tool"],"result":result})}])
        raise ToolError("step_budget")

def run_method(observation,method,config,seed):
    if method not in METHODS: raise ValueError("unknown method")
    session=ToolSession(observation,config["max_tools"]);controller=None;verification=None
    failure=None;pred=None
    try:
        if method.startswith("fixed"):
            for tool in BASE_TOOLS: session.call(tool)
            if method=="fixed_full_tools":
                for tool in VERIFY_TOOLS: session.call(tool)
            pred=from_evidence(session.cache,robust=method=="fixed_full_tools")
        else:
            if config["backend"]=="scripted": pred=scripted_agent(session)
            else:
                controller=OllamaController(config,seed);pred=controller.run(session)
            if method=="agent_verified": pred,verification=verify(pred,session)
        validate_prediction(pred)
    except (ToolError,ValueError,KeyError,TypeError,OSError) as exc:
        failure={"type":type(exc).__name__,"reason":str(exc)[:200]}
        pred=None
    return {"prediction":pred,"failure":failure,"trace":session.trace,
            "verification":verification,"model_calls":controller.calls if controller else [],
            "input_tokens":controller.input_tokens if controller else 0,
            "output_tokens":controller.output_tokens if controller else 0,
            "backend":config["backend"] if method.startswith("agent") else "fixed"}

def verified_from(base, observation, config):
    """Reuse the SAME agent proposal, avoiding an independent-generation confound."""
    result=copy.deepcopy(base)
    if result["failure"]: return result
    session=ToolSession(observation,config["max_tools"])
    session.trace=copy.deepcopy(base["trace"])
    session.cache={event["tool"]:event["result"] for event in session.trace}
    try:
        result["prediction"],result["verification"]=verify(base["prediction"],session)
    except (ToolError,ValueError,KeyError,TypeError) as exc:
        result["failure"]={"type":type(exc).__name__,"reason":str(exc)[:200]}
        result["prediction"]=None
    result["trace"]=session.trace
    return result
