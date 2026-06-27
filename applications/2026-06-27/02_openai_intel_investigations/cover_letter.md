# Cover Letter — OpenAI, Trust & Safety / Intelligence & Investigations

Dear OpenAI Hiring Team,

I'm applying for the Intelligence & Investigations Engineer role on the Trust
& Safety team. The work — identifying, investigating, and disrupting malicious
uses of OpenAI products — is the exact intersection of ML, eval rigor, and
policy that I've optimized my last six months of work for.

Built a 3-model LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) that detects
harmful AI outputs at **96.8% accuracy and 340x lower cost** than human
annotation, while preserving audit-grade reliability (Krippendorff's
alpha = 0.81). The pipeline auto-grades 12,500 response pairs across 6 harm
categories at ROC-AUC 0.9923 and **850 samples/hr** with circuit-breakered
async API integration. Where I think this is directly relevant to I&I: the 47
features powering the stacking classifier are split between linguistic,
semantic, and structural harm signals — the same shape of features you'd
build to triage real-world abuse patterns across product surfaces. The
Bayesian hierarchical layer on top produces 95% HDI risk intervals per
sample, which matters when an investigator has to decide between escalation
and dismissal.

The supporting projects extend the same toolkit into adjacent abuse-detection
territory. My LLM bias-detection study ran 67,500 ratings over 4,500 passages
and found statistically significant publisher-level directional bias (Friedman
chi-squared = 42.73, p < 0.001) — the kind of multi-source signal triangulation
that matches investigation work. And a clinical-grade ensemble classifier
(99.12% accuracy, 100% precision, ROC-AUC 0.9987) taught me to operate when
false positives have a cost — which they always do in T&S.

Two habits I'd bring to I&I: every detector ships with a calibrated uncertainty
estimate, not a raw score, and every claim about a model's behavior is
defensible from the eval design upward. I'm in / available for New York, with
a 2026 start once I wrap my MS in Applied Statistics at RIT.

Best,
Derek Lankeaux
