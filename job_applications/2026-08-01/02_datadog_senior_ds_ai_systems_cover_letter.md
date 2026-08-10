# Datadog — Senior Data Scientist, AI Systems for Business Teams

**Location:** New York, NY
**Apply:** https://careers.datadoghq.com/detail/6679282/
**Lead project:** AI Safety Red-Team Evaluation Framework
**Supporting projects:** LLM Bias Detection (Bayesian judge-disagreement modeling), Breast Cancer (calibration & threshold policies for AI-assisted decisions)

---

Dear Datadog AI Systems team,

Datadog's whole product is "does the signal actually mean what we think it means?" — and pushing that discipline into an internal AI Systems team, where LLM outputs directly shape how business partners make decisions, is one of the more interesting eval problems in the industry right now. I've spent the last year building exactly this kind of harness and would love to bring it to your team.

My most directly relevant work is an AI Safety Red-Team Evaluation Framework I built and published in April 2026. It stacks GPT-4o, Claude-3.5, and Llama-3.2 as judges on 12,500 response pairs across 6 harm categories, then trains a 47-feature stacking meta-classifier that reaches 96.8% accuracy, 97.2% precision, and ROC-AUC 0.9923 on a held-out benchmark — with Krippendorff's α = 0.81 holding inter-judge reliability at the audit-grade line. The eval harness runs at 850 samples/hour and $0.018/sample, a 340× cost reduction versus human annotation, with circuit breakers, async batching, exponential backoff, and MLflow experiment tracking baked in from the start. That's the same operational hygiene Datadog would expect from any monitoring system it ships to internal customers.

For the "when do the judges disagree, and does it matter?" question that a Senior DS on this team will get pulled into constantly, I built a PyMC Bayesian hierarchical model on top of the ensemble outputs. It produces 95% HDIs per judge and surfaces systematic per-model blind spots — the difference between "GPT-4o is noisier here" and "GPT-4o is systematically wrong here." Combined with a dual-filter defense analysis showing 78% harm reduction (21.8% → 4.8%), it gave me a framework for reasoning about eval trustworthiness end-to-end, which is exactly what "AI Systems for Business Teams" needs before results ship to non-technical partners.

I bring an MS in Applied Statistics (RIT, 2026), fluency in the production stack Datadog already uses (Python, SQL, MLflow, FastAPI, Docker), and three publication-grade technical reports — one per project — showing the full lifecycle from framing to shipped artifact. Would welcome a first conversation.

Best,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
