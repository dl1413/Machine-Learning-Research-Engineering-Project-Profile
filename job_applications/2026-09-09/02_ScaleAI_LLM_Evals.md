# Scale AI — ML Research Scientist / Tech Lead, LLM Evals

- **Posting:** https://scale.com/careers/4304790005
- **Location:** NYC / remote
- **Projects highlighted:** AI Safety Red-Team Evaluation · LLM Ensemble Bias Detection · RAG Production Pipeline
- **Attach:** `Resume_Derek_Lankeaux.md`, `AI_Safety_RedTeam_Evaluation_Publication.pdf`, `RAG_Project_Publication.pdf`

---

Dear Scale Evaluations Hiring Team,

I'm writing about the LLM Evals research role on the Scale AI evaluations team.
My independent research is a direct match: three published case studies, all
organized around evaluation design, reliability, and the operational plumbing
that connects them.

The AI Safety Red-Team Evaluation project designed a two-stage evaluation for
harm classification — an LLM ensemble produced labels for 12,500 response pairs
across six harm categories, and a stacked supervised classifier consumed those
labels. I reported Krippendorff's α = 0.81 for inter-rater reliability and
96.8% held-out classifier accuracy, treated as separate measurements rather
than a single conflated number, and used PyMC and SHAP to support Bayesian
risk analysis and per-category attribution. Every metric is defensible by its
own uncertainty analysis.

The LLM Ensemble Bias Detection case study is the "many judges, calibrated
disagreement" pattern you already run at scale: 4,500 textbook passages, a
rubric-based LLM ensemble, 67,500 ratings, Krippendorff's α = 0.84,
publisher-level partial pooling with MCMC diagnostics. The end product is a
disagreement- and uncertainty-aware review workflow — closer to how humans
actually use evaluation output than a leaderboard.

The RAG Production Pipeline case study covers the deployment side. Hybrid
dense-plus-BM25 retrieval with a re-ranker, grounding checks, confidence
calibration; 96.3% Recall@10 and 94.2% citation precision on the documented
evaluation; and a specification of observability, drift monitoring, latency,
and rollback that the evaluation frameworks feed into. If Scale's LLM-judge
pipelines need someone who thinks about the eval and the pipeline together,
that's the framing I bring.

Formally I'm an Applied Statistics M.S. candidate (RIT, 2026) with strong
Bayesian, calibration, and reliability chops in addition to production
Python/FastAPI/MLflow/Qdrant experience.

Full portfolio: https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile

Thank you for the consideration — I'd be glad to walk through any of these
projects in more detail.

Sincerely,
Derek Lankeaux
dlankeaux12@gmail.com · [LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413)
