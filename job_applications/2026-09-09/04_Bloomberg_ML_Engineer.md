# Bloomberg — Machine Learning Engineer

- **Posting:** https://careers.bloomberg.com/job/search?q=machine+learning+engineer&loc=new-york
- **Location:** NYC (hybrid)
- **Projects highlighted:** RAG Production Pipeline · AI Safety Red-Team Evaluation · Breast Cancer Classification
- **Attach:** `Resume_Derek_Lankeaux.md`, `RAG_Project_Publication.pdf`, `AI_Safety_RedTeam_Evaluation_Publication.pdf`
- **Note:** Bloomberg's careers site rotates individual req URLs; the search
  URL above returns the current NYC ML Engineer listings. Pick the closest
  match by team (AI Group / Data Science / Search).

---

Dear Bloomberg AI Team,

I'm writing about your Machine Learning Engineer openings in New York. Your AI
Group's work — retrieval, grounded generation, and evaluation over a very
large, high-signal corpus — is a good match for what I've been building as an
Applied Statistics M.S. candidate at RIT (2026).

My RAG Production Pipeline case study is the most direct fit. I designed a
hybrid retrieval architecture combining dense embeddings, BM25, and a
ColBERT-style re-ranker, added grounding checks and confidence calibration,
and reported 96.3% Recall@10 with 94.2% citation precision on the documented
evaluation. Just as important, I specified the observability, latency,
throughput, drift monitoring, and rollback considerations that turn a demo
into something operable — the reader can see exactly where Prometheus,
Kafka, and MLflow plug in, and what the alerting looks like when retrieval
quality drifts.

The AI Safety Red-Team Evaluation study is the measurement discipline behind
that pipeline: a two-stage evaluation across 12,500 response pairs and six
harm categories, with an LLM ensemble producing labels for a supervised
classifier. Krippendorff's α = 0.81 for annotator agreement and 96.8%
classifier accuracy, kept separate. Bayesian hierarchical modeling and SHAP
support the audit-style reporting that regulated environments need.

The Breast Cancer Classification benchmark closes the loop on ML rigor. Eight
ensemble models on the WDBC dataset with cross-validation, calibration, and
threshold analysis; 99.12% held-out accuracy and 0.9987 ROC-AUC on the best
configuration. It's the discipline of comparing models honestly and
communicating limits — the pattern I would carry into any production ML
review at Bloomberg.

Stack alignment: Python, FastAPI, MLflow, Qdrant, Docker, Kubernetes, plus
strong SQL and R. Portfolio:
https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile

Thank you for the consideration.

Sincerely,
Derek Lankeaux
dlankeaux12@gmail.com · [LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413)
