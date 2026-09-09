# Anthropic — Research Engineer / Scientist, Alignment Science

- **Posting:** https://job-boards.greenhouse.io/anthropic/jobs/4009165008
- **Location:** SF / NYC / remote-friendly
- **Projects highlighted:** AI Safety Red-Team Evaluation · LLM Ensemble Bias Detection · RAG Production Pipeline
- **Attach:** `Resume_Derek_Lankeaux.md`, `AI_Safety_RedTeam_Evaluation_Publication.pdf`, `LLM_Bias_Detection_Publication.pdf`

---

Dear Alignment Science Team,

I'm applying for the Research Engineer / Scientist role on Alignment Science.
I'm an Applied Statistics M.S. candidate at RIT (expected 2026) whose
independent research has been organized around exactly the problem your team
works on: measuring model behavior carefully enough to be trusted as a basis
for decisions.

My AI Safety Red-Team Evaluation case study built a two-stage workflow that
scaled harm labeling beyond manual review — an LLM ensemble annotated 12,500
response pairs across six harm categories, with a supervised classifier on top.
I reported Krippendorff's α = 0.81 for annotator agreement and 96.8% held-out
accuracy for the classifier, and I kept those numbers separate on purpose so a
reader can reason about reliability and downstream error independently.
Bayesian hierarchical modeling and SHAP-based attribution supported audit-style
review of high-risk cases.

The companion LLM Ensemble Bias Detection project extended the same pattern to
content review: 4,500 textbook passages evaluated by a rubric-based LLM
ensemble, 67,500 individual ratings, Krippendorff's α = 0.84, publisher-level
effects modeled with Bayesian partial pooling and MCMC diagnostics. The output
was a workflow for expert reviewers that surfaces disagreement and uncertainty
rather than a single collapsed score — a shape of evaluation I think matches
how alignment work has to communicate risk.

The RAG Production Pipeline case study grounds the systems side. I built a
hybrid retrieval architecture (dense + BM25 + re-ranking) with grounding checks
and confidence calibration, reported 96.3% Recall@10 and 94.2% citation
precision, and documented drift, failure-mode, and rollback considerations.
It's a compact demonstration that I can move an evaluation-first mindset into
running infrastructure.

I care about interpretable, honest measurement — where an evaluation is
defended by its uncertainty analysis as much as by its headline number — and
I'd like to bring that to Alignment Science. Portfolio, reports, and PDFs
are at https://github.com/dl1413/Machine-Learning-Research-Engineering-Project-Profile.

Thank you for considering my application. I'd welcome the chance to discuss
where my work could contribute.

Sincerely,
Derek Lankeaux
dlankeaux12@gmail.com · [LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413)
