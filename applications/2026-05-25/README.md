# Daily Application Batch — 2026-05-25 (Mon)

Five tailored applications, one per role family per the playbook. Each app
leads with the best-fit primary project and lists the other two as
supporting evidence.

| # | Company | Role | Tier | Lead Project | Geo |
|---|---|---|---|---|---|
| 1 | Anthropic | Research Engineer, Model Evaluations | A — Frontier | Red-Team Eval | Remote / NYC |
| 2 | OpenAI | MTS, Red Team | A — Frontier | Red-Team Eval | NYC |
| 3 | Two Sigma | ML Research Engineer | C — NYC Finance | Breast Cancer | NYC |
| 4 | Memorial Sloan Kettering | ML Engineer, Computational Oncology | D — NYC Healthcare | Breast Cancer | NYC |
| 5 | Patronus AI | LLM Evaluation Engineer | B — Eval Platform | Red-Team Eval | Remote |

## Project rotation this batch

- **Project 1 — AI Safety Red-Team Eval** leads 3/5 (Anthropic, OpenAI, Patronus)
- **Project 3 — Breast Cancer Classification** leads 2/5 (Two Sigma, MSK)
- **Project 2 — LLM Bias Detection** supports 4/5 (Bayesian rigor, multi-LLM-at-scale)

This distribution is intentional: Project 1 is the strongest signal for the
biggest funnel (safety + eval-platform), Project 3 anchors NYC finance and
healthcare where production deployment + clinical-grade rigor matter, and
Project 2 reinforces statistical defensibility across all five.

## Submission steps (manual — Claude generates packages, you submit)

For each of the 5 packages in this folder:

1. Open the company's careers page (link in the file).
2. Save the live JD as `applications/2026-05-25/jd_<NN>_<company>.pdf`.
3. Paste the tailored resume bullets into the top of the Projects section of
   `Resume_Derek_Lankeaux.md`, export PDF as the listed `resume_version`.
4. Paste the cover letter into the application form (or attach as PDF).
5. Submit.
6. Flip the row in `application_tracker.csv` from `ready_to_submit` to
   `submitted`, fill `jd_link`, and add a one-line note.

## Tomorrow (2026-05-26)

Rotate to:
- Cohere or DeepMind NYC (Tier A) — Project 1 lead
- Scale AI or Hugging Face (Tier B) — Project 1 lead
- Citadel or Jane Street (Tier C) — Project 3 lead
- Tempus or Flatiron (Tier D) — Project 3 lead
- Stripe, Notion, or Datadog (Tier E) — Project 3 lead with Project 2 support

Keep ~80 active leads. Re-rotate Tier A weekly (1/week per frontier lab) to
avoid double-applying.
