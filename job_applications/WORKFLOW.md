# Daily Job Application Workflow

**Goal:** 5 high-quality applications per day to NYC or remote roles. Quality means: targeted cover letter (not a paste), tailored resume bullets, and a tracked follow-up cadence.

---

## The 60-minute application loop (per application)

| Step | Time | What |
|---|---|---|
| 1. Triage | 5 min | Read JD. Score 1–5 on fit. Skip anything below 3. |
| 2. Reference hunt | 5 min | Find one specific company artifact (paper, blog, product launch, model card) to reference in line 1 of the letter. If you can't in 5 min, skip the company. |
| 3. Template select | 2 min | Pick the right cover-letter template + resume bullet variant from `resume_bullets/`. |
| 4. Tailor letter | 15 min | Copy template → fill placeholders → drop in 1–2 JD-specific bullets → cut to 300–400 words. |
| 5. Tailor resume | 10 min | Pick role archetype in `resume_bullets/`, swap matching bullets into your master resume. Reorder skills if needed. |
| 6. Submit | 10 min | Application + portfolio link + LinkedIn note to one person on the team (use `outreach/linkedin_messages.md` Variant A/B/C). |
| 7. Log | 3 min | Add row to `application_tracker.csv`. Set follow-up date = today + 7 business days. |

**Total:** ~50 minutes per application × 5 = ~4 hours/day. The first week will feel slower (~60 min each); from week 2 you'll have your own snippets in muscle memory.

---

## Daily routine

**Morning (90 min)** — Sourcing
- LinkedIn Jobs · filter: "New York City Metro Area" + "Remote (United States)" + 24-hour posting window
- Built-in Greenhouse, Lever, Ashby boards (search keywords: "ML Research Engineer", "AI Safety", "Applied ML", "Machine Learning Engineer", "Data Scientist", "Quantitative Researcher")
- Anthropic, OpenAI, Google DeepMind, Cohere, Scale, AI2, Hugging Face career pages
- NYC-specific: jobsnyc.gov for civic; Bloomberg, Two Sigma, Jane Street, Ramp, Datadog, Spotify NYC, Etsy, Hugging Face NYC
- Triage to ~10 candidates. Pick the **top 5 by fit + salary band**.

**Midday (4 hr)** — Apply
- Run the 60-minute loop above × 5.

**Late afternoon (30 min)** — Follow-up + outreach
- Check `application_tracker.csv` for entries hitting their follow-up date.
- Send 3–5 LinkedIn connection notes (`outreach/linkedin_messages.md`) to people on teams you applied to today.
- Reply to any recruiter messages from the morning.

**Evening (15 min)** — Maintenance
- Update tracker rows with any responses received.
- Note one thing that worked and one that didn't in `lessons.md` (create it on day 1).

---

## Application-tracker columns

`date_applied, company, role_title, job_id, location, remote_ok, salary_min, salary_max, source, jd_link, cover_letter_template, resume_version, referral_contact, application_status, follow_up_date, response_date, interview_dates, final_outcome, notes`

**Status enum:** `submitted | acknowledged | rejected | screen_scheduled | screen_passed | onsite_scheduled | onsite_passed | offer | offer_accepted | offer_declined | withdrew | ghosted`

**Rule:** Anything `submitted` with no `response_date` after 14 days → mark `ghosted` and stop following up.

---

## When to skip a role

- JD is generic boilerplate with no team/product detail → you can't write a targeted letter.
- Salary band is below your floor → don't anchor low.
- "Senior" role requiring 5+ years industry → save your application bandwidth for fights you can win.
- You can't find one specific reference for line 1 of the letter in 5 minutes → the company doesn't want to be found, or you don't have time today.
- Title is a clear stretch-down AND salary is below market → only apply if you actually want the role for non-comp reasons (mission, learning).

---

## Volume reality check

5 apps/day × 5 weekdays = **25/week**. Typical response rates for tailored ML/AI applications in 2026:
- ~15–25% recruiter screens
- ~5–10% technical screens
- ~1–3% offers

So 25/week → expect ~1–2 offers per month with consistent execution. If you're hitting that rate, the funnel works. If after 6 weeks of consistent volume you're below 10% screens, the problem is upstream (letter quality, resume fit, target selection) — not volume.

---

## What's in this folder

```
job_applications/
├── FACTS.md                                # Canonical facts about you (use to fill templates)
├── WORKFLOW.md                             # This file
├── application_tracker.csv                 # Daily log
├── cover_letters/
│   ├── 01_ml_research_engineer.md          # Template — frontier-lab / research
│   ├── 02_ai_safety_alignment.md           # Template — safety / evals
│   ├── 03_applied_ml_engineer.md           # Template — product ML
│   ├── 04_ml_platform_infra.md             # Template — MLOps / platform
│   ├── 05_data_scientist_quant.md          # Template — quant / DS / stats-heavy
│   └── _live_applications/
│       └── NYC_Mayors_Office_Data_Manager_780387.md   # Tailored, ready to submit
├── resume_bullets/
│   ├── ai_safety_redteam.md                # Project bullets by role archetype
│   ├── llm_bias_detection.md
│   └── breast_cancer_classification.md
└── outreach/
    └── linkedin_messages.md                # Connect notes, recruiter replies, intro asks
```
