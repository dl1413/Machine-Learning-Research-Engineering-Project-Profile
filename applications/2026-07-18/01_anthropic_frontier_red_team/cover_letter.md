# Cover Letter — Anthropic, Frontier Red Team (Cyber)

Dear Frontier Red Team,

I'm applying for the Research Scientist / Engineer role on the Frontier Red Team (Cyber). The framing on your job page — that 2026 is the year frontier models cross into expert-level cybersecurity capability and that FRT exists to measure and contain that before it deploys — is the exact problem I've been building toward for the past year.

My AI Safety Red-Team Evaluation Framework is a dual-stage system: an LLM ensemble (GPT-4o, Claude-3.5, Llama-3.2) annotates 12,500 model responses across six harm categories, and a Stacking Classifier over 47 engineered linguistic / semantic / structural features scores them at 96.8% accuracy (97.2% precision, 96.1% recall, ROC-AUC 0.9923). Inter-rater reliability across the LLMs hits Krippendorff's α = 0.81, matching the audit threshold your public safety writeups use. Cost per sample lands at $0.018 versus $6.12 for human labeling — a 340× reduction — and the pipeline processes 850 samples/hour, so evaluations that used to take a quarter can now run overnight.

The technical result I'd most want to walk you through in an interview is the adversarial-attack taxonomy: eight MITRE ATLAS-aligned vectors, in which multi-turn escalation was the highest-risk pathway at 31.8% success — a finding that only surfaced because I paired the eval framework with a Bayesian hierarchical risk model (PyMC, R-hat < 1.01, 95% HDI) rather than reporting point estimates. That combination — adversarial eval design + honest uncertainty quantification + audit-grade documentation (IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act) — is the shape of work I want to keep doing, and it seems to be exactly what FRT ships.

Two adjacent projects reinforce the profile: an LLM ensemble textbook-bias detector (67,500 ratings, α = 0.84, Friedman χ² = 42.73, p < 0.001, PyMC partial pooling) and a clinical-grade breast-cancer classifier (99.12% accuracy, 100% precision, Platt calibration to ECE 0.0089, FastAPI < 100 ms p95). Together they show I can carry an eval question from prompt design through hierarchical inference to a deployed, monitored service.

I'd love to talk about how the red-team framework could extend to cyber-capability probes — chained-tool exploitation, sandbox-escape scenarios, and reward-hacking under agentic scaffolds — and about how FRT frames "shipped verdict vs. we still don't know" internally.

Thank you for reading.

Warmly,
Derek Lankeaux
[LinkedIn](https://linkedin.com/in/derek-lankeaux) · [GitHub](https://github.com/dl1413) · [Portfolio](https://dl1413.github.io/LLM-Portfolio/)
