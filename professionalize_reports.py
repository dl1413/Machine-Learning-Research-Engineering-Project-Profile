#!/usr/bin/env python3
"""
Build two publication-formatted project reports under projects/:

  01_AI_Safety_RedTeam_Evaluation.{md,pdf}
  02_Bayesian_Methods_in_Applied_Classification.{md,pdf}
       (combines the WBCD calibrated-classifier project and the
        LLM-ensemble textbook-bias-detection project as two case
        studies of applied Bayesian methodology.)

Reuses the WeasyPrint pipeline from generate_publication_pdfs.py:
markdown -> HTML -> PDF with the existing PUBLICATION_CSS / PDF_METADATA.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Reuse renderer + CSS from the existing pipeline.
from generate_publication_pdfs import generate_pdf

ROOT = Path(__file__).parent.resolve()
OUT  = ROOT / "projects"
OUT.mkdir(exist_ok=True)

SRC_AI_SAFETY = ROOT / "AI Safety Red-Team Evaluation_ Technical Analysis Report.md"
SRC_BIAS      = ROOT / "LLM_Ensemble_Bias_Detection_Report.md"
SRC_WBCD      = ROOT / "Breast_Cancer_Classification_Report.md"

# ─── Shared header template ──────────────────────────────────────────────────

PRO_HEADER = (
    "# {title}\n\n"
    "**Project:** {subtitle}  \n"
    "**Author:** Derek Lankeaux, MS Applied Statistics  \n"
    "**Role:** Data Scientist | Applied Statistician  \n"
    "**Institution:** Rochester Institute of Technology  \n"
    "**Date:** April 2026  \n"
    "**Version:** {version}  \n"
    "**AI Standards Compliance:** IEEE 2830-2025, ISO/IEC 23894:2025, EU AI Act (2025)\n\n"
    "> **Data Science Focus:** This report documents an end-to-end data "
    "science project — problem framing, statistical methodology, results with "
    "quantified uncertainty, and stakeholder-ready deliverables — relevant to "
    "2026 Data Scientist roles (experimentation, Bayesian inference, predictive "
    "modeling, and responsible-AI practice).\n\n"
    "---\n"
)


def strip_old_header(text: str) -> str:
    """Drop the original metadata header block up to (but not including) `## Abstract`."""
    m = re.search(r"^## Abstract\s*$", text, flags=re.MULTILINE)
    return text[m.start():] if m else text


def normalize_simple(src: Path, title: str, subtitle: str, version: str) -> str:
    """Header-normalize a single report (no body changes)."""
    body = strip_old_header(src.read_text(encoding="utf-8"))
    return PRO_HEADER.format(title=title, subtitle=subtitle, version=version) + body


# ─── Combined report builder ─────────────────────────────────────────────────

def extract_body_from(src: Path, body_start_pattern: str, body_end_pattern: str) -> str:
    """Return the slice of `src` between two anchor headings (inclusive of start,
    exclusive of end). Used to lift the substantive technical body of a source
    report into a Part of the combined report."""
    text = src.read_text(encoding="utf-8")
    start = re.search(body_start_pattern, text, flags=re.MULTILINE)
    end   = re.search(body_end_pattern, text, flags=re.MULTILINE)
    if not start:
        raise ValueError(f"Body start not found in {src.name}: {body_start_pattern!r}")
    end_pos = end.start() if end else len(text)
    return text[start.start():end_pos]


def renumber_sections(body: str, offset: int) -> str:
    """Shift numeric section prefixes in `## N. Heading`, `## Na. Heading`,
    and `### N.M Heading` by `offset` so a Part can be embedded in the
    combined report without colliding with the host's section numbering.
    Handles letter-suffixed sub-sections (e.g. `5a`, `8a`) so the relative
    order against their parent number is preserved after shifting."""
    def repl_h2(m: re.Match) -> str:
        return f"## {int(m.group(1)) + offset}{m.group(2)}. {m.group(3)}"
    def repl_h3(m: re.Match) -> str:
        return f"### {int(m.group(1)) + offset}.{m.group(2)} {m.group(3)}"
    body = re.sub(r"^##\s+(\d+)([a-z]?)\.\s+(.+)$",  repl_h2, body, flags=re.MULTILINE)
    body = re.sub(r"^###\s+(\d+)\.(\d+)\s+(.+)$",    repl_h3, body, flags=re.MULTILINE)
    return body


COMBINED_FRONTMATTER = """## Abstract

This report presents an integrated case-study portfolio in **applied Bayesian
methodology for classification and inference**, drawing together two end-to-end
data-science projects that share a common methodological backbone: explicit
prior–data fusion, hierarchical structure for grouped data, MCMC inference with
convergence diagnostics, and calibrated probabilistic outputs tied to
operational decision rules.

**Case Study A — Calibrated Predictive Modeling (Wisconsin Diagnostic Breast
Cancer).** An eight-algorithm ensemble benchmark optimized with Bayesian
hyperparameter search (Optuna TPE), Platt-scaled probability calibration
(ECE 0.0312 → 0.0089, a **71.5%** reduction), and context-specific decision
thresholds. AdaBoost delivered **99.12%** held-out accuracy with **100%**
precision, **98.59%** recall, ROC-AUC **0.9987**, Cohen's κ **0.9823**, and
stable 10-fold cross-validation (98.46% ± 1.12%).

**Case Study B — Bayesian Hierarchical Inference at Scale (LLM-Ensemble
Textbook Bias Detection).** A three-model LLM ensemble (GPT-4o,
Claude-3.5-Sonnet, Llama-3.2-90B) produced **67,500** bias ratings across
**4,500** textbook passages and 5 publishers. Krippendorff's α = **0.84**
established multi-rater agreement; a PyMC hierarchical model with partial
pooling (R-hat **< 1.01**, ESS **> 3,000**) and a non-parametric Friedman
test (χ² = **42.73**, p **< 0.001**) identified **3 of 5** publishers with
publisher-level bias whose 95% HDIs excluded zero.

Together, the case studies illustrate how the same Bayesian-methods toolkit —
priors as regularization, partial pooling for grouped data, posterior-based
decision rules, calibrated probabilities, and quantified uncertainty — applies
across both high-stakes predictive modeling and GenAI evaluation. All
artifacts are aligned with IEEE 2830-2025, ISO/IEC 23894:2025, and the EU AI
Act.

**Keywords:** Bayesian Inference, Hierarchical Models, MCMC, Krippendorff's α,
Probability Calibration, Decision Policy, LLM-as-Judge, Ensemble Learning,
Optuna, Platt Scaling, SHAP, Responsible AI

---
## Executive Summary

| Dimension | Case Study A — WBCD | Case Study B — Textbook Bias |
|-----------|---------------------|------------------------------|
| **DS sub-discipline** | Predictive modeling + decision policy | GenAI evaluation + Bayesian inference |
| **Data** | 569 tumors × 30 features | 67,500 ratings × 4,500 passages × 5 publishers |
| **Headline accuracy/agreement** | 99.12% accuracy (AdaBoost) | Krippendorff's α = 0.84 |
| **Bayesian rigor signal** | ECE 0.0089 (Platt-scaled), Optuna TPE | R-hat < 1.01, ESS > 3,000, 95% HDI |
| **Statistical confirmation** | 10-fold CV = 98.46% ± 1.12% | Friedman χ² = 42.73, p < 0.001 |
| **Decision artifact** | Two operating points tied to cost ratios | Publisher-level posterior + 12.3% high-uncertainty triage |
| **Production deliverable** | Calibrated probability API + drift monitor | Quarterly per-publisher report card |

---
## 1. Introduction

The unifying claim of this report is methodological rather than topical: a
small kit of Bayesian techniques — **priors that regularize, partial pooling
that borrows strength across groups, MCMC that quantifies posterior
uncertainty, and calibration that makes probabilities decision-ready** —
applies to both classical predictive modeling and modern GenAI evaluation.

The two case studies that follow are deliberately drawn from different domains
and data scales to exercise the shared toolkit under contrasting conditions:

- **Case Study A** is a tabular, low-N, high-stakes binary classification
  problem with asymmetric error costs. Bayesian methods enter as Bayesian
  hyperparameter optimization (Optuna's TPE sampler), Bayesian (Platt-scaled)
  probability calibration, and threshold tuning for two operating points
  tied to decision cost ratios. ROC-AUC of 0.9987 and an ECE of 0.0089
  illustrate that the model is not just accurate but well-calibrated, which
  is the prerequisite for downstream decision policies.

- **Case Study B** is a high-N, high-dimensional inference problem over
  grouped data (passages within textbooks within publishers). Bayesian
  methods enter as a hierarchical model with partial pooling, MCMC
  inference (PyMC / NUTS), 95% HDI-based decision rules, and a Friedman
  test as a non-parametric confirmation. The methodology is layered on top
  of an LLM-ensemble (GPT-4o, Claude-3.5-Sonnet, Llama-3.2-90B) labelling
  pipeline whose multi-rater agreement (Krippendorff's α = 0.84) is the
  prerequisite for trusting downstream inference.

The remainder of the report presents each case study with its full
methodology, results, diagnostics, explainability analysis, and production
considerations, followed by a synthesis section that distils the shared
methodological lessons.

---
"""

COMBINED_SYNTHESIS = """
---
## Synthesis and Discussion

The two case studies illustrate that a single Bayesian methodology kit
generalizes across very different DS settings:

1. **Priors do real work.** In Case Study A, the TPE sampler's prior over
   the hyperparameter space (Optuna) reached the operating point in 45
   trials rather than ~240 for grid search — a 5× reduction in compute
   for the same calibration. In Case Study B, the publisher-level
   hyperprior shrunk small-cohort estimates toward the global mean,
   stabilizing inference where evidence was thin.

2. **Calibration and uncertainty are decision-grade outputs, not nice-to-haves.**
   Case Study A's ECE drop from 0.0312 to 0.0089 (71.5%) is what makes the
   downstream threshold policy (mass screening at 0.31, confirmation at
   0.62) auditable. Case Study B's 95% HDIs are what let the report
   credibly flag 3 of 5 publishers as biased while remaining honest about
   the other two.

3. **Hierarchy maps cleanly onto real organizational structure.** Tumors
   inside cohorts, passages inside publishers, customers inside segments —
   partial pooling is the right default when groups vary in evidence
   weight and an analyst needs honest small-group estimates.

4. **MCMC diagnostics are not optional.** R-hat < 1.01 and bulk- /
   tail-ESS > 400 (Vehtari et al., 2021) are the modern thresholds; in
   Case Study B both diagnostics cleared with margin (R-hat < 1.01, ESS
   > 3,000), giving the posterior credibility a downstream stakeholder
   can rely on.

5. **Calibrated probabilities + explicit decision rules + monitoring**
   close the loop. Both case studies ship with stakeholder-facing
   artifacts (operating-point spec, per-publisher report card),
   reproducibility artifacts (MLflow / model cards), and drift monitoring
   tied to the calibration / posterior — not just to accuracy. This is
   the difference between "we built a model" and "we deployed an
   auditable decision system."

---
## Conclusions

This integrated report demonstrates the methodological coherence of applied
Bayesian inference across two distinct domains. The headline numbers — 99.12%
accuracy with ECE 0.0089 in WBCD; Krippendorff's α 0.84, R-hat < 1.01, and 3/5
credibly-biased publishers in the textbook study — are the visible outputs,
but the durable contribution is the shared decision-grade workflow: explicit
priors, partial pooling for grouped data, calibrated probabilities,
HDI-based or cost-ratio-based decision rules, MCMC diagnostics as quality
gates, and stakeholder-facing artifacts (model cards, posterior plots,
operating-point specs) aligned with IEEE 2830-2025 / EU AI Act expectations
for regulated DS work.

The same toolkit underwrites both classical predictive modeling and GenAI
evaluation — a portable, audit-ready foundation for 2026 data-science
practice.
"""


def build_combined() -> str:
    """Compose the unified Bayesian-methods report (WBCD + Textbook Bias)."""
    # WBCD case study body: from "## 1. Introduction" through (but not including)
    # the References section. Re-number so its sections become 2..N in the host.
    wbcd_body = extract_body_from(
        SRC_WBCD,
        body_start_pattern=r"^## 1\. Introduction\s*$",
        body_end_pattern=r"^## References\s*$",
    )
    # Shift section numbering by +1 so "1." → "2." in the host doc.
    wbcd_body = renumber_sections(wbcd_body, offset=1)

    # Textbook-bias case study body: same slice strategy.
    bias_body = extract_body_from(
        SRC_BIAS,
        body_start_pattern=r"^## 1\. Introduction\s*$",
        body_end_pattern=r"^## References\s*$",
    )
    # Shift by the count of WBCD's section numbers used so far. WBCD body has
    # sections 1-12 in the source → 2-13 after offset. Bias body sections
    # become 14-26 (max source is "## 13." so offset is +13).
    bias_body = renumber_sections(bias_body, offset=13)

    header = PRO_HEADER.format(
        title="Bayesian Methods in Applied Classification: A Two-Case-Study Portfolio",
        subtitle=("Case Study A — Calibrated Predictive Modeling (WBCD) · "
                  "Case Study B — Bayesian Hierarchical Inference at Scale "
                  "(LLM-Ensemble Textbook Bias)"),
        version="1.0.0",
    )

    part_a_heading = (
        "## Part A — Case Study A: Calibrated Predictive Modeling (WBCD)\n\n"
        "*Sections 2 through 13 below originate from the Wisconsin Diagnostic "
        "Breast Cancer case study, re-numbered for inclusion in this combined "
        "report.*\n\n---\n"
    )
    part_b_heading = (
        "\n\n---\n## Part B — Case Study B: LLM-Ensemble Textbook Bias Detection\n\n"
        "*Sections 14 through 26 below originate from the LLM-Ensemble Textbook "
        "Bias Detection case study, re-numbered for inclusion in this combined "
        "report.*\n\n---\n"
    )

    return (
        header
        + COMBINED_FRONTMATTER
        + part_a_heading + wbcd_body
        + part_b_heading + bias_body
        + COMBINED_SYNTHESIS
    )


# ─── Render to MD + PDF ──────────────────────────────────────────────────────

def render(out_stem: str, md_text: str) -> None:
    """Write the source markdown as `<stem>_Report.md` and render the
    publication-formatted PDF as `<stem>_Publication.pdf`, matching the
    repository's established source/publication file convention."""
    md_path  = OUT / f"{out_stem}_Report.md"
    pdf_path = OUT / f"{out_stem}_Publication.pdf"
    md_path.write_text(md_text, encoding="utf-8")
    generate_pdf(str(md_path), str(pdf_path))
    print(f"  Source MD  : {md_path.relative_to(ROOT)}")
    print(f"  Publication: {pdf_path.relative_to(ROOT)} ({pdf_path.stat().st_size // 1024} KB)")


def main() -> int:
    print("=" * 70)
    print("Building combined + standalone project reports -> projects/")
    print("=" * 70)

    # 01: AI Safety, header-normalized only.
    ai_md = normalize_simple(
        SRC_AI_SAFETY,
        title="AI Safety Red-Team Evaluation: Technical Analysis Report",
        subtitle=("Automated Harm Detection Using LLM-Ensemble Annotation and "
                  "Bayesian ML Classification"),
        version="2.0.0",
    )
    render("01_AI_Safety_RedTeam_Evaluation", ai_md)

    # 02: Combined Bayesian Methods (WBCD + Textbook Bias).
    combined_md = build_combined()
    render("02_Bayesian_Methods_in_Applied_Classification", combined_md)

    print("=" * 70)
    print("Done. 2 reports rendered.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
