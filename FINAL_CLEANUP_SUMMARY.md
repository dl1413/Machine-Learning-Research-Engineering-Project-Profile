# Final Repository Cleanup Summary

**Date:** June 17, 2026  
**Status:** Cleanup Plan for Production Release  
**Target:** Keep only 2 final projects + necessary documentation

---

## 📊 Current Repository State

### **Projects Directory** (`/projects/`)
Currently contains **3 PDFs**:
1. ✅ `01_AI_Safety_RedTeam_Evaluation.pdf` — **KEEP** (Project 1)
2. ✅ `02_Bayesian_Methods_in_Applied_Classification.pdf` — **KEEP** (Project 2)
3. ❌ *Implicit 3rd project reference* — **REMOVE from root**

### **LaTeX Directory** (`/latex/`)
Contains complete source materials:
- `AI_Safety_RedTeam_Evaluation.tex` + `.pdf`
- `Breast_Cancer_Classification.tex` + `.pdf` (older version)
- `LLM_Bias_Detection.tex` + `.pdf` (older version)
- `compile_latex.sh` (build script)
- `README.md` (documentation)

### **Root Directory Duplicates**
Old versions to be **DELETED**:
1. ❌ `AI_Safety_RedTeam_Evaluation_Publication.pdf` (duplicate in `/projects/`)
2. ❌ `Breast_Cancer_Classification_Publication.pdf` (superseded by `/projects/02_*`)
3. ❌ `LLM_Bias_Detection_Publication.pdf` (superseded by `/projects/02_*`)
4. ❌ `AI Safety Red-Team Evaluation_ Technical Analysis Report.md` (markdown version)
5. ❌ `Breast_Cancer_Classification_Report.md` (markdown version)
6. ❌ `LLM_Ensemble_Bias_Detection_Report.md` (markdown version)

---

## 🎯 Final Structure After Cleanup

```
Machine-Learning-Research-Engineering-Project-Profile/
├── README.md                          ✅ Main documentation
├── Resume_Derek_Lankeaux.md           ✅ Professional resume
├── APPLICATION_SNIPPETS.md            ✅ Job application guide
├── JOB_APPLICATIONS_PLAYBOOK.md       ✅ Strategy document
│
├── projects/                          ✅ FINAL PROJECTS
│   ├── 01_AI_Safety_RedTeam_Evaluation.pdf
│   ├── 02_Bayesian_Methods_in_Applied_Classification.pdf
│   └── README.md
│
├── latex/                             ✅ Source materials (archived)
│   ├── AI_Safety_RedTeam_Evaluation.tex
│   ├── AI_Safety_RedTeam_Evaluation.pdf
│   ├── Breast_Cancer_Classification.tex
│   ├── Breast_Cancer_Classification.pdf
│   ├── LLM_Bias_Detection.tex
│   ├── LLM_Bias_Detection.pdf
│   ├── compile_latex.sh
│   └── README.md
│
└── main branch ONLY                   ✅ Single stable branch
```

---

## 🗑️ Items to Delete

### **Files to Remove from Root (6 files)**
```bash
git rm "AI Safety Red-Team Evaluation_ Technical Analysis Report.md"
git rm "AI_Safety_RedTeam_Evaluation_Publication.pdf"
git rm "Breast_Cancer_Classification_Publication.pdf"
git rm "Breast_Cancer_Classification_Report.md"
git rm "LLM_Bias_Detection_Publication.pdf"
git rm "LLM_Ensemble_Bias_Detection_Report.md"
```

### **Branches to Delete (57 experimental branches)**
All `claude/*` prefixed branches should be removed to clean the repository history:

**Pattern 1: Implementation duplicates (6 branches)**
- `claude/implementation-plan`
- `claude/implementation-plan-again`
- `claude/implementation-plan-another-one`
- `claude/implementation-plan-for-repository`
- `claude/implementation-plan-for-repository-again`
- `claude/implement-projects-plan`

**Pattern 2: Heisenberg variants (30+ branches)**
- `claude/happy-heisenberg-2u03gr` through `claude/happy-heisenberg-wAwyU`

**Other experimental branches (15+ branches)**
- `claude/combine-llm-ensemble-projects`
- `claude/create-implementation-plan`
- `claude/create-latex-pdf-documents`
- `claude/data-science-proposal-am2yxm`
- `claude/explain-repository-structure`
- `claude/fix-commit-issue`
- `claude/fix-issue-in-data-processing`
- `claude/gracious-volta-9K2st`
- `claude/gracious-volta-aeFOy`
- `claude/move-latex-materials-to-folder`
- `claude/new-session-dZ3iV`
- `claude/optimize-files`
- `claude/optimize-github-projects-ASkiT`
- `claude/professionalize-projects-2026`
- `claude/project-portfolio`
- `claude/review-job-description-4afdA`
- `claude/sync-resume-and-readme`
- `claude/two-extensive-projects-2026`
- `claude/update-pull-request-template`

---

## 📋 Cleanup Checklist

- [ ] **Delete 6 root-level duplicate files** (via GitHub UI or `git rm`)
- [ ] **Delete 57 experimental branches** (keep only `main`)
- [ ] **Verify `/projects/` folder has only 2 final PDFs**
- [ ] **Archive `/latex/` folder** (keep for reference, mark as read-only)
- [ ] **Update README.md** with final 2-project portfolio
- [ ] **Final commit:** "cleanup: finalize portfolio with 2 main projects"

---

## 🎓 2 Final Projects Description

### **Project 1: AI Safety Red-Team Evaluation**
- **File:** `projects/01_AI_Safety_RedTeam_Evaluation.pdf`
- **Size:** 135 KB
- **Focus:** Adversarial red-team approaches for LLM safety testing
- **Skills:** AI safety, prompt engineering, evaluation methodology

### **Project 2: Bayesian Methods in Applied Classification**
- **File:** `projects/02_Bayesian_Methods_in_Applied_Classification.pdf`
- **Size:** 181 KB
- **Focus:** Bayesian inference for medical classification (cancer detection)
- **Skills:** Probabilistic modeling, statistical inference, healthcare ML

---

## ✅ Key Benefits of Cleanup

1. **Portfolio clarity** — Only 2 polished, production-ready projects
2. **Reduced clutter** — 57 fewer branches to maintain
3. **Smaller repo size** — ~1.2 MB reduction (6 duplicate files)
4. **Professional appearance** — Clean git history for employers
5. **Faster cloning** — Less overhead for recruiters reviewing work

---

## 🚀 Next Steps

1. Execute file deletions
2. Execute branch deletions (bulk or scripted)
3. Create final commit on `main`
4. Update portfolio links/documentation
5. Push cleaned repository to GitHub

---

**Target Outcome:** A pristine, 2-project ML/AI portfolio ready for 2026 job market
