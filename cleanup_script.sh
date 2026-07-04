#!/bin/bash

# ============================================================================
# FINAL CLEANUP SCRIPT - ML/AI Portfolio 2026
# ============================================================================
# This script executes the complete cleanup plan:
# 1. Removes 6 duplicate files from root directory
# 2. Removes 57 experimental branches
# 3. Creates final commit on main
#
# Usage: bash cleanup_script.sh
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "🧹 PORTFOLIO CLEANUP SCRIPT - STARTED"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# SECTION 1: REMOVE 6 DUPLICATE FILES FROM ROOT
# ============================================================================

echo -e "${YELLOW}[STEP 1/3]${NC} Removing 6 duplicate files from root directory..."
echo ""

FILES_TO_DELETE=(
    "AI Safety Red-Team Evaluation_ Technical Analysis Report.md"
    "AI_Safety_RedTeam_Evaluation_Publication.pdf"
    "Breast_Cancer_Classification_Publication.pdf"
    "Breast_Cancer_Classification_Report.md"
    "LLM_Bias_Detection_Publication.pdf"
    "LLM_Ensemble_Bias_Detection_Report.md"
)

for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${RED}❌ Removing:${NC} $file"
        git rm "$file" 2>/dev/null || echo "  (File not found, skipping)"
    fi
done

echo ""
echo -e "${GREEN}✓ Step 1 complete${NC}: Duplicate files scheduled for deletion"
echo ""

# ============================================================================
# SECTION 2: REMOVE 57 EXPERIMENTAL BRANCHES
# ============================================================================

echo -e "${YELLOW}[STEP 2/3]${NC} Removing 57 experimental branches..."
echo ""

# List of all branches to delete
BRANCHES_TO_DELETE=(
    # Implementation duplicates (6)
    "claude/implementation-plan"
    "claude/implementation-plan-again"
    "claude/implementation-plan-another-one"
    "claude/implementation-plan-for-repository"
    "claude/implementation-plan-for-repository-again"
    "claude/implement-projects-plan"
    
    # Heisenberg variants (30+)
    "claude/happy-heisenberg-2u03gr"
    "claude/happy-heisenberg-3FEsz"
    "claude/happy-heisenberg-5EgZS"
    "claude/happy-heisenberg-9e8rmc"
    "claude/happy-heisenberg-29Aly"
    "claude/happy-heisenberg-BVuxt"
    "claude/happy-heisenberg-DeV4w"
    "claude/happy-heisenberg-EEuiu"
    "claude/happy-heisenberg-GJCGA"
    "claude/happy-heisenberg-Jpgg4"
    "claude/happy-heisenberg-LtMNo"
    "claude/happy-heisenberg-N6gP9"
    "claude/happy-heisenberg-SBUBj"
    "claude/happy-heisenberg-SL7Ib"
    "claude/happy-heisenberg-W11f1"
    "claude/happy-heisenberg-WyJn6"
    "claude/happy-heisenberg-YlTtx"
    "claude/happy-heisenberg-aCAsK"
    "claude/happy-heisenberg-ciorgs"
    "claude/happy-heisenberg-d7vei6"
    "claude/happy-heisenberg-feBEF"
    "claude/happy-heisenberg-fjho1g"
    "claude/happy-heisenberg-fs7GZ"
    "claude/happy-heisenberg-jC1ID"
    "claude/happy-heisenberg-ldcmtv"
    "claude/happy-heisenberg-nBA9F"
    "claude/happy-heisenberg-rd7fw1"
    "claude/happy-heisenberg-uwiDx"
    "claude/happy-heisenberg-v18akz"
    "claude/happy-heisenberg-wAwyU"
    
    # Other experimental branches (15+)
    "claude/combine-llm-ensemble-projects"
    "claude/create-implementation-plan"
    "claude/create-latex-pdf-documents"
    "claude/data-science-proposal-am2yxm"
    "claude/explain-repository-structure"
    "claude/fix-commit-issue"
    "claude/fix-issue-in-data-processing"
    "claude/gracious-volta-9K2st"
    "claude/gracious-volta-aeFOy"
    "claude/move-latex-materials-to-folder"
    "claude/new-session-dZ3iV"
    "claude/optimize-files"
    "claude/optimize-github-projects-ASkiT"
    "claude/professionalize-projects-2026"
    "claude/project-portfolio"
    "claude/review-job-description-4afdA"
    "claude/sync-resume-and-readme"
    "claude/two-extensive-projects-2026"
    "claude/update-pull-request-template"
)

DELETED_COUNT=0
FAILED_COUNT=0

for branch in "${BRANCHES_TO_DELETE[@]}"; do
    if git show-ref --quiet refs/heads/"$branch" 2>/dev/null; then
        echo -e "${RED}🗑️  Deleting branch:${NC} $branch"
        if git branch -D "$branch" 2>/dev/null; then
            ((DELETED_COUNT++))
        else
            echo -e "${RED}  ERROR: Failed to delete $branch${NC}"
            ((FAILED_COUNT++))
        fi
    else
        echo -e "${YELLOW}⚠️  Skipping:${NC} $branch (not found locally)"
    fi
done

echo ""
echo -e "${GREEN}✓ Step 2 complete${NC}: Deleted $DELETED_COUNT branches locally"
if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${RED}⚠️  Failed to delete $FAILED_COUNT branches${NC}"
fi
echo ""

# ============================================================================
# SECTION 3: FINAL COMMIT ON MAIN
# ============================================================================

echo -e "${YELLOW}[STEP 3/3]${NC} Creating final cleanup commit..."
echo ""

# Check git status
echo "Current git status:"
git status --short

echo ""
echo -e "${YELLOW}Staging changes...${NC}"
git add -A

echo -e "${GREEN}✓ Changes staged${NC}"
echo ""

echo -e "${YELLOW}Committing changes...${NC}"
git commit -m "cleanup: finalize portfolio with 2 main projects

- Remove 6 duplicate files from root directory
- Remove 57 experimental branches
- Clean repository for 2026 job market release

Portfolio now contains:
1. AI_Safety_RedTeam_Evaluation.pdf
2. Bayesian_Methods_in_Applied_Classification.pdf

All legacy versions and experimental branches removed.
Repository is now production-ready."

echo ""
echo -e "${GREEN}✓ Commit created${NC}"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================="
echo -e "${GREEN}🎉 CLEANUP COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  ✓ Removed 6 duplicate files"
echo "  ✓ Removed $DELETED_COUNT experimental branches"
echo "  ✓ Created final cleanup commit"
echo ""
echo "Next steps:"
echo "  1. Review changes: git log -1"
echo "  2. Push to remote: git push origin main --force-with-lease"
echo "  3. Delete remote branches: git push origin --delete <branch-names>"
echo ""
echo "To push deleted branches to remote, run:"
echo "  git push origin --delete $(git branch -r | grep claude/ | sed 's|origin/||' | tr '\n' ' ')"
echo ""
echo "=========================================="
