#!/usr/bin/env bash
# restructure.sh — reorganize Batched_GPU_Diff_MPC into core/ + experiments/ + agents/
set -euo pipefail

[[ -f mpc_controller.py ]] || { echo "Not in repo root (mpc_controller.py missing). Abort."; exit 1; }

if ! git diff-index --quiet HEAD --; then
  echo "Working tree not clean. Commit or stash first, then re-run."
  exit 1
fi

mkdir -p core
mkdir -p experiments/0001_hw_v1_bi_actuated
mkdir -p experiments/0002_hw_v6_sa010_single_actuated
mkdir -p agents/prompts
mkdir -p agents/lib
mkdir -p scripts
mkdir -p worker
mkdir -p .github/workflows

git mv mpc_controller.py    core/mpc_controller.py
git mv MPC_dynamics.py       core/MPC_dynamics.py
git mv true_dynamics.py      core/true_dynamics.py
git mv lin_net.py            core/lin_net.py
git mv Simulate.py           core/Simulate.py
git mv Simulate_batched.py   core/Simulate_batched.py
git mv batch_helpers.py      core/batch_helpers.py

git mv exp_hardware_v1_batched.py        experiments/0001_hw_v1_bi_actuated/experiment.py
git mv exp_hardware_v6_sa010_batched.py  experiments/0002_hw_v6_sa010_single_actuated/experiment.py

inject_path_fix() {
  local file="$1"
  python - <<PYEOF
import re
path = "$file"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
preamble = (
    "\n# --- path fix: make core/ modules importable when run from this folder ---\n"
    "import sys as _sys, os as _os\n"
    "_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'core'))\n"
    "# --- end path fix ---\n\n"
)
m = re.match(r'^(\s*"""[\s\S]*?"""\s*\n)', src)
if m:
    new_src = src[:m.end()] + preamble + src[m.end():]
else:
    new_src = preamble + src
with open(path, "w", encoding="utf-8") as f:
    f.write(new_src)
PYEOF
}

inject_path_fix experiments/0001_hw_v1_bi_actuated/experiment.py
inject_path_fix experiments/0002_hw_v6_sa010_single_actuated/experiment.py

cat > CLAUDE.md <<'CLAUDEEOF'
# Global agent constitution — placeholder

This file is read by every Claude Code invocation in this repo.
Will be filled in once the agent architecture is live. Until then,
no autonomous runs.
CLAUDEEOF

cat > progress.md <<'PROGEOF'
# Master agent progress log

Newest entries at top. Each tick appends one line summarizing the
decision taken and why.
PROGEOF

touch registry.jsonl
touch agents/master.md
touch agents/analyzer.md
touch agents/coder.md
touch agents/reviewer.md

cat > .gitignore <<'GITIGNOREEOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Local experiment artifacts (real ones live in R2)
logs/
saved_models/
*.pt
*.pth
*.ckpt

# OS
.DS_Store
Thumbs.db

# Secrets — never commit
.env
*.pem
GITIGNOREEOF

git rm -r --cached --ignore-unmatch logs/ __pycache__/ 2>/dev/null || true

git add CLAUDE.md progress.md registry.jsonl .gitignore
git add agents/

echo ""
echo "==============================================================="
echo "Restructure staged. Review with:"
echo "    git status"
echo "    git diff --stat HEAD"
echo ""
echo "If it looks right:"
echo "    git commit -m 'Restructure: core/ + experiments/ + agents/ layout'"
echo "    git push"
echo ""
echo "If anything looks wrong:"
echo "    git reset --hard HEAD"
echo "==============================================================="
