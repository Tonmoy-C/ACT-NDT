#!/usr/bin/env bash
# ============================================================================
# run_everything.sh
#
# One paste, one script: setup + dependencies + Ollama + tests + all 7
# scenarios, in order, automatically. Tailored for Mac Studio 2025 /
# Apple M3 Ultra / 96GB / macOS Tahoe.
#
# Safe to paste directly into an interactive Terminal — everything runs
# inside a subshell ( ... ) below, so `set -e` and any early exit only
# stops this job, never your actual Terminal session.
#
# Want your Mac to stay awake for a long unattended run? Run this once,
# separately, first:
#     caffeinate -i &
#
# Usage as a saved script instead of a paste:
#     chmod +x run_everything.sh
#     nohup ./run_everything.sh > run_everything.log 2>&1 &
#     tail -f run_everything.log
# ============================================================================

(
set -e

# ---------------------------------------------------------------------
# Config you may want to change before running
# ---------------------------------------------------------------------
ZIP_PATH="$HOME/Downloads/telco_framework_fixed.zip"   # only used if NOT already inside a bundle folder
OLLAMA_MODEL="llama3.2"                                # fast model - see README
S2_EPISODES=40    # Scenario 2 is the only LLM-heavy one (4 variants x
                  # episodes x 100 steps real calls). Full paper scale is
                  # 200, which can run many hours. This default (40) keeps a
                  # first full pass to roughly a plausible afternoon instead
                  # of days - bump to 200 for the real paper-scale numbers
                  # once you've seen your machine's actual per-call speed
                  # (printed partway through this run).
S2_SHIFT_AT=$((S2_EPISODES / 2))

echo "############################################################"
echo "# Phase 0 - Locate the bundle"
echo "############################################################"
if [ -f "./config.yaml" ] && [ -f "./scenario1_baseline.py" ]; then
  # Already running from inside an unzipped bundle folder - just use it.
  # (This is the common case for a repeat run, and was previously broken:
  # the script used to always assume it needed to unzip a fresh download,
  # even when you were already sitting right where it needed to be.)
  BUNDLE_DIR="$(pwd)"
  echo "Already inside a bundle folder: $BUNDLE_DIR"
else
  BUNDLE_DIR="$HOME/Documents/telco_framework"
  if [ ! -f "$ZIP_PATH" ]; then
    echo "ERROR: not inside a bundle folder, and $ZIP_PATH not found."
    echo "Either cd into your unzipped bundle folder and re-run this script,"
    echo "or edit ZIP_PATH at the top of this script to point at your download."
    exit 1
  fi
  mkdir -p "$(dirname "$BUNDLE_DIR")"
  unzip -o "$ZIP_PATH" -d "$(dirname "$BUNDLE_DIR")" > /dev/null
  rm -rf "$BUNDLE_DIR"
  mv "$(dirname "$BUNDLE_DIR")/bundle" "$BUNDLE_DIR"
  cd "$BUNDLE_DIR"
  echo "Unpacked to $BUNDLE_DIR"
fi

echo "############################################################"
echo "# Phase 1 - Homebrew"
echo "############################################################"
if ! command -v brew &> /dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "Already installed - skipping."
fi

echo "############################################################"
echo "# Phase 2 - Python 3.11 + virtual environment"
echo "############################################################"
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
python --version

echo "############################################################"
echo "# Phase 3 - Python dependencies"
echo "############################################################"
pip install --upgrade pip -q
pip install pyyaml numpy pandas scikit-learn tqdm matplotlib requests scipy -q
pip install torch torchvision torchaudio -q   # MPS backend, uses your GPU cores

echo "############################################################"
echo "# Phase 4 - Ollama"
echo "############################################################"
brew install ollama
ollama --version
nohup ollama serve > ollama.log 2>&1 &
echo "Waiting for Ollama..."
for i in $(seq 1 20); do
  curl -s http://localhost:11434 > /dev/null && { echo "Ollama is up."; break; }
  sleep 1
done
ollama pull "$OLLAMA_MODEL"
sed -i '' "s/model: \".*\"/model: \"$OLLAMA_MODEL\"/" config.yaml 2>/dev/null || true

echo "############################################################"
echo "# Phase 5 - Verify MPS + run the full test suite"
echo "############################################################"
python -c "import torch; print('MPS (Apple GPU) available:', torch.backends.mps.is_available())"
python test_framework.py
mkdir -p logs results data

echo "############################################################"
echo "# Phase 6 - Fast scenarios first (no real LLM calls - minutes)"
echo "############################################################"
time python scenario1_baseline.py --steps 100
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

time python scenario3_multicell_icic.py --episodes 100
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

time python scenario4_mobility_blockage.py --episodes 150
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

time python scenario5_evo_optimizer.py
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

time python scenario6_neural_rx.py
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

echo "############################################################"
echo "# Phase 7 - Scenario 2 (the slow one: $S2_EPISODES episodes this run)"
echo "############################################################"
echo "Started at $(date)"
time python scenario2_react_agents.py --episodes "$S2_EPISODES" --shift-at "$S2_SHIFT_AT"
echo "Finished at $(date)"
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

echo "############################################################"
echo "# Phase 8 - Scenario 7: aggregate cost/performance across everything"
echo "############################################################"
python scenario7_cost_analysis.py
python generate_all_plots.py --shift-at "$S2_SHIFT_AT"

echo "############################################################"
echo "# ALL DONE."
echo "# Results: $BUNDLE_DIR/results"
echo "# Plots (updated after every scenario): $BUNDLE_DIR/results/plots"
echo "# Cost logs: $BUNDLE_DIR/logs"
echo "# To re-run Scenario 2 at full 200-episode paper scale:"
echo "#   cd $BUNDLE_DIR && source .venv/bin/activate"
echo "#   python scenario2_react_agents.py --episodes 200 --shift-at 100"
echo "#   python generate_all_plots.py --shift-at 100"
echo "############################################################"
)
