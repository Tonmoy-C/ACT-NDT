#!/usr/bin/env bash
# ============================================================================
# setup_and_validate.sh
#
# One-time environment setup + fast validation run for the AI Telco Engineer
# framework. Tailored for: Mac Studio 2025, Apple M3 Ultra, 96GB RAM,
# macOS Tahoe 26.x.
#
# What this script does NOT do: kick off the full 200-episode / 150-episode
# paper-scale runs. Those can realistically take hours with a real LLM
# connected (math explained at the end) - you want to see this script's
# timing numbers first and decide episode counts deliberately, not
# accidentally start a multi-hour job inside a "setup" script.
#
# Usage:
#   chmod +x setup_and_validate.sh
#   ./setup_and_validate.sh
# ============================================================================
set -e

ZIP_PATH="$HOME/Downloads/telco_framework_fixed.zip"
DEST_PARENT="$HOME/Documents"
BUNDLE_DIR="$DEST_PARENT/telco_framework"

echo "############################################################"
echo "# Phase 0 - Unpack"
echo "############################################################"
if [ ! -f "$ZIP_PATH" ]; then
  echo "Couldn't find $ZIP_PATH"
  echo "Update ZIP_PATH at the top of this script to wherever you saved the download."
  exit 1
fi
mkdir -p "$DEST_PARENT"
unzip -o "$ZIP_PATH" -d "$DEST_PARENT" > /dev/null
rm -rf "$BUNDLE_DIR"
mv "$DEST_PARENT/bundle" "$BUNDLE_DIR"
cd "$BUNDLE_DIR"
echo "Unpacked to $BUNDLE_DIR"

echo "############################################################"
echo "# Phase 1 - Homebrew"
echo "############################################################"
if ! command -v brew &> /dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
else
  echo "Homebrew already installed - skipping."
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
# PyTorch with Apple Metal (MPS) - your M3 Ultra's GPU cores are used automatically
pip install torch torchvision torchaudio -q

echo "############################################################"
echo "# Phase 4 - Ollama (real LLM reasoning, free & local)"
echo "############################################################"
brew install ollama
ollama --version

# Start the server in the background with logging, not a second terminal tab
nohup ollama serve > ollama.log 2>&1 &
echo $! > ollama.pid
echo "Waiting for Ollama to come up..."
for i in $(seq 1 20); do
  if curl -s http://localhost:11434 > /dev/null; then
    echo "Ollama is up."
    break
  fi
  sleep 1
done

echo "Pulling llama3.2 (fast model - this framework fires thousands of small"
echo "calls per full run, so call latency matters more than raw model size)."
ollama pull llama3.2

echo "############################################################"
echo "# Phase 5 - Verify MPS + run the full test suite"
echo "############################################################"
python -c "import torch; print('MPS (Apple GPU) available:', torch.backends.mps.is_available())"
python test_framework.py
# Expect: 29 passed | 0 failed

echo "############################################################"
echo "# Phase 6 - Timed validation run (small, real-LLM, ~1-2 min)"
echo "############################################################"
mkdir -p logs results data
echo "Timing a small Scenario 2 run (1 variant, 4 episodes, 20 steps = 80 real LLM calls)..."
time python scenario2_react_agents.py --episodes 4 --shift-at 2 --steps 20 --variants actonly

echo ""
echo "############################################################"
echo "# Setup complete."
echo "############################################################"
echo "Look at the 'real' time printed just above (Phase 6) - that's the"
echo "wall-clock cost of 80 real LLM calls on YOUR machine. Divide it by 80"
echo "to get your seconds-per-call, then see the README section this script"
echo "prints below for how that scales to a full run, and the exact commands"
echo "to kick one off safely (background + your Mac won't sleep mid-run)."
