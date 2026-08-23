#!/usr/bin/env bash
# ============================================================================
# run_full_experiments.sh
#
# Full paper-scale run of all 7 scenarios. Run setup_and_validate.sh first.
#
# Only Scenario 2 makes real LLM calls at volume (4 variants x 200 episodes
# x 100 steps = 80,000 calls) - Scenarios 1,3,4,5,6,7 are pure numeric/local
# PyTorch, so they finish in minutes regardless of Ollama. This script runs
# the fast ones first so you get most results quickly, then Scenario 2 last.
#
# `caffeinate -i` keeps your Mac awake for the duration (macOS will otherwise
# sleep and pause everything if the lid closes or it idles out).
#
# Usage:
#   chmod +x run_full_experiments.sh
#   nohup caffeinate -i ./run_full_experiments.sh > full_run.log 2>&1 &
#   tail -f full_run.log          # watch progress; safe to close this and
#                                  # come back later, the job keeps running
# ============================================================================
set -e
source .venv/bin/activate
mkdir -p logs results data

echo "=== Fast scenarios first (no real LLM calls - minutes, not hours) ==="
time python scenario1_baseline.py --steps 100
time python scenario3_multicell_icic.py --episodes 100
time python scenario4_mobility_blockage.py --episodes 150
time python scenario5_evo_optimizer.py
time python scenario6_neural_rx.py

echo "=== Scenario 2: the slow one (80,000 real LLM calls) ==="
echo "Started at $(date)"
time python scenario2_react_agents.py --episodes 200 --shift-at 100
echo "Finished at $(date)"

echo "=== Scenario 7: aggregate cost/performance across everything above ==="
python scenario7_cost_analysis.py

echo "All done. Results in ./results, per-step cost logs in ./logs."
