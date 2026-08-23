# ACT-NDT Framework

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-29%20passing-brightgreen)

A reproducible research framework for evaluating LLM-driven agent policies against
classical and learned baselines on 5G/6G radio-resource control tasks. Seven
scenarios share one physical-layer model, one configuration file, and one cost
logger, so results are comparable across experiments.

## Contents

- [Scenarios](#scenarios)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Full-scale runs](#full-scale-runs)
- [Optional backends](#optional-backends)
- [Statistical evaluation](#statistical-evaluation)
- [Plots](#plots)
- [Testing](#testing)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [License](#license)

## Scenarios

| # | Entry point | Description |
|---|---|---|
| 1 | `scenario1_baseline.py` | PHY validation with a fixed action; optional real digital-twin backend |
| 2 | `scenario2_react_agents.py` | ActOnly / ReAct / ReActMemory / Reflexion under a TDL-C→TDL-D channel shift |
| 3 | `scenario3_multicell_icic.py` | 3-cell ICIC, 4 policies, Jain fairness |
| 4 | `scenario4_mobility_blockage.py` | Pedestrian → vehicular → blockage phases, including a real DQN baseline |
| 5 | `scenario5_evo_optimizer.py` | Evolutionary policy search |
| 6 | `scenario6_neural_rx.py` | Neural receiver hyperparameter search (320 configurations) |
| 7 | `scenario7_cost_analysis.py` | Cost/performance meta-analysis across Scenarios 1–6 |

Scenarios 1–4 subclass the shared `BaseEnv` interface and emit per-step rows to
`cost_logger.py`, which Scenario 7 aggregates.

## Requirements

- Python 3.11
- PyTorch (Scenario 4's DQN only)
- Optional: [Ollama](https://ollama.com) or an OpenAI/Anthropic API key for real LLM reasoning
- Optional: a Sionna-capable workstation for real channel data

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Scenario 4's DQN baseline
pip install torch torchvision torchaudio
```

`requirements.txt` pins exact versions. To install without pinning:

```bash
pip install pyyaml numpy pandas scikit-learn tqdm matplotlib requests
```

## Quick start

Verify the installation, then run each scenario at reduced scale:

```bash
python test_framework.py            # full suite, ~1-2 min
python test_framework.py --fast     # skip scenario smoke tests

python scenario1_baseline.py --steps 20
python scenario2_react_agents.py --episodes 10 --shift-at 5 --variants actonly,reflexion
python scenario3_multicell_icic.py --episodes 10
python scenario4_mobility_blockage.py --episodes 20
python scenario5_evo_optimizer.py
python scenario6_neural_rx.py
python scenario7_cost_analysis.py   # run last; aggregates the others' cost logs
```

`run_everything.sh` runs the full sequence and regenerates plots after each
scenario, so `results/plots/` fills in incrementally rather than only at the end.

## Full-scale runs

```bash
python scenario2_react_agents.py --episodes 200 --shift-at 100
python scenario4_mobility_blockage.py --episodes 150
```

**Scenario 2 is the only scenario that makes LLM calls at volume.** Scenarios 1
and 3–7 are pure numeric simulation or local PyTorch and finish in minutes
whether or not an LLM backend is running; the ReAct/Reflexion policies in
Scenarios 3 and 4 are rule-based approximations of LLM reasoning by design.

At full scale, Scenario 2 issues 4 variants × 200 episodes × 100 steps =
**80,000 LLM calls** — roughly 22 hours at 1 s/call, 44–66 hours at 2–3 s/call.
Before committing to a long run:

```bash
./setup_and_validate.sh     # times an 80-call slice (1/1000th of the full run)
./run_full_experiments.sh   # wrap with `caffeinate -i` on macOS to prevent sleep
```

Each policy reports the fraction of steps that came from a real LLM response
rather than the heuristic fallback, and warns above 10%. Check this figure
before interpreting any cross-policy comparison.

## Optional backends

### LLM

Without a reachable backend, every policy falls back to a labeled heuristic and
all agent variants behave identically. `llm_client.py` cascades
Ollama → OpenAI → Anthropic → heuristic.

```bash
brew install ollama && ollama serve && ollama pull llama3
```

### Sionna

**Scenario 6** — run `sionna_rest_server.py` on a Sionna-capable workstation,
point `global.sionna_rest_url` in `config.yaml` at it, and set
`scenario6.use_synthetic_fallback: false`.

**Scenario 1** — pass `--api-url http://<host>:5000` to use a stateful digital
twin exposing `/health`, `/reset`, and `/step`. Only the client side ships with
this repository; the twin server does not.

## Statistical evaluation

`run_multiseed.py` repeats Scenarios 3, 4, and 5 across N seeds (default 10,
~11 minutes) and reports mean ± std per policy plus a Welch's t-test between
every pair.

```bash
python run_multiseed.py
python run_multiseed.py --include-scenario2 --s2-episodes 20
```

Scenario 2 is excluded by default because of its LLM-call cost.

## Plots

`generate_all_plots.py` reads whatever exists in `results/` and
`results/multiseed/`, writes PNGs to `results/plots/`, and skips anything not
yet run.

```bash
python generate_all_plots.py
python generate_all_plots.py --shift-at 100   # match a full 200-episode S2 run
```

Coverage: S1 reward/BLER/throughput over steps; S2 reward/BLER per episode per
policy with the channel-shift point marked; S3 reward/fairness by policy ×
traffic mode; S4 reward by policy across the three mobility/blockage phases;
S5 convergence curve; S6 accuracy by activation function and accuracy vs.
training time; S7 cost-vs-performance scatter; error bars for any multi-seed
data present.

## Testing

```bash
python test_framework.py            # 29 tests
python test_framework.py --fast     # unit tests only
```

Tests are deterministic and run against an isolated temporary log directory, so
they never contaminate `logs/`. CI (`.github/workflows/tests.yml`) runs `--fast`
on every push and the full suite on manual dispatch.

Not yet covered: full runners for Scenarios 4 and 5, and end-to-end assertions
on `usd_cost` accuracy.

## Configuration

`config.yaml` is the single source of truth for every tunable parameter; load it
through `config_loader.py` rather than importing values at module scope. Notable
defaults:

| Key | Value | Note |
|---|---|---|
| `global.base_snr_db` | `18.0` | Shifts every BLER curve; do not change without re-baselining |
| `global.seed` | — | Read at call time, not import time, so multi-seed runs work |
| `scenario4.pretrain_episodes` | `200` | Below ~200 the DQN is still near-random at comparison time |
| `scenario6.eval_epochs` | `15` | Lower values hit sklearn's convergence cap and bias config ranking |

BLER coefficients are centralized in `phy_models.py` but still differ per
scenario by design. Document that choice in any write-up that treats scenarios
as physically comparable.

## Project structure

```
config.yaml, config_loader.py    Single source of truth for parameters
phy_models.py                    Shared BLER/SNR/fitness/fairness kernels
base_env.py                      Abstract Observation/StepInfo/BaseEnv
base_reflexion_policy.py         Shared Reflexion lesson-generation logic
llm_client.py                    Ollama -> OpenAI -> Anthropic -> heuristic cascade
cost_logger.py                   Per-step wall-time/token/cost logging
dqn_baseline.py                  DQN with replay buffer and target network
scenario1_baseline.py            PHY validation, fixed action
scenario2_react_agents.py        ActOnly/ReAct/ReActMemory/Reflexion, TDL-C->D shift
scenario3_multicell_icic.py      3-cell ICIC, 4 policies, Jain fairness
scenario4_mobility_blockage.py   Pedestrian->vehicular->blockage, real DQN
scenario5_evo_optimizer.py       Evolutionary policy search
scenario6_neural_rx.py           Neural receiver hyperparameter search
scenario7_cost_analysis.py       Cost/performance meta-analysis
run_multiseed.py                 N-seed comparison with Welch's t-tests
generate_all_plots.py            Reads results/, writes PNGs to results/plots/
sionna_rest_server.py            Run on a Sionna workstation for real S6 data
test_framework.py                29 tests; --fast skips scenario smoke tests
pyproject.toml, requirements.txt Packaging; requirements.txt is pinned
.github/workflows/tests.yml      CI: --fast on push, full suite on dispatch
SETUP_MACOS.md                   Historical macOS setup detail
```

Outputs are written to `logs/` (cost logs), `results/` (CSV results), and
`results/plots/` (PNGs). All three are gitignored.



## License

No license file is currently included. Add a `LICENSE` before publishing.
