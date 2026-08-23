# Framework Optimization — Complete Setup Guide
## macOS Apple Silicon (M1 / M2 / M3)

> **Superseded — read README.md first.** This guide was originally written
> when only the shared fix modules existed and you had to hand-apply
> before/after snippets to your own scenario1-4/7 files. Those files are now
> included directly and pre-integrated (`scenario1_baseline.py` etc.) — you
> no longer need to manually apply the snippets below. This file is kept for
> the phase-by-phase historical detail on *why* each change was made, and
> because its dependency-install steps (Ollama, PyTorch/MPS) are still
> accurate. For setup, provenance, and honest caveats, start with README.md.

This guide walks through all 15 fixes in the correct order.
Follow each phase sequentially — later phases depend on earlier ones.

---

## Prerequisites (one-time)

```bash
# 1. Install Homebrew if not already present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python 3.11 (Apple Silicon native)
brew install python@3.11
python3.11 --version   # should print 3.11.x

# 3. Create a virtual environment inside your framework folder
cd /path/to/your/framework      # <-- change this to your actual folder
python3.11 -m venv .venv
source .venv/bin/activate       # always activate before running anything

# 4. Install all Python dependencies
pip install --upgrade pip
pip install pyyaml numpy scikit-learn tqdm pytest

# 5. PyTorch with Apple Metal (MPS) backend — native M1/M2/M3 acceleration
pip install torch torchvision torchaudio

# 6. Ollama — free local LLM that runs natively on Apple Silicon
brew install ollama
```

---

## Phase 1 — LLM Setup (fixes #1 and #10)

```bash
# Start the Ollama server (keep this terminal open)
ollama serve

# In a NEW terminal, pull a model (choose one):
ollama pull llama3          # recommended — best quality, ~4.7 GB
# ollama pull mistral       # faster, ~4.1 GB, slightly lower quality
# ollama pull phi3          # smallest, ~2.3 GB, good for testing

# Verify Ollama is working:
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3","prompt":"Say OK","stream":false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['response'])"
# Expected output:  OK

# Copy the fix files into your framework folder:
cp llm_client.py         /path/to/your/framework/
cp base_reflexion_policy.py /path/to/your/framework/
```

**What this fixes:**
- Issue #1:  `LLMClient.generate()` now calls real Ollama (llama3) instead of `_simple_heuristic_response()`
- Issue #10: Responses are parsed as JSON (`{"mcs": int, "pmi": int, "power_db": float}`)
             — no more fragile regex on freeform text

**In your existing scenario files**, replace every occurrence of:
```python
# OLD
response = self.llm.generate(prompt)
action = self._parse_action(response)   # or the regex call
```
with:
```python
# NEW
from llm_client import LLMClient
self._client = LLMClient()              # instantiate once in __init__

response = self._client.generate(prompt)
action   = response.action             # already a clean dict: {"mcs", "pmi", "power_db"}
tokens   = response.tokens_used        # for cost logging
```

---

## Phase 2 — Shared PHY Layer (fixes #6, #7, #8, #9)

```bash
cp phy_models.py /path/to/your/framework/
```

**In your existing scenario files**, replace the inline BLER sigmoid with:
```python
# OLD (example from S2)
bler = 1 / (1 + math.exp(0.9 * (eff_snr - (10 + 0.55 * mcs))))

# NEW — one canonical call, same profile key used in all scenarios
from phy_models import compute_bler, compute_effective_snr, tdl_channel_penalty_db

eff_snr = compute_effective_snr(base_snr, power_db, pmi, channel_penalty_db=penalty)
bler    = compute_bler(eff_snr, mcs, profile="default")   # S1/S2
# profile="mobility"  for S4
# profile="evolution" for S5
# profile="neural_rx" for S6
```

**For S5 mobility penalty** (fixes #8 — the 0.03 coefficient bug):
```python
# OLD  — penalty was negligible
penalty = 0.07 * mobility_kmh * 0.03   # gives only 0.126 dB at 60 km/h

# NEW  — use canonical function
penalty, blockage = tdl_channel_penalty_db(
    channel_model=channel_model,
    mobility_kmh=mobility_kmh,
    blockage_prob=0.02,
    rng=self.rng,
)
```

**For Scenario 5 fitness function** (fixes #7 — adds missing Eq.5 terms):
```python
# OLD  — three terms only
fitness = 0.55 * avg_tput - 8.0 * max(0, avg_bler - 0.10) + 0.45 * avg_reward

# NEW  — full Eq.5
from phy_models import compute_fitness_eq5
fitness = compute_fitness_eq5(
    avg_throughput=avg_tput,
    avg_bler=avg_bler,
    avg_reward=avg_reward,
    # For multi-cell, add: avg_fairness=..., lam_fair=0.3, avg_interference=..., lam_int=0.2
)
```

---

## Phase 3 — Config File (fixes #11)

```bash
cp config.yaml   /path/to/your/framework/
cp config_loader.py /path/to/your/framework/
```

**In each scenario file**, replace every hardcoded literal:
```python
# OLD
episodes       = 50
steps          = 100
shift_episode  = 25
bler_target    = 0.10

# NEW
from config_loader import CFG, get
episodes      = CFG["training"]["episodes"]
steps         = CFG["training"]["steps_per_episode"]
shift_episode = CFG["training"]["shift_episode"]
bler_target   = CFG["phy"]["bler_target"]
```

Edit `config.yaml` in one place to tune anything — no more hunting through script files.

---

## Phase 4 — Cost Logging (fixes #3)

```bash
cp cost_logger.py /path/to/your/framework/
mkdir -p logs results data
```

**Wrap your existing step loops** (do this in S1, S2, S3, S4):
```python
from cost_logger import CostLogger

logger = CostLogger(experiment="exp2")   # exp1 / exp2 / exp3 / exp4

for ep in range(episodes):
    obs = env.reset()
    for st in range(steps_per_episode):
        with logger.step(episode=ep, step=st):
            response = self._client.generate(prompt)
            obs, info = env.step(response.action)
            logger.record(
                reward=info.reward,
                bler=obs.bler,
                throughput=obs.throughput_mbps,
                tokens=response.tokens_used,
                sionna_calls=1,
            )

logger.save()   # writes logs/exp2_cost_log.csv
print(logger.summary())
```

Scenario 7 will then read `logs/exp*_cost_log.csv` files with real numbers.

---

## Phase 5 — BaseEnv Interface (fixes #12)

```bash
cp base_env.py /path/to/your/framework/
```

**Make each scenario environment inherit from BaseEnv:**
```python
# OLD
class MockSionnaEnv:
    def reset(self): ...
    def step(self, mcs, pmi, power_db): ...   # different signature per scenario!

# NEW
from base_env import BaseEnv, Observation, StepInfo

class MockSionnaEnv(BaseEnv):
    def reset(self) -> Observation:
        self._new_episode()
        return Observation(snr_db=self.base_snr, bler=0.0, throughput_mbps=0.0)

    def step(self, action: dict) -> tuple[Observation, StepInfo]:
        action = self._clip_action(action)   # free bounds enforcement
        # ... your existing logic ...
        self._advance()
        return obs, StepInfo(reward=reward, terminated=(self._step_count >= self.max_steps))

    def render(self) -> str:
        return f"SNR={self.current_snr:.1f} dB  BLER={self.current_bler:.3f}"
```

---

## Phase 6 — BaseReflexionPolicy (fixes #13)

```bash
cp base_reflexion_policy.py /path/to/your/framework/
```

**Refactor S2, S3, S4 policies to use the parent class:**
```python
from base_reflexion_policy import BaseReflexionPolicy

class Scenario2Policy(BaseReflexionPolicy):
    kpi_thresholds = {"bler": 0.10, "throughput": 5.0}

    def _lesson_from_kpi(self, kpi, value):
        if kpi == "bler" and value > self.kpi_thresholds["bler"]:
            return f"BLER {value:.3f} too high — lower MCS next episode."
        if kpi == "throughput" and value < self.kpi_thresholds["throughput"]:
            return f"Throughput {value:.1f} Mbps below target — try higher MCS if BLER allows."
        return ""

    def build_prompt(self, obs, history):
        return (
            f"Current network state:\n"
            f"  SNR: {obs.snr_db:.1f} dB\n"
            f"  BLER: {obs.bler:.3f}\n"
            f"  Throughput: {obs.throughput_mbps:.1f} Mbps\n"
            f"  Channel: {obs.channel_model}\n"
            f"Select optimal MCS (0-28), PMI (0-15), power_dB (10-30)."
        )
```

---

## Phase 7 — Real DQN (fixes #4)

```bash
cp dqn_baseline.py /path/to/your/framework/
```

**In your Scenario 4 runner:**
```python
from dqn_baseline import DQNAgent

# Pretrain DQN before the comparison window
dqn = DQNAgent(seed=42)
dqn.train(env, episodes=CFG["scenario4"]["dqn"]["pretrain_episodes"])
dqn.save("logs/dqn_s4.pt")

# During evaluation episodes:
dqn_action, _ = dqn.select_action(obs, greedy=True)   # no exploration
obs, info = env.step(dqn_action)
```

**Verify MPS (Apple Silicon GPU) is being used:**
```python
import torch
print(torch.backends.mps.is_available())   # should print True on M1/M2/M3
```

---

## Phase 8 — Scenario 5 Parallel Evolution (fixes #7, #8, #15)

```bash
cp scenario5_evo_optimizer.py /path/to/your/framework/
```

```bash
# Run Scenario 5 directly:
python scenario5_evo_optimizer.py
```

On a MacBook M2 with `parallel_workers: 4`, 20 candidates × 10 generations
runs in roughly 3–6 minutes instead of 20–30 minutes sequentially.

---

## Phase 9 — Scenario 6 Real Data + Exhaustive Grid (fixes #2, #5)

```bash
cp scenario6_neural_rx.py /path/to/your/framework/
mkdir -p data
pip install scikit-learn tqdm
```

**Option A — use the structured surrogate (no Sionna needed):**
Edit `config.yaml`:
```yaml
scenario6:
  use_synthetic_fallback: true
```
Then run:
```bash
python scenario6_neural_rx.py
```

**Option B — generate real Sionna data (requires Sionna install):**
```bash
pip install sionna
# Start your Sionna REST twin:
python your_sionna_server.py &
```
Edit `config.yaml`:
```yaml
scenario6:
  use_synthetic_fallback: false
  real_data_path: "data/s6_neural_rx_dataset.npz"
```
The script will auto-generate and cache the dataset on first run.

---

## Phase 10 — Tests and CI (fixes #14)

```bash
cp -r tests/ /path/to/your/framework/tests/
pip install pytest

# Run all tests:
python -m pytest tests/ -v

# Or without pytest:
python tests/test_framework.py
```

**Expected output (all green):**
```
  ✓  test_bler_valid_range
  ✓  test_bler_monotone_in_snr
  ✓  test_bler_monotone_in_mcs
  ✓  test_bler_profiles_consistent
  ✓  test_tdl_mobility_penalty
  ✓  test_tdl_shift_penalty
  ✓  test_effective_snr_power_sensitivity
  ✓  test_fitness_eq5_all_terms
  ✓  test_jains_fairness_equal
  ✓  test_jains_fairness_one_user
  ✓  test_base_env_abstract
  ✓  test_base_env_clip_action
  ✓  test_config_dot_access
  ✓  test_config_missing_key
  ✓  test_config_scenario5_fitness_keys
  ✓  test_parse_action_valid_json
  ✓  test_parse_action_embedded_json
  ✓  test_parse_action_clipping
  ✓  test_parse_action_fallback
  ✓  test_cost_logger_writes_csv
  ✓  test_s6_surrogate_has_signal
  ✓  test_s6_grid_size

22 passed  |  0 failed  |  22 total
```

---

## Summary: Files to Copy Into Your Framework

| File | Fixes |
|---|---|
| `phy_models.py` | #6, #7, #8, #9 |
| `base_env.py` | #12 |
| `config.yaml` | #11 |
| `config_loader.py` | #11 |
| `llm_client.py` | #1, #10 |
| `cost_logger.py` | #3 |
| `base_reflexion_policy.py` | #13 |
| `dqn_baseline.py` | #4 |
| `scenario5_evo_optimizer.py` | #7, #8, #15 |
| `scenario6_neural_rx.py` | #2, #5 |
| `tests/test_framework.py` | #14 |

---

## Verification Checklist

After completing all phases, verify:

- [ ] `python tests/test_framework.py` → 22 passed, 0 failed
- [ ] `ollama serve` running, `llama3` pulled
- [ ] `torch.backends.mps.is_available()` → `True`
- [ ] `logs/exp1_cost_log.csv` … `logs/exp6_cost_log.csv` all exist after runs
- [ ] `data/s6_neural_rx_dataset.npz` exists (auto-generated on first S6 run)
- [ ] S5 evaluation prints parallel worker output, completes in < 10 min
- [ ] S6 grid search evaluates exactly 320 configurations (exhaustive mode)
