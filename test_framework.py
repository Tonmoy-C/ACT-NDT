"""
tests/test_framework.py  —  Seeded unit tests for all framework modules.

Fixes addressed:
  #14 — CI-ready unit tests; all tests are deterministic (seed=42).

Run:
    cd telco_framework_fixes
    python -m pytest tests/ -v

Or for quick smoke test without pytest:
    python tests/test_framework.py
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# PHY models
# ─────────────────────────────────────────────────────────────────────────────

def test_bler_valid_range():
    from phy_models import compute_bler
    for mcs in [0, 10, 20, 28]:
        for snr in [-5.0, 0.0, 15.0, 30.0]:
            b = compute_bler(snr, mcs)
            assert 0.001 <= b <= 0.999, f"BLER {b} out of [0.001, 0.999] for snr={snr} mcs={mcs}"


def test_bler_monotone_in_snr():
    """Higher SNR → lower BLER (for fixed MCS)."""
    from phy_models import compute_bler
    mcs = 14
    blers = [compute_bler(snr, mcs) for snr in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]]
    for i in range(len(blers) - 1):
        assert blers[i] >= blers[i+1] - 1e-6, \
            f"BLER not monotone at SNR step {i}: {blers[i]:.4f} vs {blers[i+1]:.4f}"


def test_bler_monotone_in_mcs():
    """Higher MCS → higher BLER (for fixed SNR)."""
    from phy_models import compute_bler
    snr = 15.0
    blers = [compute_bler(snr, mcs) for mcs in range(0, 29, 4)]
    for i in range(len(blers) - 1):
        assert blers[i] <= blers[i+1] + 1e-6, \
            f"BLER not monotone at MCS step {i}: {blers[i]:.4f} vs {blers[i+1]:.4f}"


def test_bler_profiles_consistent():
    """All profiles produce valid BLER and differ from each other."""
    from phy_models import compute_bler, BLER_PROFILES
    results = {}
    for profile in BLER_PROFILES:
        b = compute_bler(15.0, 14, profile=profile)
        assert 0.001 <= b <= 0.999
        results[profile] = b
    # Profiles must not all be identical
    vals = list(results.values())
    assert max(vals) - min(vals) > 1e-6, "All BLER profiles produce identical output"


def test_tdl_mobility_penalty():
    """Fix #8 — mobility coefficient: 60 km/h must give >= 4.0 dB penalty."""
    from phy_models import tdl_channel_penalty_db
    penalty, _ = tdl_channel_penalty_db(
        channel_model="TDL-C", mobility_kmh=60.0, blockage_prob=0.0
    )
    assert penalty >= 4.0, f"Mobility penalty {penalty} dB too low for 60 km/h (expected >= 4.0)"


def test_tdl_shift_penalty():
    """Fix #9 — TDL-D shift adds a consistent dB penalty."""
    from phy_models import tdl_channel_penalty_db
    pen_c, _ = tdl_channel_penalty_db("TDL-C", mobility_kmh=0, blockage_prob=0)
    pen_d, _ = tdl_channel_penalty_db("TDL-D", mobility_kmh=0, blockage_prob=0)
    assert pen_d > pen_c, "TDL-D must have higher penalty than TDL-C"


def test_effective_snr_power_sensitivity():
    """Increasing power must increase effective SNR."""
    from phy_models import compute_effective_snr
    snr_low  = compute_effective_snr(15.0, power_db=15.0)
    snr_high = compute_effective_snr(15.0, power_db=25.0)
    assert snr_high > snr_low


def test_fitness_eq5_all_terms():
    """Fix #7 — fitness must change when fairness/interference/cost terms are non-zero."""
    from phy_models import compute_fitness_eq5
    f_base = compute_fitness_eq5(10.0, 0.05, 2.0)
    f_fair = compute_fitness_eq5(10.0, 0.05, 2.0, avg_fairness=0.5, lam_fair=5.0)
    f_int  = compute_fitness_eq5(10.0, 0.05, 2.0, avg_interference=3.0, lam_int=0.5)
    assert f_fair != f_base, "Fairness term has no effect"
    assert f_int  != f_base, "Interference term has no effect"


def test_jains_fairness_equal():
    """Jain's index = 1.0 when all throughputs are equal."""
    from phy_models import jains_fairness
    tp = np.array([10.0, 10.0, 10.0])
    ji = jains_fairness(tp)
    assert abs(ji - 1.0) < 1e-6, f"Jain's index for equal TP = {ji}, expected 1.0"


def test_jains_fairness_one_user():
    """Jain's index = 1.0 for single user."""
    from phy_models import jains_fairness
    assert abs(jains_fairness(np.array([5.0])) - 1.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Base environment
# ─────────────────────────────────────────────────────────────────────────────

def test_base_env_abstract():
    """BaseEnv cannot be instantiated directly."""
    from base_env import BaseEnv
    try:
        _ = BaseEnv({})  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_base_env_clip_action():
    """_clip_action enforces valid ranges."""
    from base_env import BaseEnv, Observation, StepInfo

    class ConcreteEnv(BaseEnv):
        def reset(self): return Observation()
        def step(self, action): return Observation(), StepInfo()
        def render(self): return ""

    env = ConcreteEnv({})
    clipped = env._clip_action({"mcs": 99, "pmi": -5, "power_db": 100.0})
    assert clipped["mcs"] == 28
    assert clipped["pmi"] == 0
    assert clipped["power_db"] == 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def test_config_dot_access():
    from config_loader import get
    seed = get("global.seed")
    assert seed == 42, f"Expected seed=42, got {seed}"


def test_config_missing_key():
    from config_loader import get
    val = get("nonexistent.key.deep", default="fallback")
    assert val == "fallback"


def test_config_scenario5_fitness_keys():
    from config_loader import CFG
    required = {"lam_tp", "lam_bler", "lam_reward", "lam_fair", "lam_int", "lam_cost", "bler_target"}
    actual   = set(CFG["scenario5"]["fitness"].keys())
    missing  = required - actual
    assert not missing, f"Missing fitness keys in config: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM client  (offline / no Ollama needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_action_valid_json():
    """JSON parser extracts correct action from well-formed response."""
    import importlib, types
    # Import the private function
    import llm_client as lc
    action, err = lc._parse_action('{"mcs": 14, "pmi": 2, "power_db": 22.5}')
    assert action["mcs"] == 14
    assert action["pmi"] == 2
    assert abs(action["power_db"] - 22.5) < 1e-6
    assert err == ""


def test_parse_action_embedded_json():
    """Parser finds JSON block inside prose."""
    import llm_client as lc
    raw = 'I recommend setting {"mcs": 20, "pmi": 4, "power_db": 25.0} for this channel.'
    action, err = lc._parse_action(raw)
    assert action["mcs"] == 20


def test_parse_action_clipping():
    """Out-of-range values are clipped."""
    import llm_client as lc
    action, _ = lc._parse_action('{"mcs": 99, "pmi": -1, "power_db": 5.0}')
    assert action["mcs"] == 28
    assert action["pmi"] == 0
    assert action["power_db"] == 10.0


def test_parse_action_fallback():
    """Unparseable response returns safe default without raising."""
    import llm_client as lc
    action, err = lc._parse_action("I have no idea what to do here.")
    assert action == lc._SAFE_ACTION
    assert err != ""


# ─────────────────────────────────────────────────────────────────────────────
# Cost logger
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_logger_writes_csv(tmp_path, monkeypatch=None):
    """Logger creates a valid CSV after save()."""
    import csv as _csv
    from pathlib import Path

    # Patch log dir to tmp
    import config_loader
    orig_cfg = config_loader.CFG["global"]["log_dir"]
    config_loader.CFG["global"]["log_dir"] = str(tmp_path)

    from cost_logger import CostLogger
    logger = CostLogger(experiment="test_exp", provider="ollama")
    with logger.step(episode=0, step=0):
        logger.record(reward=1.5, bler=0.08, throughput=12.3, tokens=50, sionna_calls=1)
    with logger.step(episode=0, step=1):
        logger.record(reward=2.0, bler=0.06, throughput=14.0, tokens=60)
    path = logger.save()

    assert path.exists()
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["experiment"] == "test_exp"
    assert float(rows[0]["reward"]) == 1.5

    config_loader.CFG["global"]["log_dir"] = orig_cfg


# ─────────────────────────────────────────────────────────────────────────────
# base_reflexion_policy - regression test for the memory_window bug
# (previously hardcoded to scenario2's config for every subclass; fixed to
# accept scenario_key so each scenario gets its own value.)
# ─────────────────────────────────────────────────────────────────────────────

def test_reflexion_policy_per_scenario_memory_window():
    from base_reflexion_policy import BaseReflexionPolicy

    class Dummy(BaseReflexionPolicy):
        kpi_thresholds = {"bler": 0.10}
        def _lesson_from_kpi(self, kpi, value): return ""
        def build_prompt(self, obs, history): return "x"

    import config_loader
    config_loader.CFG["scenario3"]["memory_window"] = 7
    d2 = Dummy(scenario_key="scenario2")
    d3 = Dummy(scenario_key="scenario3")
    assert d2._memory_window == config_loader.CFG["scenario2"]["memory_window"]
    assert d3._memory_window == 7, "scenario3 must NOT silently inherit scenario2's memory_window"


def test_reflexion_policy_has_episode_lifecycle():
    """Regression test: BaseReflexionPolicy must provide on_episode_start,
    not just on_episode_end (crashed Scenario 3's runner before this fix)."""
    from base_reflexion_policy import BaseReflexionPolicy

    class Dummy(BaseReflexionPolicy):
        kpi_thresholds = {}
        def _lesson_from_kpi(self, kpi, value): return ""
        def build_prompt(self, obs, history): return "x"

    d = Dummy()
    d.on_episode_start(0)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# dqn_baseline - had ZERO test coverage before this bundle
# ─────────────────────────────────────────────────────────────────────────────

def test_dqn_agent_constructs_and_selects_action():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  (skipped: torch not installed)")
        return
    from dqn_baseline import DQNAgent, N_ACTIONS
    from base_env import Observation

    agent = DQNAgent(seed=42)
    obs = Observation(snr_db=15.0, bler=0.1, throughput_mbps=10.0, mobility_kmh=0.0)
    action, idx = agent.select_action(obs, greedy=True)
    assert 0 <= idx < N_ACTIONS
    assert set(action.keys()) == {"mcs", "pmi", "power_db"}


# ─────────────────────────────────────────────────────────────────────────────
# Reconstructed scenario runners - smoke tests only (verifies the pipeline
# runs end-to-end; does not assert specific numeric outcomes, since those
# depend on whether a real LLM/Sionna backend is connected).
# ─────────────────────────────────────────────────────────────────────────────

def _with_isolated_log_dir(fn):
    """Run fn() with CFG['global']['log_dir'] pointed at a throwaway temp dir,
    then restore it. Confirmed-needed fix: a real run showed exp1's cost log
    with 105 sionna_calls instead of 100 - the 5-step smoke test below was
    silently appending into the SAME logs/exp1_cost_log.csv the real 100-step
    run also writes to, because CostLogger's log_dir comes from global config,
    independent of each test's own out_dir. Every scenario smoke test must
    isolate this or it will keep contaminating real results."""
    import tempfile
    import config_loader
    orig = config_loader.CFG["global"]["log_dir"]
    with tempfile.TemporaryDirectory() as tmp:
        config_loader.CFG["global"]["log_dir"] = tmp
        try:
            fn()
        finally:
            config_loader.CFG["global"]["log_dir"] = orig


def test_scenario1_smoke():
    from scenario1_baseline import run_scenario1
    def _run():
        df = run_scenario1(num_steps=5, out_dir="/tmp/s1_smoke_test")
        assert len(df) == 5
        assert {"reward", "bler", "throughput"}.issubset(df.columns)
    _with_isolated_log_dir(_run)


def test_scenario2_smoke():
    from scenario2_react_agents import run_scenario2
    def _run():
        df = run_scenario2(episodes=2, shift_at=1, variants=["actonly"], num_steps=5, out_dir="/tmp/s2_smoke_test")
        assert len(df) == 2
    _with_isolated_log_dir(_run)


def test_scenario3_smoke():
    from scenario3_multicell_icic import run_scenario3
    def _run():
        df = run_scenario3(episodes=2, num_steps=5, out_dir="/tmp/s3_smoke_test")
        assert (df["avg_fairness"] >= 0).all() and (df["avg_fairness"] <= 1.0001).all()
    _with_isolated_log_dir(_run)


def test_scenario7_placeholder_safe():
    """Scenario 7 must not crash even if no cost logs exist yet. Deliberately
    NOT log-dir-isolated - this test's whole point is checking behaviour
    against whatever logs/ genuinely contains right now."""
    from scenario7_cost_analysis import run_scenario7
    df = run_scenario7(out_dir="/tmp/s7_smoke_test")
    assert len(df) == 6
    assert "status" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 data
# ─────────────────────────────────────────────────────────────────────────────

def test_s6_surrogate_has_signal():
    """Fix #2 — surrogate data must be learnable (acc > 25% baseline)."""
    from scenario6_neural_rx import _generate_structured_surrogate
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y = _generate_structured_surrogate(n_samples=800, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    model  = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)
    model.fit(scaler.fit_transform(X_tr), y_tr)
    acc = model.score(scaler.transform(X_te), y_te)

    assert acc > 0.40, (
        f"Surrogate data has no learnable signal: acc={acc:.3f} barely above random (0.25). "
        "Check _generate_structured_surrogate()."
    )


def test_s6_grid_size():
    """Fix #5 — exhaustive grid must cover all 320 combinations."""
    import config_loader
    config_loader.CFG["scenario6"]["search_strategy"] = "exhaustive"
    from scenario6_neural_rx import build_grid
    configs = build_grid()
    assert len(configs) == 320, f"Expected 320 grid points, got {len(configs)}"


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback
    tests = [
        test_bler_valid_range,
        test_bler_monotone_in_snr,
        test_bler_monotone_in_mcs,
        test_bler_profiles_consistent,
        test_tdl_mobility_penalty,
        test_tdl_shift_penalty,
        test_effective_snr_power_sensitivity,
        test_fitness_eq5_all_terms,
        test_jains_fairness_equal,
        test_jains_fairness_one_user,
        test_base_env_abstract,
        test_base_env_clip_action,
        test_config_dot_access,
        test_config_missing_key,
        test_config_scenario5_fitness_keys,
        test_parse_action_valid_json,
        test_parse_action_embedded_json,
        test_parse_action_clipping,
        test_parse_action_fallback,
        test_s6_surrogate_has_signal,
        test_s6_grid_size,
        test_reflexion_policy_per_scenario_memory_window,
        test_reflexion_policy_has_episode_lifecycle,
        test_dqn_agent_constructs_and_selects_action,
    ]

    # Slower smoke tests - actually run each reconstructed scenario end to end.
    # Kept separate so a quick `python test_framework.py --fast` can skip them.
    scenario_smoke_tests = [
        test_scenario1_smoke,
        test_scenario2_smoke,
        test_scenario3_smoke,
        test_scenario7_placeholder_safe,
    ]
    if "--fast" not in sys.argv:
        tests += scenario_smoke_tests


    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗  {t.__name__}")
            traceback.print_exc()
            failed += 1

    # Cost logger test needs tmp_path simulation
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            test_cost_logger_writes_csv(td)
            print(f"  ✓  test_cost_logger_writes_csv")
            passed += 1
        except Exception:
            print(f"  ✗  test_cost_logger_writes_csv")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed  |  {failed} failed  |  {passed+failed} total")
    sys.exit(0 if failed == 0 else 1)
