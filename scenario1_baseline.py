"""
scenario1_baseline.py — PHY validation baseline.

Reconstructed from the docx code-walkthrough (not your original byte-for-byte
file — see README.md "Provenance" section), now wired to the shared fixed
modules: BaseEnv, phy_models, cost_logger.

Purpose: confirm the PHY twin behaves physically consistently under a
completely fixed, non-adaptive controller (mcs=15, pmi=0, power_db=20.0,
every step, 100 steps), so later Scenarios' gains/failures can be attributed
to controller design rather than simulator instability.

Dual pathway (matches the original design's env_client.py pattern used in
S2/S4): if a real Sionna twin REST server is reachable at --api-url
(endpoints /health, /reset, /step), use it. Otherwise fall back to the
local analytic PHY model in phy_models.py — same one used by every other
scenario, so BLER/throughput are computed identically either way.

HONEST NOTE: the docx walkthrough describes Scenario 1's real numbers
(reward 17.41+/-2.17, BLER 0.082+/-0.041) as coming from genuine Sionna
MIMO-OFDM simulation via the REST twin, not from a documented local
formula. The analytic fallback here is the same simplified sigmoid model
used across the rest of the framework (via phy_models.py) - it will NOT
numerically reproduce those exact Sionna-derived numbers, because it's a
deliberately simpler stand-in, not a re-tuned copy of real Sionna physics.
Connect --api-url to a real twin for numbers you can report; use the
fallback for fast offline iteration and pipeline testing.

Run:
    python scenario1_baseline.py
    python scenario1_baseline.py --api-url http://localhost:5000
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import CFG
from base_env import BaseEnv, Observation, StepInfo
from phy_models import compute_bler, compute_effective_snr, compute_reward_single_link, spectral_efficiency
from cost_logger import CostLogger

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

FIXED_ACTION = {"mcs": 15, "pmi": 0, "power_db": 20.0}


class Scenario1Env(BaseEnv):
    """Single-link PHY twin. Uses a real Sionna REST server if reachable,
    otherwise the shared analytic BLER model (phy_models.compute_bler)."""

    def __init__(self, config: dict, rng_seed: int = 42, api_url: str | None = None):
        super().__init__(config, rng_seed)
        self.base_snr = CFG["phy"]["base_snr_db"]
        self.bler_target = CFG["phy"]["bler_target"]
        self.max_steps = config.get("steps_per_episode", 100)
        self.api_url = api_url
        self.use_api = False
        if api_url and _HAS_REQUESTS:
            try:
                r = requests.get(f"{api_url}/health", timeout=2)
                self.use_api = r.status_code == 200
            except Exception:
                self.use_api = False
        self._last_snr = self.base_snr

    def reset(self) -> Observation:
        self._new_episode()
        if self.use_api:
            try:
                requests.post(f"{self.api_url}/reset", timeout=5)
            except Exception:
                self.use_api = False
        return Observation(snr_db=self.base_snr, bler=0.0, throughput_mbps=0.0,
                            channel_model="TDL-C", step=0)

    def step(self, action: dict) -> tuple[Observation, StepInfo]:
        action = self._clip_action(action)

        if self.use_api:
            try:
                resp = requests.post(f"{self.api_url}/step", json=action, timeout=5).json()
                bler = float(resp["metrics"]["bler"])
                tput = float(resp["metrics"]["throughput_mbps"])
                reward = float(resp["reward"])
                snr = float(resp.get("metrics", {}).get("snr_db", self._last_snr))
            except Exception:
                self.use_api = False
                bler, tput, reward, snr = self._local_step(action)
        else:
            bler, tput, reward, snr = self._local_step(action)

        self._last_snr = snr
        self._advance()
        obs = Observation(snr_db=snr, bler=bler, throughput_mbps=tput,
                           channel_model="TDL-C", step=self._step_count)
        info = StepInfo(reward=reward, terminated=(self._step_count >= self.max_steps))
        return obs, info

    def _local_step(self, action: dict):
        noisy_snr = self.base_snr + self.rng.normal(0, 1.5)
        eff_snr = compute_effective_snr(noisy_snr, power_db=action["power_db"], pmi=action["pmi"])
        bler = compute_bler(eff_snr, action["mcs"], profile="default")
        se = spectral_efficiency(action["mcs"])
        tput = se * 20.0 * (1.0 - bler)  # 20 MHz BW, matches phy_models convention
        reward = compute_reward_single_link(tput, bler, bler_target=self.bler_target)
        return bler, tput, reward, noisy_snr

    def render(self) -> str:
        return f"SNR={self._last_snr:.1f} dB  step={self._step_count}/{self.max_steps}  api={self.use_api}"


def run_scenario1(num_steps: int = 100, api_url: str | None = None, out_dir: str = "results") -> pd.DataFrame:
    env = Scenario1Env(config={"steps_per_episode": num_steps}, rng_seed=CFG["global"]["seed"], api_url=api_url)
    logger = CostLogger(experiment="exp1")

    print(f"[S1] {'Real Sionna twin via REST' if env.use_api else 'Local analytic PHY model (phy_models.py)'}")
    obs = env.reset()
    rows = []
    for step in range(num_steps):
        with logger.step(episode=0, step=step):
            obs, info = env.step(FIXED_ACTION)
            logger.record(reward=info.reward, bler=obs.bler, throughput=obs.throughput_mbps, sionna_calls=1)
        rows.append({"step": step, "reward": info.reward, "bler": obs.bler, "throughput": obs.throughput_mbps})
        if info.terminated:
            break
    logger.save()

    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp1_baseline_results.csv", index=False)

    print(f"\n[S1] Saved {len(df)} rows to {out / 'exp1_baseline_results.csv'}")
    print(df[["reward", "bler", "throughput"]].describe().loc[["mean", "std", "min", "max"]])
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=CFG["training"]["steps_per_episode"])
    p.add_argument("--api-url", type=str, default=None, help="e.g. http://localhost:5000 for a real Sionna twin")
    args = p.parse_args()
    run_scenario1(num_steps=args.steps, api_url=args.api_url)
