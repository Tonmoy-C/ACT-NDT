"""
scenario3_multicell_icic.py — Multi-cell ICIC: 3 cells, 7 UEs, 4 coordination policies.

Reconstructed from the docx walkthrough (exp3_env.py's MultiCellTelcoEnv,
the four ICIC policy classes, run_episode/run_scenario orchestration).

Fixes applied during reconstruction:
  - ReflexionICICPolicy now subclasses base_reflexion_policy.BaseReflexionPolicy
    (fix #13), with its own scenario3 memory_window (was silently reading
    scenario2's in the earlier fix bundle - see README).
  - Cost logging via cost_logger.py (fix #3).
  - Jain's fairness computed via phy_models.jains_fairness (single source
    of truth, fix #6/#8 spirit - same formula, no re-derivation per scenario).

Run:
    python scenario3_multicell_icic.py --episodes 100
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config_loader import CFG
from base_env import BaseEnv, Observation, StepInfo
from base_reflexion_policy import BaseReflexionPolicy
from phy_models import jains_fairness, compute_reward_multi_cell
from cost_logger import CostLogger

_S3 = CFG["scenario3"]
NUM_CELLS = _S3["num_cells"]
UE_CELLS = [0, 0, 1, 1, 2, 2, 2]           # 7 UEs, 2/2/3 split
NUM_UES = len(UE_CELLS)
NOISE_DBM = -94.0


def _pathloss_db(distance_m: float) -> float:
    """3GPP UMa pathloss: 128.1 + 37.6*log10(d_km), per TR 38.901."""
    return 128.1 + 37.6 * math.log10(max(distance_m, 35.0) / 1000.0)


class MultiCellTelcoEnv(BaseEnv):
    """3-cell Urban Macro twin with interference, handover, and Jain fairness."""

    def __init__(self, config: dict, rng_seed: int = 42, traffic_mode: str = "static"):
        super().__init__(config, rng_seed)
        self.traffic_mode = traffic_mode
        self.cells = [{"cell_id": c, "power_dbm": 43.0, "beamforming": False, "ho_margin_db": 3.0}
                       for c in range(NUM_CELLS)]
        self.ues = [{"ue_id": u, "serving_cell": UE_CELLS[u],
                     "distance_m": float(self.rng.uniform(80, 450)),
                     "load": 1.0, "last_handover_step": -10} for u in range(NUM_UES)]
        self.step_id = 0

    def reset(self) -> Observation:
        self._new_episode()
        self.step_id = 0
        return Observation(step=0, extra={"cells": self.cells, "ues": self.ues})

    def _update_traffic(self) -> None:
        if self.traffic_mode == "dynamic":
            for ue in self.ues:
                burst = self.rng.poisson(5) / 10.0
                size_factor = self.rng.exponential(1.0)
                ue["load"] = float(np.clip(0.25 + 0.08 * burst + 0.12 * size_factor, 0.2, 1.6))
        else:
            for ue in self.ues:
                ue["load"] = 1.0

    def _sinr_for_ue(self, ue: dict) -> float:
        serving = ue["serving_cell"]
        signal_cell = self.cells[serving]
        beam_gain = 3.0 if signal_cell["beamforming"] else 0.0
        pathloss = _pathloss_db(ue["distance_m"])
        signal_dbm = signal_cell["power_dbm"] + beam_gain - pathloss
        interf_linear = 0.0
        for c in self.cells:
            if c["cell_id"] == serving:
                continue
            interference_pathloss = _pathloss_db(ue["distance_m"] * self.rng.uniform(0.85, 1.25))
            coord_gain = -2.5 if (c["beamforming"] and signal_cell["beamforming"]) else 0.0
            interf_dbm = c["power_dbm"] - interference_pathloss + coord_gain
            interf_linear += 10 ** (interf_dbm / 10.0)
        noise_linear = 10 ** (NOISE_DBM / 10.0)
        signal_linear = 10 ** (signal_dbm / 10.0)
        sinr_linear = signal_linear / (interf_linear + noise_linear)
        return 10.0 * math.log10(max(sinr_linear, 1e-12))

    def _maybe_handover(self, ue: dict) -> tuple[bool, bool]:
        current = ue["serving_cell"]
        current_margin = self.cells[current]["ho_margin_db"]
        best_score = self._sinr_for_ue(ue)
        best_cell = current
        for c in range(NUM_CELLS):
            if c == current:
                continue
            tmp = ue["serving_cell"]
            ue["serving_cell"] = c
            cand_score = self._sinr_for_ue(ue)
            ue["serving_cell"] = tmp
            if cand_score > best_score + current_margin:
                best_score, best_cell = cand_score, c

        ho_failure = ping_pong = False
        if best_cell != current:
            if self.rng.random() < 0.08:
                ho_failure = True
            else:
                if self.step_id - ue["last_handover_step"] <= 3:
                    ping_pong = True
                ue["serving_cell"] = best_cell
                ue["last_handover_step"] = self.step_id
        return ho_failure, ping_pong

    def step(self, action: dict) -> tuple[Observation, StepInfo]:
        for c in range(NUM_CELLS):
            self.cells[c]["power_dbm"] = float(action.get(f"power_dbm_cell{c}", self.cells[c]["power_dbm"]))
            self.cells[c]["beamforming"] = bool(action.get(f"beamforming_cell{c}", self.cells[c]["beamforming"]))
            self.cells[c]["ho_margin_db"] = float(action.get(f"ho_margin_cell{c}", self.cells[c]["ho_margin_db"]))

        self._update_traffic()
        throughputs, ho_failures, ping_pongs = [], [], []
        for ue in self.ues:
            sinr_db = self._sinr_for_ue(ue)
            se = max(0.2, 0.6 * math.log2(1.0 + 10 ** (sinr_db / 10.0)))
            tput = se * 20.0 / NUM_CELLS * ue["load"]
            throughputs.append(tput)
            ho_fail, ping_pong = self._maybe_handover(ue)
            ho_failures.append(ho_fail); ping_pongs.append(ping_pong)

        thr_arr = np.array(throughputs)
        fairness = jains_fairness(thr_arr)
        p5 = float(np.percentile(thr_arr, 5))
        avg_interf = 0.0  # aggregate interference proxy; detailed per-UE interference already folded into SINR
        ho_failure_rate = float(np.mean(ho_failures))
        ping_pong_ratio = float(np.mean(ping_pongs))

        reward = compute_reward_multi_cell(
            avg_throughput=float(thr_arr.mean()), fairness=fairness, p5_throughput=p5,
            ho_failure_rate=ho_failure_rate, ping_pong_ratio=ping_pong_ratio, avg_interference=avg_interf,
        )

        self.step_id += 1
        self._advance()
        obs = Observation(step=self._step_count,
                           extra={"avg_throughput": float(thr_arr.mean()), "fairness": fairness,
                                  "p5_throughput": p5, "ho_failure_rate": ho_failure_rate,
                                  "ping_pong_ratio": ping_pong_ratio, "loads": [u["load"] for u in self.ues]})
        info = StepInfo(reward=reward, terminated=(self._step_count >= 100))
        return obs, info

    def render(self) -> str:
        return f"step={self._step_count} cells={[c['power_dbm'] for c in self.cells]}"


# ---------------------------------------------------------------------------
# The four ICIC policies
# ---------------------------------------------------------------------------

def _cell_action(power, beamforming, ho_margin) -> dict:
    a = {}
    for c in range(NUM_CELLS):
        a[f"power_dbm_cell{c}"] = power[c] if isinstance(power, (list, np.ndarray)) else power
        a[f"beamforming_cell{c}"] = beamforming[c] if isinstance(beamforming, (list, np.ndarray)) else beamforming
        a[f"ho_margin_cell{c}"] = ho_margin[c] if isinstance(ho_margin, (list, np.ndarray)) else ho_margin
    return a


class StaticICICPolicy:
    name = "Static-ICIC"

    def on_episode_start(self, ep): pass
    def on_episode_end(self, summary): pass

    def select_action(self, loads: np.ndarray) -> dict:
        return _cell_action(43.0, False, 3.0)


class DecentralizedHeuristicPolicy:
    name = "Decentralized"

    def on_episode_start(self, ep): pass
    def on_episode_end(self, summary): pass

    def select_action(self, loads: np.ndarray) -> dict:
        power = [41.0 if loads[c] < 0.6 else 45.0 for c in range(NUM_CELLS)]
        beam = [1 if loads[c] > 0.5 else 0 for c in range(NUM_CELLS)]
        return _cell_action(power, beam, 3.0)


class ReActICICPolicy:
    name = "ReAct-ICIC"

    def on_episode_start(self, ep): pass
    def on_episode_end(self, summary): pass

    def select_action(self, loads: np.ndarray) -> dict:
        imbalance = loads - loads.mean()
        power = list(np.clip(43.0 + 4.0 * imbalance, 38.0, 46.0))
        margin = list(np.clip(3.0 - 1.2 * imbalance, 1.5, 5.0))
        beam = [1 if abs(imbalance[c]) > 0.1 else 0 for c in range(NUM_CELLS)]
        return _cell_action(power, beam, margin)


class ReflexionICICPolicy(BaseReflexionPolicy):
    name = "Reflexion-ICIC"
    kpi_thresholds = {"ping_pong_ratio": 0.08, "fairness": 0.72}

    def __init__(self):
        super().__init__(scenario_key="scenario3")
        self.power_bias = 0.0
        self.margin_bias = 0.0

    def _lesson_from_kpi(self, kpi: str, value: float) -> str:
        if kpi == "ping_pong_ratio" and value > self.kpi_thresholds["ping_pong_ratio"]:
            return "pingpong too high: increase handover margin and avoid aggressive power bias"
        if kpi == "fairness" and value < self.kpi_thresholds["fairness"]:
            return "fairness low: shift power toward overloaded cells"
        return ""

    def build_prompt(self, obs, history) -> str:
        return "n/a"  # this policy is numeric/rule-based, not LLM-prompted (matches original design)

    def on_episode_end(self, summary: dict) -> None:
        super().on_episode_end(summary)
        recent = self._lessons[-3:]
        self.power_bias = -1.0 if any("pingpong" in s.lower() for s in recent) else 0.0
        self.margin_bias = 0.8 if any("handover" in s.lower() for s in recent) else 0.0

    def select_action(self, loads: np.ndarray) -> dict:
        imbalance = loads - loads.mean()
        power = list(np.clip(43.0 + (4.0 + self.power_bias) * imbalance, 38.0, 46.0))
        margin = list(np.clip(3.0 - 1.2 * imbalance + self.margin_bias, 1.5, 5.0))
        beam = [1 if abs(imbalance[c]) > 0.1 else 0 for c in range(NUM_CELLS)]
        return _cell_action(power, beam, margin)


POLICY_REGISTRY = {
    "static": StaticICICPolicy, "decentralized": DecentralizedHeuristicPolicy,
    "react": ReActICICPolicy, "reflexion": ReflexionICICPolicy,
}


def run_episode(env: MultiCellTelcoEnv, policy, ep: int, num_steps: int, cfg: dict, logger: CostLogger) -> dict:
    env.traffic_mode = cfg["traffic_mode"]
    obs = env.reset()
    rows = []
    for st in range(num_steps):
        loads = np.array([ue["load"] for ue in env.ues])
        with logger.step(episode=ep, step=st):
            action = policy.select_action(loads)
            obs, info = env.step(action)
            logger.record(reward=info.reward, bler=0.0, throughput=obs.extra["avg_throughput"])
        rows.append(obs.extra)
        if info.terminated:
            break
    df = pd.DataFrame(rows)
    summary = {
        "episode": ep, "traffic_mode": cfg["traffic_mode"],
        "avg_reward": float(info.reward), "avg_throughput_mbps": float(df["avg_throughput"].mean()),
        "avg_fairness": float(df["fairness"].mean()), "avg_p5_throughput_mbps": float(df["p5_throughput"].mean()),
        "avg_ho_failure_rate": float(df["ho_failure_rate"].mean()), "avg_ping_pong_ratio": float(df["ping_pong_ratio"].mean()),
    }
    return summary


def run_scenario3(episodes: int = 100, num_steps: int = 100, out_dir: str = "results") -> pd.DataFrame:
    scenarios = [{"traffic_mode": "static"}, {"traffic_mode": "dynamic"}]
    all_rows = []
    for pkey, policy_cls in POLICY_REGISTRY.items():
        policy = policy_cls()
        env = MultiCellTelcoEnv(config={}, rng_seed=CFG["global"]["seed"])
        logger = CostLogger(experiment="exp3")
        print(f"[S3] Running {policy.name} for {episodes} episodes ...")
        for ep in range(episodes):
            cfg = scenarios[0] if ep < episodes // 2 else scenarios[1]
            policy.on_episode_start(ep)
            summary = run_episode(env, policy, ep, num_steps, cfg, logger)
            policy.on_episode_end(summary)
            all_rows.append({**summary, "policy": policy.name})
        logger.save()

    df = pd.DataFrame(all_rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp3_multicell_results.csv", index=False)
    print(f"\n[S3] Saved {len(df)} rows to {out / 'exp3_multicell_results.csv'}")
    print(df.groupby(["policy", "traffic_mode"])[["avg_reward", "avg_fairness", "avg_p5_throughput_mbps"]].mean())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    run_scenario3(episodes=args.episodes, num_steps=args.steps)
