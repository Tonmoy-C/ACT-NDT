"""
scenario4_mobility_blockage.py — Two sequential shifts: pedestrian->vehicular
mobility, then TDL-D + blockage. Static / DQN / ReAct / Reflexion policies.

Reconstructed from the docx walkthrough (NonStationaryTelcoEnv, the four
policies, runner.py's three-phase schedule + recovery-time estimator).

Fixes applied during reconstruction:
  - DQNStylePolicy now wraps dqn_baseline.DQNAgent (fix #4 - real replay
    buffer + target network + epsilon-greedy), not a static 3-branch lookup
    table. It's pretrained for scenario4.dqn.pretrain_episodes before the
    comparison window, per SETUP_MACOS.md's Phase 7.
  - ReflexionPolicy subclasses base_reflexion_policy.BaseReflexionPolicy
    (fix #13) with its own scenario4 memory_window.
  - Channel penalty via phy_models.tdl_channel_penalty_db (fix #9), same
    canonical function Scenario 2/5 use - not a separately-tuned inline copy.
  - Cost logging via cost_logger.py (fix #3).

Run:
    python scenario4_mobility_blockage.py --episodes 150
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config_loader import CFG
from base_env import BaseEnv, Observation, StepInfo
from base_reflexion_policy import BaseReflexionPolicy
from phy_models import compute_bler, compute_effective_snr, tdl_channel_penalty_db, spectral_efficiency
from cost_logger import CostLogger

_S4 = CFG["scenario4"]


def scenario_for_episode(ep: int) -> dict:
    """Three-phase schedule: 0-49 pedestrian, 50-99 vehicular shift,
    100-149 vehicular + TDL-D + blockage (stacks, doesn't reset to pedestrian)."""
    if ep < 50:
        return {"mobility_kmh": 3.0, "channel_model": "TDL-C", "blockage_prob": 0.0, "phase": "pedestrian"}
    if ep < 100:
        return {"mobility_kmh": 60.0, "channel_model": "TDL-C", "blockage_prob": 0.0, "phase": "vehicular_shift"}
    return {"mobility_kmh": 60.0, "channel_model": "TDL-D", "blockage_prob": 0.05,
            "blockage_loss_db": 20.0, "phase": "blockage_shift"}


class NonStationaryTelcoEnv(BaseEnv):
    """Single-link twin whose mobility/channel/blockage parameters are set
    per-episode by scenario_for_episode(), letting one environment class
    represent three very different operating regimes."""

    def __init__(self, config: dict, rng_seed: int = 42):
        super().__init__(config, rng_seed)
        self.base_snr = CFG["phy"]["base_snr_db"]
        self.bler_target = CFG["phy"]["bler_target"]
        self.phase_cfg = scenario_for_episode(0)
        self._last_bler = 0.0

    def configure(self, phase_cfg: dict) -> None:
        self.phase_cfg = phase_cfg

    def reset(self) -> Observation:
        self._new_episode()
        return Observation(snr_db=self.base_snr, bler=0.0, throughput_mbps=0.0,
                            channel_model=self.phase_cfg["channel_model"],
                            mobility_kmh=self.phase_cfg["mobility_kmh"], step=0)

    def step(self, action: dict) -> tuple[Observation, StepInfo]:
        action = self._clip_action(action)
        penalty_db, blockage_hit = tdl_channel_penalty_db(
            channel_model=self.phase_cfg["channel_model"], mobility_kmh=self.phase_cfg["mobility_kmh"],
            blockage_prob=self.phase_cfg.get("blockage_prob", 0.0),
            blockage_loss_db=self.phase_cfg.get("blockage_loss_db", 20.0), rng=self.rng)
        noisy_snr = self.base_snr + self.rng.normal(0, 1.5)
        eff_snr = compute_effective_snr(noisy_snr, power_db=action["power_db"], pmi=action["pmi"],
                                         channel_penalty_db=penalty_db)
        bler = compute_bler(eff_snr, action["mcs"], profile="mobility")
        se = spectral_efficiency(action["mcs"])
        tput = se * 20.0 * (1.0 - bler)
        reward = tput - 18.0 * max(0.0, bler - self.bler_target)
        if blockage_hit:
            reward -= 0.8

        self._last_bler = bler
        self._advance()
        obs = Observation(snr_db=noisy_snr, bler=bler, throughput_mbps=tput,
                           channel_model=self.phase_cfg["channel_model"],
                           mobility_kmh=self.phase_cfg["mobility_kmh"], step=self._step_count,
                           extra={"blockage_hit": blockage_hit})
        info = StepInfo(reward=reward, terminated=(self._step_count >= 100))
        return obs, info

    def render(self) -> str:
        return f"phase={self.phase_cfg['phase']} BLER={self._last_bler:.3f}"


# ---------------------------------------------------------------------------
# The four policies
# ---------------------------------------------------------------------------

class StaticPolicy:
    name = "Static"
    def on_episode_start(self, ep): pass
    def on_episode_end(self, summary): pass
    def select_action(self, obs: Observation) -> dict:
        return {"mcs": 15, "pmi": 0, "power_db": 20.0}


class DQNStylePolicy:
    """Wraps the real DQNAgent (fix #4). Pretrains once, then runs greedy."""
    name = "DQN-Replay"

    def __init__(self, env: NonStationaryTelcoEnv):
        from dqn_baseline import DQNAgent
        self.agent = DQNAgent(seed=CFG["global"]["seed"])
        n_pretrain = _S4["dqn"]["pretrain_episodes"]
        print(f"[S4] Pretraining DQN for {n_pretrain} episodes ...")
        env.configure(scenario_for_episode(0))  # pretrain under pedestrian conditions
        rewards = self.agent.train(env, episodes=n_pretrain)
        self._report_convergence(rewards)

    def _report_convergence(self, rewards: list[float]) -> None:
        """Surfaces whether pretraining actually converged, rather than
        silently discarding the reward curve - a real run once produced a
        DQN baseline that scored *worse* than the Static policy across every
        phase, and this would have caught it before the 3-hour comparison
        run even started."""
        n = len(rewards)
        if n < 10:
            print(f"[S4] DQN pretrain reward curve too short ({n} episodes) to assess convergence.")
            return
        first_avg = sum(rewards[: max(1, n // 5)]) / max(1, n // 5)
        last_avg = sum(rewards[-max(1, n // 5):]) / max(1, n // 5)
        print(f"[S4] DQN pretrain reward: first-20% avg={first_avg:.2f} -> last-20% avg={last_avg:.2f}")
        if last_avg <= first_avg:
            print("[S4] WARNING: DQN reward did not improve over pretraining - "
                  "it is not a fair 'trained' baseline yet. Increase "
                  "scenario4.dqn.pretrain_episodes in config.yaml and re-run "
                  "before trusting this comparison.")
        print("[S4] DQN pretraining done.")

    def on_episode_start(self, ep): pass
    def on_episode_end(self, summary): pass

    def select_action(self, obs: Observation) -> dict:
        action, _ = self.agent.select_action(obs, greedy=True)
        return action


class ReActPolicy:
    """Rule-based reasoning that explicitly conditions on mobility state
    (kept as the original heuristic reference point; see llm-driven variants
    in Scenario 2 for the actual LLM-backed ReAct implementation)."""
    name = "ReAct"
    def on_episode_start(self, ep): pass
    def on_episode_end(self, summary): pass

    def select_action(self, obs: Observation) -> dict:
        bler, mobility = obs.bler, obs.mobility_kmh
        if mobility >= 60 and bler > 0.10:
            return {"mcs": 8, "pmi": 0, "power_db": 22.0}
        if bler > 0.12:
            return {"mcs": 10, "pmi": 0, "power_db": 21.0}
        if bler < 0.06:
            return {"mcs": 14, "pmi": 0, "power_db": 20.0}
        return {"mcs": 12, "pmi": 0, "power_db": 20.0}


class ReflexionPolicy(BaseReflexionPolicy):
    name = "Reflexion"
    kpi_thresholds = {"bler": 0.13, "throughput": 9.0, "blockage_rate": 0.02}

    def __init__(self):
        super().__init__(scenario_key="scenario4")
        self.cautious_mode = False
        self.power_boost = 0.0

    def _lesson_from_kpi(self, kpi: str, value: float) -> str:
        if kpi == "bler" and value > self.kpi_thresholds["bler"]:
            return "need more robust MCS under shift"
        if kpi == "throughput_mobility" and value:  # sentinel handled in on_episode_end
            return "high mobility requires robust scheduling and conservative MCS"
        if kpi == "blockage_rate" and value > self.kpi_thresholds["blockage_rate"]:
            return "blockage events require extra power margin and lower MCS"
        return ""

    def build_prompt(self, obs, history) -> str:
        return "n/a"  # numeric/rule-based reflection, matches original design

    def on_episode_end(self, summary: dict) -> None:
        # extra combined condition the base class's per-KPI loop can't express directly
        if summary.get("avg_throughput", 99) < 9.0 and summary.get("mobility_kmh", 0) >= 60:
            lesson = "high mobility requires robust scheduling and conservative MCS"
            if lesson not in self._lessons:
                self._lessons.append(lesson)
        super().on_episode_end(summary)
        recent = self._lessons[-3:]
        self.cautious_mode = any("robust" in s.lower() for s in recent)
        self.power_boost = 1.0 if any("blockage" in s.lower() for s in recent) else 0.0

    def select_action(self, obs: Observation) -> dict:
        bler = obs.bler
        blockage_prob = 0.05 if obs.mobility_kmh >= 60 and obs.channel_model == "TDL-D" else 0.0
        if blockage_prob > 0 and (self.cautious_mode or bler > 0.10):
            return {"mcs": 7, "pmi": 0, "power_db": 22.0 + self.power_boost}
        if bler > 0.12:
            return {"mcs": 9, "pmi": 0, "power_db": 21.0}
        return {"mcs": 13, "pmi": 0, "power_db": 20.0}


# ---------------------------------------------------------------------------
# Recovery-time estimator + orchestration
# ---------------------------------------------------------------------------

def estimate_recovery_time(ep_df: pd.DataFrame, shift_episode: int, metric_col: str = "avg_reward",
                            window: int = 5) -> int:
    pre = ep_df[ep_df["episode"] < shift_episode].tail(window)[metric_col].mean()
    target = 0.95 * pre
    post = ep_df[ep_df["episode"] >= shift_episode]
    for _, row in post.iterrows():
        trailing = post[post["episode"] <= row["episode"]].tail(window)[metric_col].mean()
        if trailing >= target:
            return int(row["episode"] - shift_episode)
    return -1


def run_episode(env: NonStationaryTelcoEnv, policy, ep: int, num_steps: int, logger: CostLogger) -> dict:
    cfg = scenario_for_episode(ep)
    env.configure(cfg)
    obs = env.reset()
    rewards, blers, tps, blockages = [], [], [], []
    for st in range(num_steps):
        with logger.step(episode=ep, step=st):
            action = policy.select_action(obs)
            obs, info = env.step(action)
            logger.record(reward=info.reward, bler=obs.bler, throughput=obs.throughput_mbps)
        rewards.append(info.reward); blers.append(obs.bler); tps.append(obs.throughput_mbps)
        blockages.append(obs.extra.get("blockage_hit", 0))
        if info.terminated:
            break
    return {"episode": ep, "phase": cfg["phase"], "mobility_kmh": cfg["mobility_kmh"],
            "avg_reward": float(np.mean(rewards)), "avg_bler": float(np.mean(blers)),
            "avg_throughput": float(np.mean(tps)), "blockage_rate": float(np.mean(blockages))}


def run_scenario4(episodes: int = 150, num_steps: int = 100, out_dir: str = "results") -> pd.DataFrame:
    all_rows = []
    for name in ["Static", "DQN-Replay", "ReAct", "Reflexion"]:
        env = NonStationaryTelcoEnv(config={}, rng_seed=CFG["global"]["seed"])
        policy = ({"Static": lambda: StaticPolicy(), "DQN-Replay": lambda: DQNStylePolicy(env),
                   "ReAct": lambda: ReActPolicy(), "Reflexion": lambda: ReflexionPolicy()})[name]()
        logger = CostLogger(experiment="exp4")
        print(f"[S4] Running {policy.name} for {episodes} episodes ...")
        ep_rows = []
        for ep in range(episodes):
            policy.on_episode_start(ep)
            summary = run_episode(env, policy, ep, num_steps, logger)
            policy.on_episode_end(summary)
            ep_rows.append(summary)
            all_rows.append({**summary, "policy": policy.name})
        logger.save()
        ep_df = pd.DataFrame(ep_rows)
        r1 = estimate_recovery_time(ep_df, 50)
        r2 = estimate_recovery_time(ep_df, 100)
        print(f"      recovery after shift1: {r1} eps | after shift2: {r2} eps")

    df = pd.DataFrame(all_rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp4_shift_results.csv", index=False)
    print(f"\n[S4] Saved {len(df)} rows to {out / 'exp4_shift_results.csv'}")
    print(df.groupby(["policy", "phase"])["avg_reward"].mean())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    run_scenario4(episodes=args.episodes, num_steps=args.steps)
