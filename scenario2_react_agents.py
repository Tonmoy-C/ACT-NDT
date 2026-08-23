"""
scenario2_react_agents.py — Channel-shift robustness: 4 LLM agent architectures.

Reconstructed from the docx walkthrough (env_client.py's TelcoTwinEnv,
policy_variants.py's Act-only/ReAct/ReAct+Memory/Reflexion classes, main.py's
orchestration). See README.md "Provenance" for what's reconstructed vs fixed.

Fixes applied during reconstruction (vs. the original code the docx describes):
  - Environment now uses phy_models.py (shared, consistent BLER/shift model)
    instead of its own inline sigmoid + a flat +0.02 BLER shift penalty.
  - ReflexionPolicy now subclasses base_reflexion_policy.BaseReflexionPolicy
    (fix #13) instead of duplicating the lesson-generation logic inline.
  - All four policies call the real llm_client.LLMClient (fix #1) instead of
    LLMClient._simple_heuristic_response(); action parsing is the structured
    JSON parser (fix #10) instead of the regex `stepaction(mcs=...)` parser.
  - Cost logging via cost_logger.py (fix #3) on every step.

Run:
    python scenario2_react_agents.py --episodes 200 --shift-at 100
    python scenario2_react_agents.py --episodes 20 --shift-at 10 --variants react,reflexion
"""
from __future__ import annotations
import argparse
import itertools
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config_loader import CFG
from base_env import BaseEnv, Observation, StepInfo
from base_reflexion_policy import BaseReflexionPolicy
from llm_client import LLMClient
from phy_models import (
    compute_bler, compute_effective_snr, compute_reward_single_link,
    spectral_efficiency, tdl_channel_penalty_db,
)
from cost_logger import CostLogger

_S2 = CFG["scenario2"]


# ---------------------------------------------------------------------------
# Part 1: TelcoTwinEnv — the local digital-twin surrogate (env_client.py)
# ---------------------------------------------------------------------------

class TelcoTwinEnv(BaseEnv):
    """4x4 MIMO single-link twin with a swappable channel model, used to
    study TDL-C -> TDL-D shift robustness. Local analytic stand-in for the
    Sionna twin (same role as Scenario 1's --api-url pathway, but always
    local here since S2's focus is agent architecture, not PHY validation)."""

    def __init__(self, config: dict, rng_seed: int = 42):
        super().__init__(config, rng_seed)
        self.num_ues = config.get("num_ues", 4)
        self.base_snr = CFG["phy"]["base_snr_db"]
        self.bler_target = CFG["phy"]["bler_target"]
        self.channel_model = "TDL-C"
        self.mobility_kmh = 0.0
        self._last_bler = 0.0
        self._last_tp = 0.0

    def set_channel_model(self, model: str) -> None:
        """Called by the runner to trigger the TDL-C -> TDL-D shift."""
        self.channel_model = model

    def reset(self) -> Observation:
        self._new_episode()
        self._last_bler, self._last_tp = 0.0, 0.0
        return Observation(snr_db=self.base_snr, bler=0.0, throughput_mbps=0.0,
                            channel_model=self.channel_model, step=0)

    def step(self, action: dict) -> tuple[Observation, StepInfo]:
        action = self._clip_action(action)
        penalty_db, _ = tdl_channel_penalty_db(channel_model=self.channel_model,
                                                mobility_kmh=self.mobility_kmh, rng=self.rng)
        noisy_snr = self.base_snr + self.rng.normal(0, 1.5)
        eff_snr = compute_effective_snr(noisy_snr, power_db=action["power_db"], pmi=action["pmi"],
                                         channel_penalty_db=penalty_db)
        bler = compute_bler(eff_snr, action["mcs"], profile="default")
        se = spectral_efficiency(action["mcs"])
        tput = se * 20.0 * (1.0 - bler)
        reward = compute_reward_single_link(tput, bler, bler_target=self.bler_target)

        self._last_bler, self._last_tp = bler, tput
        self._advance()
        obs = Observation(snr_db=noisy_snr, bler=bler, throughput_mbps=tput,
                           channel_model=self.channel_model, step=self._step_count)
        info = StepInfo(reward=reward, terminated=(self._step_count >= 100))
        return obs, info

    def render(self) -> str:
        return f"BLER={self._last_bler:.3f}  Throughput={self._last_tp:.1f} Mbps  channel={self.channel_model}"


# ---------------------------------------------------------------------------
# Part 2: choose_adaptive_action — non-LLM heuristic baseline (comparison only)
# ---------------------------------------------------------------------------

_MCS_GRID = [8, 10, 12, 14, 16, 18]
_PMI_GRID = [0, 1, 2]
_POWER_GRID = [18.0, 20.0, 22.0]


def choose_adaptive_action(state: dict) -> dict:
    """Brute-force one-step-lookahead grid search (54 candidates), scored by a
    linear prediction from the last observed state. Non-AI reference point,
    not one of the four agent variants."""
    best_score, best_action = -1e9, {"mcs": 14, "pmi": 0, "power_db": 20.0}
    for mcs, pmi, power_db in itertools.product(_MCS_GRID, _PMI_GRID, _POWER_GRID):
        predicted_bler = state["last_bler"] + 0.015 * (mcs - 12) - 0.01 * (power_db - 20.0)
        predicted_tp = state["last_throughput"] + 0.35 * (mcs - 12) + 0.10 * (power_db - 20.0)
        score = predicted_tp - 25.0 * max(0.0, predicted_bler - 0.10)
        if score > best_score:
            best_score, best_action = score, {"mcs": mcs, "pmi": pmi, "power_db": power_db}
    return best_action


# ---------------------------------------------------------------------------
# Part 3: The four LLM policy classes
# ---------------------------------------------------------------------------

class BasePolicy:
    name = "base"

    def __init__(self):
        self.client = LLMClient()

    def on_episode_start(self, episode_num: int) -> None:
        pass

    def on_episode_end(self, summary: dict) -> None:
        pass

    def _state_block(self, obs: Observation) -> str:
        return (f"Current state: BLER={obs.bler:.3f}, Throughput={obs.throughput_mbps:.1f} Mbps, "
                f"Channel={obs.channel_model}\nChoose the next action (mcs 0-28, pmi 0-15, power_db 10-30):")

    def select_action(self, obs: Observation) -> tuple[dict, float, int]:
        raise NotImplementedError


class ActOnlyPolicy(BasePolicy):
    """No memory, no reasoning trace — direct observation -> action mapping."""
    name = "ActOnly"

    def select_action(self, obs: Observation):
        prompt = self._state_block(obs)
        resp = self.client.generate(prompt)
        return resp.action, resp.wall_s, resp.tokens_used, resp.provider


class ReActPolicy(BasePolicy):
    """Adds an explicit reasoning trace within an episode, reset every episode."""
    name = "ReAct"

    def on_episode_start(self, episode_num: int) -> None:
        self.thought_log: list[str] = []  # reset each episode - no cross-episode memory

    def select_action(self, obs: Observation):
        prompt = (self._state_block(obs) +
                  "\nFirst reason step by step about BLER/throughput tradeoffs, then choose.")
        resp = self.client.generate(prompt)
        self.thought_log.append(resp.text[:200])
        return resp.action, resp.wall_s, resp.tokens_used, resp.provider


class ReActMemoryPolicy(BasePolicy):
    """Adds a rolling window of the last K episode summaries to every prompt."""
    name = "ReActMemory"

    def __init__(self, k: Optional[int] = None):
        super().__init__()
        self.k = k or _S2["memory_window"]
        self.episode_summaries: list[dict] = []

    def _format_memory(self) -> str:
        if not self.episode_summaries:
            return ""
        recent = self.episode_summaries[-self.k:]
        lines = [f"  ep {s['episode']}: avg_bler={s['avg_bler']:.3f}, avg_tp={s['avg_throughput']:.1f}"
                  for s in recent]
        return "Previous episodes:\n" + "\n".join(lines) + "\n"

    def select_action(self, obs: Observation):
        prompt = self._format_memory() + self._state_block(obs)
        resp = self.client.generate(prompt)
        return resp.action, resp.wall_s, resp.tokens_used, resp.provider

    def on_episode_end(self, summary: dict) -> None:
        self.episode_summaries.append(summary)


class ReflexionPolicy(BaseReflexionPolicy, BasePolicy):
    """Most complete variant — calls the LLM again after each episode to
    generate a verbal lesson, per base_reflexion_policy.py (fix #13)."""
    name = "Reflexion"
    kpi_thresholds = {"bler": 0.10, "throughput": 9.0}

    def __init__(self):
        BaseReflexionPolicy.__init__(self, scenario_key="scenario2")

    def _lesson_from_kpi(self, kpi: str, value: float) -> str:
        if kpi == "bler" and value > self.kpi_thresholds["bler"]:
            return f"BLER {value:.3f} exceeded target; prefer lower MCS or more power."
        if kpi == "throughput" and value < self.kpi_thresholds["throughput"]:
            return f"Throughput {value:.1f} Mbps low; try higher MCS if BLER allows."
        return ""

    def build_prompt(self, obs: Observation, history: list) -> str:
        return self._state_block(obs)

    def select_action(self, obs: Observation):
        action, resp = self.act(obs)
        return action, resp.wall_s, resp.tokens_used, resp.provider


POLICY_REGISTRY = {
    "actonly": ActOnlyPolicy, "react": ReActPolicy,
    "reactmemory": ReActMemoryPolicy, "reflexion": ReflexionPolicy,
}


# ---------------------------------------------------------------------------
# Part 4: Orchestration — run all variants over N episodes with a channel shift
# ---------------------------------------------------------------------------

def run_episode(env: TelcoTwinEnv, policy: BasePolicy, ep: int, num_steps: int, logger: CostLogger,
                 provider_counts: dict) -> dict:
    obs = env.reset()
    state = {"last_bler": 0.0, "last_throughput": 0.0}
    rewards, blers, tps = [], [], []
    for st in range(num_steps):
        with logger.step(episode=ep, step=st):
            action, wall_s, tokens, provider = policy.select_action(obs)
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            obs, info = env.step(action)
            logger.record(reward=info.reward, bler=obs.bler, throughput=obs.throughput_mbps, tokens=tokens)
        rewards.append(info.reward); blers.append(obs.bler); tps.append(obs.throughput_mbps)
        state = {"last_bler": obs.bler, "last_throughput": obs.throughput_mbps}
        if info.terminated:
            break
    return {"episode": ep, "avg_reward": float(np.mean(rewards)), "avg_bler": float(np.mean(blers)),
            "avg_throughput": float(np.mean(tps))}


def run_scenario2(episodes: int = 200, shift_at: int = 100, variants: Optional[list[str]] = None,
                   num_steps: int = 100, out_dir: str = "results") -> pd.DataFrame:
    variants = variants or list(POLICY_REGISTRY.keys())
    all_rows = []

    for vkey in variants:
        policy_cls = POLICY_REGISTRY[vkey]
        policy = policy_cls()
        env = TelcoTwinEnv(config={"num_ues": 4}, rng_seed=CFG["global"]["seed"])
        logger = CostLogger(experiment="exp2", provider=CFG["llm"]["provider"])
        print(f"[S2] Running {policy.name} for {episodes} episodes (shift at {shift_at}) ...")
        provider_counts: dict = {}

        for ep in range(episodes):
            if ep == shift_at:
                env.set_channel_model("TDL-D")
            policy.on_episode_start(ep)
            summary = run_episode(env, policy, ep, num_steps, logger, provider_counts)
            policy.on_episode_end(summary)
            all_rows.append({**summary, "policy": policy.name})

        logger.save()
        total = sum(provider_counts.values()) or 1
        fallback_pct = 100.0 * provider_counts.get("heuristic", 0) / total
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(provider_counts.items()))
        print(f"[S2] {policy.name} provider breakdown: {breakdown}  "
              f"({fallback_pct:.1f}% fell back to heuristic default)")
        if fallback_pct > 10:
            print(f"[S2] WARNING: {policy.name} used the heuristic fallback on "
                  f"{fallback_pct:.1f}% of steps - a meaningful fraction of its "
                  f"'decisions' were actually the same fixed safe action, not "
                  f"real model output. Compare this rate across policies before "
                  f"treating differences between them as an architecture effect "
                  f"rather than an LLM output-parsing reliability difference.")

    df = pd.DataFrame(all_rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp2_full_results.csv", index=False)
    print(f"\n[S2] Saved {len(df)} rows to {out / 'exp2_full_results.csv'}")
    print(df.groupby("policy")[["avg_reward", "avg_bler", "avg_throughput"]].mean())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--shift-at", type=int, default=100)
    p.add_argument("--variants", type=str, default="all",
                   help="comma-separated: actonly,react,reactmemory,reflexion or 'all'")
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    variants = None if args.variants == "all" else args.variants.split(",")
    run_scenario2(episodes=args.episodes, shift_at=args.shift_at, variants=variants, num_steps=args.steps)
