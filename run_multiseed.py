"""
run_multiseed.py — statistical rigor across seeds.

Every scenario so far has run on a single seed (42). That's fine for
debugging, but a paper comparing agent architectures needs to show the
comparison holds up across randomness, not just on one lucky/unlucky draw.
This script re-runs a scenario across N seeds, reports mean +/- std per
policy (not just per-episode stats within one run), and a Welch's t-test
between each pair of policies so you can say "X beats Y" with a p-value
attached instead of a single point estimate.

Included by default: Scenario 3, 4, 5 - all fast (seconds to ~2 min each),
so 10 seeds costs minutes, not hours.

Scenario 2 is NOT run by default - it's the one with real LLM calls, and
10 seeds at full episode count could mean 10x today's multi-hour runtime.
Pass --include-scenario2 if you want it anyway; strongly consider pairing
that with a reduced --episodes value.

Usage:
    python run_multiseed.py                          # S3, S4, S5, 10 seeds
    python run_multiseed.py --seeds 5 --scenarios 3,5
    python run_multiseed.py --include-scenario2 --s2-episodes 20 --seeds 3
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config_loader
from config_loader import CFG


def _run_seeds(run_fn, seeds: list[int], metric_col: str, group_col: str = "policy",
                **run_kwargs) -> pd.DataFrame:
    """Call run_fn(**run_kwargs) once per seed with CFG['global']['seed']
    patched, and return one row per (seed, policy) with the seed-level mean
    of metric_col. This is the number that should vary run-to-run only due
    to genuine randomness, not because different seeds got different amounts
    of episodes/steps."""
    rows = []
    orig_seed = CFG["global"]["seed"]
    for seed in seeds:
        CFG["global"]["seed"] = seed
        df = run_fn(**run_kwargs)
        for policy, group in df.groupby(group_col):
            rows.append({"seed": seed, group_col: policy, metric_col: group[metric_col].mean()})
    CFG["global"]["seed"] = orig_seed
    return pd.DataFrame(rows)


def _pairwise_ttests(per_seed_df: pd.DataFrame, metric_col: str, group_col: str = "policy") -> pd.DataFrame:
    """Welch's t-test (unequal variance, the safer default) between every
    pair of policies, using each policy's per-seed mean as one sample."""
    rows = []
    policies = sorted(per_seed_df[group_col].unique())
    for a, b in itertools.combinations(policies, 2):
        xa = per_seed_df[per_seed_df[group_col] == a][metric_col].values
        xb = per_seed_df[per_seed_df[group_col] == b][metric_col].values
        if len(xa) < 2 or len(xb) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(xa, xb, equal_var=False)
        rows.append({
            "policy_a": a, "policy_b": b,
            "mean_a": xa.mean(), "mean_b": xb.mean(),
            "t_stat": t_stat, "p_value": p_val,
            "significant_at_0.05": bool(p_val < 0.05),
        })
    return pd.DataFrame(rows)


def _summarize(name: str, per_seed_df: pd.DataFrame, metric_col: str, out_dir: Path) -> None:
    summary = per_seed_df.groupby("policy")[metric_col].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["policy", f"{metric_col}_mean", f"{metric_col}_std", "n_seeds"]
    ttests = _pairwise_ttests(per_seed_df, metric_col)

    per_seed_df.to_csv(out_dir / f"{name}_per_seed.csv", index=False)
    summary.to_csv(out_dir / f"{name}_summary.csv", index=False)
    ttests.to_csv(out_dir / f"{name}_pairwise_ttests.csv", index=False)

    print(f"\n[{name}] Per-policy {metric_col}, mean +/- std across {per_seed_df['seed'].nunique()} seeds:")
    print(summary.to_string(index=False))
    if not ttests.empty:
        print(f"\n[{name}] Pairwise significance (Welch's t-test, alpha=0.05):")
        print(ttests[["policy_a", "policy_b", "mean_a", "mean_b", "p_value", "significant_at_0.05"]]
              .to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--scenarios", type=str, default="3,4,5")
    p.add_argument("--include-scenario2", action="store_true")
    p.add_argument("--s2-episodes", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="results/multiseed")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    wanted = set(args.scenarios.split(","))

    if "3" in wanted:
        from scenario3_multicell_icic import run_scenario3
        df = _run_seeds(run_scenario3, seeds, metric_col="avg_reward",
                         episodes=30, num_steps=50, out_dir=str(out_dir / "_s3_tmp"))
        _summarize("scenario3", df, "avg_reward", out_dir)

    if "4" in wanted:
        from scenario4_mobility_blockage import run_scenario4
        df = _run_seeds(run_scenario4, seeds, metric_col="avg_reward",
                         episodes=60, num_steps=50, out_dir=str(out_dir / "_s4_tmp"))
        _summarize("scenario4", df, "avg_reward", out_dir)

    if "5" in wanted:
        # Scenario 5 has one "policy" (the best evolved genome) per run, not
        # a fixed set of named policies - summarize best fitness across seeds
        # directly rather than forcing it through the same policy-comparison shape.
        from scenario5_evo_optimizer import run_scenario5
        fitnesses = []
        orig_seed = CFG["global"]["seed"]
        for seed in seeds:
            CFG["global"]["seed"] = seed
            best = run_scenario5()
            fitnesses.append({"seed": seed, "best_fitness": best.fitness})
        CFG["global"]["seed"] = orig_seed
        df5 = pd.DataFrame(fitnesses)
        df5.to_csv(out_dir / "scenario5_per_seed.csv", index=False)
        print(f"\n[scenario5] Best fitness across {len(seeds)} seeds: "
              f"mean={df5['best_fitness'].mean():.2f}  std={df5['best_fitness'].std():.2f}")

    if args.include_scenario2 or "2" in wanted:
        from scenario2_react_agents import run_scenario2
        print(f"\n[scenario2] Running {len(seeds)} seeds x {args.s2_episodes} episodes x 4 variants - "
              f"this is the slow one, expect proportionally the same per-episode "
              f"cost you saw in your single-seed run.")
        df = _run_seeds(run_scenario2, seeds, metric_col="avg_reward",
                         episodes=args.s2_episodes, shift_at=args.s2_episodes // 2,
                         out_dir=str(out_dir / "_s2_tmp"))
        _summarize("scenario2", df, "avg_reward", out_dir)

    print(f"\nAll multi-seed summaries saved under {out_dir}/")


if __name__ == "__main__":
    main()
