"""
generate_all_plots.py — one script, every plot, based on whatever results
currently exist on disk.

Reads from results/*.csv and results/multiseed/*.csv. Anything not run yet
is skipped with a note rather than crashing - run whichever scenarios you
care about, then run this, and it plots exactly what's there.

Usage:
    python generate_all_plots.py                    # reads ./results, writes ./results/plots
    python generate_all_plots.py --shift-at 20       # Scenario 2's shift episode, if not 40/2 default
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_ORDER = ["pedestrian", "vehicular_shift", "blockage_shift"]


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  skip: {path.name} not found yet")
        return None
    return pd.read_csv(path)


def plot_scenario1(results_dir: Path, out_dir: Path) -> None:
    df = _load(results_dir / "exp1_baseline_results.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, col in zip(axes, ["reward", "bler", "throughput"]):
        ax.plot(df["step"], df[col])
        ax.set_xlabel("step")
        ax.set_ylabel(col)
        ax.set_title(f"Scenario 1: {col} vs step")
    fig.tight_layout()
    fig.savefig(out_dir / "s1_baseline.png", dpi=150)
    plt.close(fig)
    print("  wrote s1_baseline.png")


def plot_scenario2(results_dir: Path, out_dir: Path, shift_at: int) -> None:
    df = _load(results_dir / "exp2_full_results.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for policy, group in df.groupby("policy"):
        axes[0].plot(group["episode"], group["avg_reward"], label=policy, alpha=0.8)
        axes[1].plot(group["episode"], group["avg_bler"], label=policy, alpha=0.8)
    for ax, title in zip(axes, ["avg_reward vs episode", "avg_bler vs episode"]):
        ax.axvline(shift_at, color="gray", linestyle="--", linewidth=1, label="channel shift")
        ax.set_xlabel("episode")
        ax.set_title(f"Scenario 2: {title}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "s2_shift_robustness.png", dpi=150)
    plt.close(fig)
    print("  wrote s2_shift_robustness.png")


def plot_scenario3(results_dir: Path, out_dir: Path) -> None:
    df = _load(results_dir / "exp3_multicell_results.csv")
    if df is None:
        return
    agg = df.groupby(["policy", "traffic_mode"])[["avg_reward", "avg_fairness"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, col in zip(axes, ["avg_reward", "avg_fairness"]):
        pivot = agg.pivot(index="policy", columns="traffic_mode", values=col)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Scenario 3: {col} by policy x traffic")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "s3_multicell.png", dpi=150)
    plt.close(fig)
    print("  wrote s3_multicell.png")


def plot_scenario4(results_dir: Path, out_dir: Path) -> None:
    df = _load(results_dir / "exp4_shift_results.csv")
    if df is None:
        return
    agg = df.groupby(["policy", "phase"])["avg_reward"].mean().reset_index()
    present_phases = [p for p in PHASE_ORDER if p in agg["phase"].unique()]
    pivot = agg.pivot(index="policy", columns="phase", values="avg_reward")[present_phases]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Scenario 4: avg_reward by policy across mobility/blockage phases")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "s4_mobility_blockage.png", dpi=150)
    plt.close(fig)
    print("  wrote s4_mobility_blockage.png")


def plot_scenario5(results_dir: Path, out_dir: Path) -> None:
    df = _load(results_dir / "exp5_generation_history.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["generation"], df["best_fitness"], marker="o", label="best")
    ax.plot(df["generation"], df["avg_fitness"], marker="o", label="population avg")
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.set_title("Scenario 5: evolutionary convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "s5_convergence.png", dpi=150)
    plt.close(fig)
    print("  wrote s5_convergence.png")


def plot_scenario6(results_dir: Path, out_dir: Path) -> None:
    df = _load(results_dir / "exp6_grid_search_results.csv")
    if df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.boxplot(column="accuracy", by="activation", ax=axes[0])
    axes[0].set_title("Scenario 6: accuracy by activation (all 320 configs)")
    axes[0].set_xlabel("")
    fig.suptitle("")

    for act, group in df.groupby("activation"):
        axes[1].scatter(group["wall_s"], group["accuracy"], label=act, alpha=0.6, s=15)
    axes[1].set_xlabel("wall time (s)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Scenario 6: accuracy vs training time")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "s6_grid_search.png", dpi=150)
    plt.close(fig)
    print("  wrote s6_grid_search.png")


def plot_scenario7(results_dir: Path, out_dir: Path) -> None:
    df = _load(results_dir / "exp7_cost_summary.csv")
    if df is None:
        return
    real = df[df["status"] == "real data"].dropna(subset=["wall_time_s", "main_metric_value"])
    if real.empty:
        print("  skip: exp7 has no real data yet")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(real["wall_time_s"], real["main_metric_value"], s=80)
    for _, r in real.iterrows():
        ax.annotate(r["label"], (r["wall_time_s"], r["main_metric_value"]), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("main metric (avg reward)")
    ax.set_title("Scenario 7: cost vs performance across scenarios")
    fig.tight_layout()
    fig.savefig(out_dir / "s7_cost_vs_performance.png", dpi=150)
    plt.close(fig)
    print("  wrote s7_cost_vs_performance.png (duplicate of scenario7_cost_analysis.py's own plot)")


def plot_multiseed(results_dir: Path, out_dir: Path) -> None:
    ms_dir = results_dir / "multiseed"
    if not ms_dir.exists():
        print("  skip: results/multiseed/ not found - run run_multiseed.py first")
        return

    for name in ["scenario3", "scenario4"]:
        summary = _load(ms_dir / f"{name}_summary.csv")
        if summary is None:
            continue
        metric_col = [c for c in summary.columns if c.endswith("_mean")][0]
        base = metric_col.replace("_mean", "")
        std_col = f"{base}_std"
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(summary["policy"], summary[metric_col], yerr=summary[std_col], capsize=4)
        ax.set_ylabel(base)
        ax.set_title(f"{name}: {base} mean +/- std across {int(summary['n_seeds'].iloc[0])} seeds")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / f"multiseed_{name}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote multiseed_{name}.png")

    s5 = _load(ms_dir / "scenario5_per_seed.csv")
    if s5 is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(s5["seed"].astype(str), s5["best_fitness"])
        ax.set_xlabel("seed")
        ax.set_ylabel("best fitness")
        ax.set_title(f"Scenario 5: best fitness per seed "
                     f"(mean={s5['best_fitness'].mean():.2f}, std={s5['best_fitness'].std():.2f})")
        fig.tight_layout()
        fig.savefig(out_dir / "multiseed_scenario5.png", dpi=150)
        plt.close(fig)
        print("  wrote multiseed_scenario5.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--shift-at", type=int, default=20, help="Scenario 2's channel-shift episode")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scenario 1:"); plot_scenario1(results_dir, out_dir)
    print("Scenario 2:"); plot_scenario2(results_dir, out_dir, args.shift_at)
    print("Scenario 3:"); plot_scenario3(results_dir, out_dir)
    print("Scenario 4:"); plot_scenario4(results_dir, out_dir)
    print("Scenario 5:"); plot_scenario5(results_dir, out_dir)
    print("Scenario 6:"); plot_scenario6(results_dir, out_dir)
    print("Scenario 7:"); plot_scenario7(results_dir, out_dir)
    print("Multi-seed:"); plot_multiseed(results_dir, out_dir)

    n = len(list(out_dir.glob("*.png")))
    print(f"\n{n} plot(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
