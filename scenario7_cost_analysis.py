"""
scenario7_cost_analysis.py — Cost/efficiency meta-analysis across Scenarios 1-6.

LOWEST-CONFIDENCE FILE IN THIS BUNDLE. The docx walkthrough itself labels its
own Scenario 7 section "The Intended Design ... (From Spec)" - meaning even
that earlier analysis was inferring this file's behavior from a description,
not reading real source. Treat this as a reasonable-effort implementation of
the *described* design, built fresh against the real cost_logger.py schema -
not a reconstruction of verified original code the way Scenarios 1-6 are.

What it does: globs logs/exp*_cost_log.csv (written by cost_logger.py, fix
#3) for Scenarios 1-6, rolls each up to one summary row per experiment, and -
matching the spec's described fallback-safe behavior - fills a placeholder
row for any experiment whose log doesn't exist yet, so this script is always
runnable even before every scenario has been run.

Run:
    python scenario7_cost_analysis.py
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config_loader import CFG
from cost_logger import CostLogger

_S7 = CFG["scenario7"]
EXPERIMENTS = [f"exp{i}" for i in range(1, 7)]
LABELS = {"exp1": "PHY Baseline", "exp2": "ReAct Agents", "exp3": "Multi-Cell ICIC",
          "exp4": "Mobility/Blockage", "exp5": "Evolutionary Opt.", "exp6": "Neural Rx Search"}


def _load_cost_log(experiment: str) -> pd.DataFrame | None:
    log_dir = Path(CFG["global"]["log_dir"])
    path = log_dir / f"{experiment}_cost_log.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["experiment"] == experiment] if "experiment" in df.columns else df


def _summarize(experiment: str) -> dict:
    df = _load_cost_log(experiment)
    if df is None or df.empty:
        return {
            "experiment": experiment, "label": LABELS[experiment], "status": "PLACEHOLDER (no cost log yet)",
            "wall_time_s": np.nan, "episodes": np.nan, "sionna_calls": np.nan,
            "tokens_in": np.nan, "tokens_out": np.nan, "usd_cost": np.nan,
            "main_metric_name": "n/a", "main_metric_value": np.nan,
        }
    n_episodes = int(df["episode"].nunique()) if "episode" in df.columns else len(df)
    return {
        "experiment": experiment, "label": LABELS[experiment], "status": "real data",
        "wall_time_s": round(float(df["wall_s"].sum()), 2),
        "episodes": n_episodes,
        "sionna_calls": int(df["sionna_calls"].sum()) if "sionna_calls" in df.columns else 0,
        "tokens_in": int(df["tokens_in"].sum()) if "tokens_in" in df.columns else 0,
        "tokens_out": int(df["tokens_out"].sum()) if "tokens_out" in df.columns else 0,
        "usd_cost": round(float(df["usd_cost"].sum()), 6) if "usd_cost" in df.columns else 0.0,
        "main_metric_name": "avg_reward",
        "main_metric_value": round(float(df["reward"].mean()), 4) if "reward" in df.columns else np.nan,
    }


def run_scenario7(out_dir: str = "results") -> pd.DataFrame:
    found = glob.glob(_S7["cost_log_glob"])
    print(f"[S7] Found {len(found)} cost log file(s) matching {_S7['cost_log_glob']!r}")

    rows = [_summarize(exp) for exp in EXPERIMENTS]
    df = pd.DataFrame(rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "exp7_cost_summary.csv", index=False)

    n_placeholder = int((df["status"] != "real data").sum())
    if n_placeholder:
        print(f"[S7] {n_placeholder}/{len(df)} experiments have no cost log yet - "
              f"run those scenarios first (they now all call cost_logger.py, fix #3) "
              f"to replace the placeholder rows with real data.")

    print(f"\n[S7] Saved summary to {out / 'exp7_cost_summary.csv'}")
    print(df[["experiment", "label", "status", "wall_time_s", "sionna_calls", "usd_cost", "main_metric_value"]]
          .to_string(index=False))

    _plot_cost_vs_performance(df, out)
    return df


def _plot_cost_vs_performance(df: pd.DataFrame, out: Path) -> None:
    real = df[df["status"] == "real data"].dropna(subset=["wall_time_s", "main_metric_value"])
    if real.empty:
        print("[S7] No real data yet to plot - skipping exp7_cost_vs_performance.png")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(real["wall_time_s"], real["main_metric_value"], s=80)
        for _, r in real.iterrows():
            ax.annotate(r["label"], (r["wall_time_s"], r["main_metric_value"]), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Main metric (avg reward)")
        ax.set_title("Cost vs. performance across Scenarios")
        fig.tight_layout()
        fig.savefig(out / "exp7_cost_vs_performance.png", dpi=150)
        print(f"[S7] Saved plot to {out / 'exp7_cost_vs_performance.png'}")
    except ImportError:
        print("[S7] matplotlib not installed - skipping plot (pip install matplotlib)")


if __name__ == "__main__":
    run_scenario7()
