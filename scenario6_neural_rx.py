"""
scenario6_neural_rx.py  —  Neural receiver hyperparameter search.

Fixes addressed:
  #2  — Replaces fully-synthetic (no-signal) data with physically-motivated
         channel data that actually has a learnable SNR structure.
         When Sionna is available, generates via REST twin.
         Otherwise uses a structured NumPy surrogate that preserves
         the SNR → BER relationship.
  #5  — True exhaustive grid search (all 5×4×4×4 = 320 configs evaluated).
         Random-subsampling mode retained as an optional fast path.

Requires:
    pip install numpy scikit-learn tqdm
    (optional) pip install sionna   # for real Sionna data generation
"""

from __future__ import annotations
import itertools
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config_loader import CFG, get
from cost_logger import CostLogger
from phy_models import compute_bler, compute_effective_snr

_S6 = CFG["scenario6"]


# ---------------------------------------------------------------------------
# 1.  Dataset generation  (fix #2)
# ---------------------------------------------------------------------------

def _generate_sionna_data(n_samples: int, snr_db_range=(0, 30)) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempt to fetch real channel data from the Sionna REST twin.
    Returns (X, y) where X is the received symbol matrix and y is the
    transmitted symbol index (4-QAM: 0-3).
    """
    url = CFG["global"]["sionna_rest_url"] + "/api/v1/channel_samples"
    payload = json.dumps({
        "channel_model": "TDL-C",
        "n_samples": n_samples,
        "snr_range_db": list(snr_db_range),
        "modulation": "QPSK",
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    X = np.array(data["received_symbols"], dtype=np.float32)   # (N, 2) I/Q
    y = np.array(data["labels"], dtype=np.int32)                # (N,)
    return X, y


def _generate_structured_surrogate(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Physically-motivated surrogate (no Sionna dependency):
    - Sample random SNR per symbol from U(0 dB, 30 dB)
    - Generate QPSK symbol (one of 4 constellation points)
    - Add AWGN scaled by SNR
    - True label is the original symbol index

    This produces a real learnable signal because a neural network CAN learn
    to decode QPSK symbols from noisy I/Q observations.
    """
    rng = np.random.default_rng(seed)

    # QPSK constellation points (normalised)
    constellation = np.array([
        [ 1,  1],
        [-1,  1],
        [-1, -1],
        [ 1, -1],
    ], dtype=np.float32) / np.sqrt(2)

    n = n_samples
    snr_db  = rng.uniform(0.0, 30.0, size=n).astype(np.float32)
    labels  = rng.integers(0, 4, size=n)                         # true symbol
    symbols = constellation[labels]                               # (N, 2)

    noise_std = 10.0 ** (-snr_db / 20.0)                         # σ per dimension
    noise     = rng.standard_normal((n, 2)).astype(np.float32) * noise_std[:, None]
    received  = symbols + noise                                   # (N, 2)

    # Append SNR as a feature so the network can learn SNR-conditional decoding
    X = np.concatenate([received, snr_db[:, None]], axis=1)      # (N, 3)
    y = labels.astype(np.int32)
    return X, y


def load_or_generate_dataset(n_samples: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Load real dataset from disk, generate via Sionna, or fall back to surrogate.
    Caches to data/ directory.
    """
    real_path = Path(_S6["real_data_path"])
    if real_path.exists():
        d = np.load(real_path)
        print(f"[S6] Loaded real dataset from {real_path}  ({len(d['X'])} samples)")
        return d["X"], d["y"]

    if not _S6.get("use_synthetic_fallback", True) and CFG["global"]["sionna_rest_url"]:
        try:
            print("[S6] Fetching data from Sionna REST twin …")
            X, y = _generate_sionna_data(n_samples)
            real_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(real_path, X=X, y=y)
            print(f"[S6] Saved {len(X)} samples to {real_path}")
            return X, y
        except Exception as exc:
            print(f"[S6] Sionna REST failed ({exc}); using structured surrogate.")

    print("[S6] Generating physically-motivated surrogate dataset …")
    X, y = _generate_structured_surrogate(n_samples, seed=seed)
    real_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(real_path, X=X, y=y)
    return X, y


# ---------------------------------------------------------------------------
# 2.  Hyperparameter grid  (fix #5 — exhaustive, not subsampled)
# ---------------------------------------------------------------------------

def build_grid() -> list[dict]:
    """
    Build the full Cartesian product of the hyperparameter grid from config.
    For strategy="random", randomly subsample to budget.
    """
    g = _S6["hyperparameter_grid"]
    all_configs = [
        {
            "hidden_layers": hl,
            "hidden_dim":    hd,
            "activation":    act,
            "learning_rate": lr,
        }
        for hl  in g["hidden_layers"]
        for hd  in g["hidden_dim"]
        for act in g["activation"]
        for lr  in g["learning_rate"]
    ]

    strategy = _S6.get("search_strategy", "exhaustive")
    if strategy == "random":
        budget = _S6.get("random_budget", 100)
        rng = np.random.default_rng(CFG["global"]["seed"])
        indices = rng.choice(len(all_configs), size=min(budget, len(all_configs)), replace=False)
        selected = [all_configs[i] for i in sorted(indices)]
        print(f"[S6] Random search: {len(selected)}/{len(all_configs)} configs")
        return selected

    print(f"[S6] Exhaustive grid: {len(all_configs)} configs")
    return all_configs


# ---------------------------------------------------------------------------
# 3.  Evaluate a single config
# ---------------------------------------------------------------------------

def evaluate_config(
    cfg: dict,
    X_train, X_test, y_train, y_test,
    n_epochs: int = 5,
) -> dict:
    """Train and evaluate one MLP config. Returns metrics dict."""
    # Build hidden layer sizes tuple
    hidden_sizes = tuple([cfg["hidden_dim"]] * cfg["hidden_layers"])

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    t0 = time.perf_counter()
    model = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=cfg["activation"],
        learning_rate_init=cfg["learning_rate"],
        max_iter=n_epochs * 10,
        random_state=CFG["global"]["seed"],
        solver="adam",
        early_stopping=False,
    )
    model.fit(Xtr, y_train)
    wall_s = time.perf_counter() - t0

    acc   = float(model.score(Xte, y_test))
    # SER = symbol error rate; BER approximated as SER/log2(M) for QPSK
    ser   = 1.0 - acc
    ber   = ser / 2.0    # log2(4) = 2

    return {
        **cfg,
        "accuracy": round(acc, 5),
        "ser":      round(ser, 5),
        "ber":      round(ber, 5),
        "wall_s":   round(wall_s, 3),
    }


# ---------------------------------------------------------------------------
# 4.  Main search runner
# ---------------------------------------------------------------------------

def run_scenario6(n_samples: int = 2000, seed: int = 42, out_dir: str = "results") -> list[dict]:
    """
    Full S6 pipeline:
      1. Load/generate dataset
      2. Build grid
      3. Exhaustively evaluate (or random subsample)
      4. Return sorted results
    """
    logger = CostLogger(experiment="exp6", provider=CFG["llm"]["provider"])
    X, y = load_or_generate_dataset(n_samples=n_samples, seed=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    configs = build_grid()
    results = []

    n_failed = 0
    for i, cfg in enumerate(tqdm(configs, desc="S6 grid search")):
        ep, st = 0, i
        try:
            with logger.step(episode=ep, step=st):
                res = evaluate_config(cfg, X_tr, X_te, y_tr, y_te,
                                      n_epochs=_S6["eval_epochs"])
                logger.record(
                    reward=res["accuracy"],
                    bler=res["ber"],
                    throughput=0.0,
                    sionna_calls=0,
                )
            results.append(res)
        except Exception as exc:
            # One bad hyperparameter combination (e.g. a numerically unstable
            # config) must not lose the other 319 evaluations - log and continue.
            n_failed += 1
            print(f"[S6] config {i} failed ({cfg}): {exc}")
        # Flush incrementally every 25 configs so a late crash doesn't lose
        # everything evaluated so far (this loop can run for a while).
        if (i + 1) % 25 == 0:
            logger.save()

    if n_failed:
        print(f"[S6] {n_failed}/{len(configs)} configs failed and were skipped.")
    logger.save()

    # Sort by accuracy descending
    results.sort(key=lambda r: r["accuracy"], reverse=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "exp6_grid_search_results.csv", index=False)

    print("\n[S6] Top-5 configurations:")
    for rank, r in enumerate(results[:5], 1):
        print(f"  {rank}. layers={r['hidden_layers']} dim={r['hidden_dim']} "
              f"act={r['activation']} lr={r['learning_rate']:.0e} "
              f"→ acc={r['accuracy']:.4f}  BER={r['ber']:.5f}  {r['wall_s']:.2f}s")

    return results


if __name__ == "__main__":
    run_scenario6()
