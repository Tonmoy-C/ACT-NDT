"""
scenario5_evo_optimizer.py  —  Evolutionary policy optimisation.

Fixes addressed:
  #7  — Full Eq.5 fitness (all 6 terms) via phy_models.compute_fitness_eq5()
  #8  — Mobility coefficient bug fixed via phy_models.tdl_channel_penalty_db()
  #15 — Parallel candidate evaluation via ProcessPoolExecutor

All hyperparameters read from config.yaml [scenario5].
"""

from __future__ import annotations
import copy
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config_loader import CFG
from phy_models import (
    compute_bler,
    compute_effective_snr,
    compute_reward_single_link,
    compute_fitness_eq5,
    tdl_channel_penalty_db,
    spectral_efficiency,
    BLER_PROFILES,
)
from cost_logger import CostLogger

_S5 = CFG["scenario5"]
_FIT = _S5["fitness"]


# ---------------------------------------------------------------------------
# Policy chromosome representation
# ---------------------------------------------------------------------------

@dataclass
class PolicyChromosome:
    """
    A single candidate policy parameterised as a threshold-based controller.
    Fields represent tunable policy parameters (what the evolutionary search
    optimises over).
    """
    mcs_base: int            = 10     # baseline MCS when BLER is nominal
    mcs_max:  int            = 24     # cap under high mobility
    pmi_pref: int            = 0      # preferred PMI index
    power_db: float          = 20.0   # transmit power
    bler_up_thresh:   float  = 0.12   # raise MCS if BLER below this
    bler_down_thresh: float  = 0.08   # lower MCS if BLER above this
    power_boost_db:   float  = 2.0    # extra power when BLER is critical
    fitness: float           = -999.0
    id: int                  = 0


def random_chromosome(rng: np.random.Generator, chrom_id: int = 0) -> PolicyChromosome:
    return PolicyChromosome(
        mcs_base         = int(rng.integers(4, 20)),
        mcs_max          = int(rng.integers(20, 28)),
        pmi_pref         = int(rng.integers(0, 16)),
        power_db         = float(rng.uniform(10.0, 30.0)),
        bler_up_thresh   = float(rng.uniform(0.05, 0.15)),
        bler_down_thresh = float(rng.uniform(0.08, 0.20)),
        power_boost_db   = float(rng.uniform(0.5, 5.0)),
        id               = chrom_id,
    )


# ---------------------------------------------------------------------------
# Policy execution (single episode)
# ---------------------------------------------------------------------------

def _run_episode(
    chrom: PolicyChromosome,
    base_snr: float,
    mobility_kmh: float,
    channel_model: str,
    steps: int,
    seed: int,
) -> dict:
    """
    Execute one episode with a chromosome policy.
    Returns dict of per-episode aggregate KPIs.
    """
    rng = np.random.default_rng(seed)
    profile = _S5["bler_profile"]

    rewards, blers, throughputs = [], [], []
    mcs = chrom.mcs_base

    for _ in range(steps):
        # Channel penalty uses canonical model (fixes #8)
        ch_penalty, blockage = tdl_channel_penalty_db(
            channel_model=channel_model,
            mobility_kmh=mobility_kmh,
            blockage_prob=0.02,
            rng=rng,
        )
        noisy_snr = base_snr + rng.normal(0, 1.5)
        eff_snr   = compute_effective_snr(
            base_snr=noisy_snr,
            power_db=chrom.power_db,
            pmi=chrom.pmi_pref,
            channel_penalty_db=ch_penalty,
        )
        bler = compute_bler(eff_snr, mcs, profile=profile)
        se   = spectral_efficiency(mcs)
        tput = se * 20.0 * (1.0 - bler)   # 20 MHz BW

        reward = compute_reward_single_link(
            throughput_mbps=tput,
            bler=bler,
            bler_target=CFG["phy"]["bler_target"],
            blockage_hit=blockage,
        )

        # Policy action: adjust MCS based on BLER thresholds
        if bler < chrom.bler_up_thresh and mcs < chrom.mcs_max:
            mcs = min(mcs + 1, chrom.mcs_max)
        elif bler > chrom.bler_down_thresh and mcs > 0:
            mcs = max(mcs - 2, 0)

        rewards.append(reward)
        blers.append(bler)
        throughputs.append(tput)

    return {
        "avg_reward":     float(np.mean(rewards)),
        "avg_bler":       float(np.mean(blers)),
        "avg_throughput": float(np.mean(throughputs)),
    }


def _evaluate_candidate(
    chrom: PolicyChromosome,
    runs: int,
    steps: int,
    base_snr: float,
    mobility_kmh: float,
    channel_model: str,
    seed_base: int,
) -> float:
    """
    Evaluate one chromosome over `runs` independent episodes.
    Returns Eq.5 fitness score.
    """
    agg_reward = agg_bler = agg_tput = 0.0
    for r in range(runs):
        ep = _run_episode(
            chrom=chrom,
            base_snr=base_snr,
            mobility_kmh=mobility_kmh,
            channel_model=channel_model,
            steps=steps,
            seed=seed_base + chrom.id * 100 + r,
        )
        agg_reward += ep["avg_reward"]
        agg_bler   += ep["avg_bler"]
        agg_tput   += ep["avg_throughput"]

    n = float(runs)
    return compute_fitness_eq5(
        avg_throughput=agg_tput / n,
        avg_bler=agg_bler / n,
        avg_reward=agg_reward / n,
        lam_tp=_FIT["lam_tp"],
        lam_bler=_FIT["lam_bler"],
        lam_reward=_FIT["lam_reward"],
        lam_fair=_FIT["lam_fair"],
        lam_int=_FIT["lam_int"],
        lam_cost=_FIT["lam_cost"],
        bler_target=_FIT["bler_target"],
    )


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _mutate(chrom: PolicyChromosome, rng: np.random.Generator, rate: float) -> PolicyChromosome:
    """Single-parameter mutation (shared operator)."""
    c = copy.copy(chrom)
    if rng.random() < rate:
        c.mcs_base       = int(np.clip(c.mcs_base       + rng.integers(-3, 4),  0, 25))
    if rng.random() < rate:
        c.mcs_max        = int(np.clip(c.mcs_max        + rng.integers(-2, 3), c.mcs_base, 28))
    if rng.random() < rate:
        c.pmi_pref       = int(np.clip(c.pmi_pref       + rng.integers(-2, 3),  0, 15))
    if rng.random() < rate:
        c.power_db       = float(np.clip(c.power_db     + rng.normal(0, 1.5),  10.0, 30.0))
    if rng.random() < rate:
        c.bler_up_thresh = float(np.clip(c.bler_up_thresh + rng.normal(0, 0.02), 0.03, 0.20))
    if rng.random() < rate:
        c.bler_down_thresh = float(np.clip(c.bler_down_thresh + rng.normal(0, 0.02),
                                           c.bler_up_thresh, 0.30))
    if rng.random() < rate:
        c.power_boost_db = float(np.clip(c.power_boost_db + rng.normal(0, 0.5), 0.0, 6.0))
    return c


def _crossover(p1: PolicyChromosome, p2: PolicyChromosome,
                rng: np.random.Generator) -> tuple[PolicyChromosome, PolicyChromosome]:
    """Uniform crossover over numeric fields."""
    c1, c2 = copy.copy(p1), copy.copy(p2)
    fields = ["mcs_base", "mcs_max", "pmi_pref", "power_db",
              "bler_up_thresh", "bler_down_thresh", "power_boost_db"]
    for f in fields:
        if rng.random() < 0.5:
            setattr(c1, f, getattr(p2, f))
            setattr(c2, f, getattr(p1, f))
    return c1, c2


# ---------------------------------------------------------------------------
# Main evolutionary loop  (fix #15 — parallel evaluation)
# ---------------------------------------------------------------------------

def run_scenario5(verbose: bool = True, out_dir: str = "results") -> PolicyChromosome:
    """
    Full evolutionary optimisation for Scenario 5.
    Returns the best-found chromosome.
    """
    seed         = CFG["global"]["seed"]   # read fresh, NOT the module-level _SEED
                                            # (that constant is cached at import time
                                            # and silently ignores later CFG changes -
                                            # confirmed by run_multiseed.py producing
                                            # identical fitness across different seeds
                                            # until this was fixed)
    rng          = np.random.default_rng(seed)
    pop_size     = _S5["population_size"]
    generations  = _S5["generations"]
    runs         = _S5["runs_per_candidate"]
    steps        = _S5["steps_per_run"]
    mut_rate     = _S5["mutation_rate"]
    xover_rate   = _S5["crossover_rate"]
    elite_n      = _S5["elite_count"]
    n_workers    = _S5["parallel_workers"]
    base_snr     = CFG["phy"]["base_snr_db"]
    mobility     = _S5["mobility_kmh"]
    ch_model     = CFG["channel"]["default_model"]

    logger = CostLogger(experiment="exp5")

    # Initialise population
    population: list[PolicyChromosome] = [
        random_chromosome(rng, chrom_id=i) for i in range(pop_size)
    ]

    best_ever: Optional[PolicyChromosome] = None
    gen_history: list[dict] = []

    for gen in range(generations):
        t_gen = time.perf_counter()

        # --- Parallel fitness evaluation (fix #15) ---
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _evaluate_candidate,
                    chrom, runs, steps, base_snr, mobility, ch_model, seed + gen
                ): chrom
                for chrom in population
            }
            for future in as_completed(futures):
                chrom = futures[future]
                try:
                    chrom.fitness = future.result()
                except Exception as exc:
                    chrom.fitness = -999.0
                    print(f"  [S5] Candidate {chrom.id} failed: {exc}")

        population.sort(key=lambda c: c.fitness, reverse=True)
        best = population[0]

        if best_ever is None or best.fitness > best_ever.fitness:
            best_ever = copy.copy(best)

        gen_wall = time.perf_counter() - t_gen
        gen_record = {
            "generation": gen,
            "best_fitness": best.fitness,
            "avg_fitness":  np.mean([c.fitness for c in population]),
            "wall_s":       gen_wall,
        }
        gen_history.append(gen_record)

        with logger.step(episode=gen, step=0):
            logger.record(
                reward=best.fitness,
                bler=0.0,
                throughput=0.0,
            )

        if verbose:
            print(f"  Gen {gen+1:3d}/{generations}  "
                  f"best={best.fitness:+.4f}  "
                  f"avg={gen_record['avg_fitness']:+.4f}  "
                  f"wall={gen_wall:.1f}s")

        # --- Elitism + crossover + mutation ---
        elites   = [copy.copy(c) for c in population[:elite_n]]
        children = []
        while len(children) < pop_size - elite_n:
            # Tournament selection (k=3)
            def tournament(k=3):
                candidates = rng.choice(len(population), size=k, replace=False)
                return max((population[i] for i in candidates), key=lambda c: c.fitness)

            p1, p2 = tournament(), tournament()
            if rng.random() < xover_rate:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = copy.copy(p1), copy.copy(p2)
            children += [_mutate(c1, rng, mut_rate), _mutate(c2, rng, mut_rate)]

        # Re-assign IDs and combine
        next_pop = elites + children[:pop_size - elite_n]
        for i, c in enumerate(next_pop):
            c.id = i
        population = next_pop

    logger.save()

    hist_df = pd.DataFrame(gen_history)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(out / "exp5_generation_history.csv", index=False)

    if verbose and best_ever:
        print(f"\n[S5] Best policy found (fitness={best_ever.fitness:+.4f}):")
        print(f"      mcs_base={best_ever.mcs_base}  mcs_max={best_ever.mcs_max}"
              f"  pmi={best_ever.pmi_pref}  power={best_ever.power_db:.1f} dBm"
              f"  bler_thresh=[{best_ever.bler_down_thresh:.3f},{best_ever.bler_up_thresh:.3f}]")

    return best_ever


if __name__ == "__main__":
    run_scenario5()
