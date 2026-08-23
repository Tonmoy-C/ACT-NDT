"""
phy_models.py  —  Shared physical-layer kernels for all Scenarios.

Fixes addressed:
  #6  — Standardised BLER sigmoid coefficients (one source of truth)
  #7  — Full Eq.5 fitness function (throughput + reward + BLER + fairness + interference + cost)
  #8  — Unified sigmoid with scenario-selectable profile
  #9  — Canonical TDL shift penalty (same dB model used in S2 and S4)
  #13 — S5 mobility coefficient bug fixed (0.03 → 0.30)
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# 1.  BLER sigmoid profiles  (fixes #6, #8)
# ---------------------------------------------------------------------------
# Each profile is (slope_k, threshold_base, threshold_mcs_coeff)
# SINR threshold = threshold_base + threshold_mcs_coeff * mcs
# BLER = 1 / (1 + exp(slope_k * (effective_snr - threshold)))

BLER_PROFILES = {
    "default":   {"k": 0.90, "base": 10.0, "mcs_coeff": 0.55},  # S1/S2
    "mobility":  {"k": 0.78, "base":  9.0, "mcs_coeff": 0.62},  # S4
    "evolution": {"k": 0.82, "base":  9.5, "mcs_coeff": 0.58},  # S5
    "neural_rx": {"k": 0.85, "base":  9.2, "mcs_coeff": 0.60},  # S6
}


def compute_bler(effective_snr: float, mcs: int, profile: str = "default") -> float:
    """
    Canonical sigmoid BLER model shared across ALL scenarios.

    Parameters
    ----------
    effective_snr : float  — SNR after power/pmi/penalty adjustments (dB)
    mcs           : int    — modulation and coding scheme index
    profile       : str    — one of BLER_PROFILES keys

    Returns
    -------
    bler : float in [0.001, 0.999]
    """
    p = BLER_PROFILES[profile]
    threshold = p["base"] + p["mcs_coeff"] * mcs
    raw = 1.0 / (1.0 + math.exp(p["k"] * (effective_snr - threshold)))
    return float(np.clip(raw, 0.001, 0.999))


def compute_effective_snr(
    base_snr: float,
    power_db: float = 20.0,
    pmi: int = 0,
    power_sensitivity: float = 0.15,
    pmi_penalty: float = 0.25,
    channel_penalty_db: float = 0.0,
) -> float:
    """
    Effective SNR = base_snr + power_delta - pmi_loss - channel_loss.
    All callers use this so the sensitivity coefficients are consistent.
    """
    return (
        base_snr
        + power_sensitivity * (power_db - 20.0)
        - pmi_penalty * pmi
        - channel_penalty_db
    )


# ---------------------------------------------------------------------------
# 2.  Channel shift penalties  (fixes #9 — canonical dB model for TDL shifts)
# ---------------------------------------------------------------------------

def tdl_channel_penalty_db(
    channel_model: str = "TDL-C",
    mobility_kmh: float = 0.0,
    blockage_prob: float = 0.0,
    blockage_loss_db: float = 20.0,
    rng: np.random.Generator = None,
) -> tuple[float, int]:
    """
    Compute the total SNR penalty (dB) for the current channel conditions.
    Replaces the scenario-specific inline penalty logic in S2, S4, S5.

    Returns
    -------
    (penalty_db, blockage_hit)
        penalty_db   : float  — total SNR reduction to subtract from base SNR
        blockage_hit : int    — 1 if a blockage event occurred this step
    """
    if rng is None:
        rng = np.random.default_rng()

    penalty = 0.0

    # TDL-D adds richer multipath → extra Doppler / estimation error
    if channel_model == "TDL-D":
        penalty += 1.5

    # Mobility penalty — fixed steps matching 3GPP Doppler categories
    # Fix #13: was 0.07 * kmh * 0.03 = negligible; now matches S4 design
    if mobility_kmh >= 60:
        penalty += 4.5   # vehicular / high-speed
    elif mobility_kmh >= 30:
        penalty += 2.0   # suburban
    elif mobility_kmh >= 10:
        penalty += 0.8   # pedestrian-fast

    # Stochastic blockage event
    blockage_hit = 0
    if blockage_prob > 0 and rng.random() < blockage_prob:
        penalty += blockage_loss_db
        blockage_hit = 1

    return penalty, blockage_hit


# ---------------------------------------------------------------------------
# 3.  Reward functions  (unified across scenarios)
# ---------------------------------------------------------------------------

def compute_reward_single_link(
    throughput_mbps: float,
    bler: float,
    bler_target: float = 0.10,
    w_tp: float = 1.0,
    w_bler: float = 20.0,
    blockage_hit: int = 0,
    blockage_penalty: float = 0.8,
) -> float:
    """Single-link reward (Scenarios 1, 2, 4, 5)."""
    r = w_tp * throughput_mbps - w_bler * max(0.0, bler - bler_target)
    if blockage_hit:
        r -= blockage_penalty
    return float(r)


def compute_reward_multi_cell(
    avg_throughput: float,
    fairness: float,
    p5_throughput: float,
    ho_failure_rate: float,
    ping_pong_ratio: float,
    avg_interference: float,
    w_tp: float = 0.45,
    w_fair: float = 12.0,
    w_p5: float = 0.35,
    w_ho: float = 8.0,
    w_pp: float = 6.0,
    w_int: float = 0.15,
) -> float:
    """Multi-cell reward (Scenario 3)."""
    return (
        w_tp   * avg_throughput
        + w_fair * fairness
        + w_p5   * p5_throughput
        - w_ho   * ho_failure_rate
        - w_pp   * ping_pong_ratio
        - w_int  * avg_interference
    )


# ---------------------------------------------------------------------------
# 4.  Full Eq.5 fitness function  (fixes #7 — S5 previously dropped terms)
# ---------------------------------------------------------------------------

def compute_fitness_eq5(
    avg_throughput: float,
    avg_bler: float,
    avg_reward: float,
    avg_fairness: float = 1.0,
    avg_interference: float = 0.0,
    cost_proxy: float = 0.0,
    lam_tp: float   = 0.55,
    lam_bler: float = 8.0,
    lam_reward: float = 0.45,
    lam_fair: float  = 0.0,    # set > 0 for multi-cell scenarios
    lam_int: float   = 0.0,    # set > 0 when interference is tracked
    lam_cost: float  = 0.0,    # set > 0 for cost-aware optimisation
    bler_target: float = 0.10,
) -> float:
    """
    Full Eq.5 from the paper:
        J(π) = λ_T·T - λ_B·BLER + λ_F·F - λ_I·I - λ_C·C + λ_R·R

    For single-link scenarios (S5), lam_fair = lam_int = lam_cost = 0 by default.
    For multi-cell scenarios, pass the appropriate λ weights.
    """
    return (
        lam_tp     * avg_throughput
        - lam_bler * max(0.0, avg_bler - bler_target)
        + lam_reward * avg_reward
        + lam_fair * avg_fairness
        - lam_int  * avg_interference
        - lam_cost * cost_proxy
    )


# ---------------------------------------------------------------------------
# 5.  Jain's fairness index
# ---------------------------------------------------------------------------

def jains_fairness(throughputs: np.ndarray) -> float:
    """Exact Jain's index: (Σr_u)² / (n · Σr_u²)"""
    s = throughputs.sum()
    ss = np.square(throughputs).sum()
    return float((s ** 2) / (len(throughputs) * ss + 1e-9))


# ---------------------------------------------------------------------------
# 6.  Spectral efficiency helper
# ---------------------------------------------------------------------------

def spectral_efficiency(mcs: int, base: float = 1.2, slope: float = 0.18) -> float:
    """Linear MCS → spectral efficiency mapping (bits/s/Hz)."""
    return max(0.8, base + slope * mcs)
