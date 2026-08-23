"""
base_env.py  —  Abstract BaseEnv interface for all Scenarios.

Fixes addressed:
  #12 — Unified BaseEnv interface so every scenario environment is
         interchangeable and any agent policy can run against any env.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Canonical observation / info containers
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    snr_db: float = 0.0
    bler: float = 0.0
    throughput_mbps: float = 0.0
    channel_model: str = "TDL-C"
    mobility_kmh: float = 0.0
    step: int = 0
    extra: dict = field(default_factory=dict)   # scenario-specific extras


@dataclass
class StepInfo:
    reward: float = 0.0
    truncated: bool = False
    terminated: bool = False
    cost_tokens: int = 0          # LLM token usage this step (Issue #3)
    cost_wall_s: float = 0.0      # wall-clock seconds this step
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEnv(ABC):
    """
    Every scenario environment must subclass BaseEnv and implement:
        reset()  → Observation
        step()   → (Observation, StepInfo)
        render() → str

    Optional overrides:
        close()        — release resources (Sionna session, sockets …)
        action_space() — describe valid action bounds
    """

    def __init__(self, config: dict, rng_seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(rng_seed)
        self._step_count = 0
        self._episode_count = 0

    # -----------------------------------------------------------------------
    # Must implement
    # -----------------------------------------------------------------------

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment for a new episode. Returns initial observation."""
        ...

    @abstractmethod
    def step(self, action: dict) -> tuple[Observation, StepInfo]:
        """
        Execute one environment step.

        Parameters
        ----------
        action : dict  — must contain at least {"mcs": int, "pmi": int, "power_db": float}
                         scenario-specific keys go in action["extra"]

        Returns
        -------
        (Observation, StepInfo)
        """
        ...

    @abstractmethod
    def render(self) -> str:
        """Return human-readable state string (used in LLM prompt context)."""
        ...

    # -----------------------------------------------------------------------
    # Optional overrides
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Release external resources."""
        pass

    def action_space(self) -> dict:
        """Return dict describing valid action ranges."""
        return {
            "mcs":      {"type": "int",   "low": 0,  "high": 28},
            "pmi":      {"type": "int",   "low": 0,  "high": 15},
            "power_db": {"type": "float", "low": 10, "high": 30},
        }

    # -----------------------------------------------------------------------
    # Helpers available to all subclasses
    # -----------------------------------------------------------------------

    def _clip_action(self, action: dict) -> dict:
        """Clip incoming action to valid ranges from action_space()."""
        space = self.action_space()
        clipped = dict(action)
        for key, bounds in space.items():
            if key in clipped:
                clipped[key] = int(np.clip(clipped[key], bounds["low"], bounds["high"])) \
                    if bounds["type"] == "int" \
                    else float(np.clip(clipped[key], bounds["low"], bounds["high"]))
        return clipped

    def _advance(self) -> None:
        self._step_count += 1

    def _new_episode(self) -> None:
        self._step_count = 0
        self._episode_count += 1
