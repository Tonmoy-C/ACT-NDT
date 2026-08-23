"""
cost_logger.py  —  Per-step cost logging for all Scenarios.

Fixes addressed:
  #3  — S1–S4 now emit real wall-time, sionna-call counts, and token usage
         so Scenario 7's meta-analysis operates on real data.

Usage (wrap any scenario's step loop):
    from cost_logger import CostLogger

    logger = CostLogger(experiment="exp2", provider="ollama")
    ...
    with logger.step(episode=ep, step=st):
        obs, info = env.step(action)
        logger.record(
            reward=info.reward,
            bler=obs.bler,
            throughput=obs.throughput_mbps,
            tokens=llm_response.tokens_used,
            sionna_calls=1,
        )
    logger.save()   # writes logs/exp2_cost_log.csv
"""

from __future__ import annotations
import csv
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config_loader import CFG


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------

@dataclass
class CostRow:
    experiment: str   = ""
    provider: str     = ""
    episode: int      = 0
    step: int         = 0
    wall_s: float     = 0.0
    tokens_in: int    = 0
    tokens_out: int   = 0
    tokens_total: int = 0
    sionna_calls: int = 0
    reward: float     = 0.0
    bler: float       = 0.0
    throughput_mbps: float = 0.0
    usd_cost: float   = 0.0
    error: str        = ""


# ---------------------------------------------------------------------------
# CostLogger
# ---------------------------------------------------------------------------

class CostLogger:
    def __init__(self, experiment: str, provider: Optional[str] = None):
        self.experiment = experiment
        self.provider   = provider or CFG["llm"]["provider"]
        self._rows: list[CostRow] = []
        self._current: Optional[CostRow] = None
        self._t0: float = 0.0

        # Token cost table from config
        costs = CFG["scenario7"]["token_cost_per_1k"]
        self._cost_per_token = costs.get(self.provider, 0.0) / 1000.0

        log_dir = Path(CFG["global"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / f"{experiment}_cost_log.csv"

    # -----------------------------------------------------------------------
    # Context manager for a single step
    # -----------------------------------------------------------------------

    @contextmanager
    def step(self, episode: int, step: int):
        """
        Usage:
            with logger.step(ep, st):
                ... do env.step / LLM call ...
                logger.record(...)
        """
        self._current = CostRow(
            experiment=self.experiment,
            provider=self.provider,
            episode=episode,
            step=step,
        )
        self._t0 = time.perf_counter()
        try:
            yield self
        except Exception as exc:
            if self._current:
                self._current.error = str(exc)
            raise
        finally:
            if self._current:
                self._current.wall_s = time.perf_counter() - self._t0
                self._rows.append(self._current)
                self._current = None

    def record(
        self,
        reward: float = 0.0,
        bler: float = 0.0,
        throughput: float = 0.0,
        tokens: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        sionna_calls: int = 0,
        error: str = "",
    ) -> None:
        """Call inside a `with logger.step(...)` block to attach KPIs."""
        if self._current is None:
            raise RuntimeError("record() must be called inside a logger.step() context")
        tok_in  = tokens_in  or (tokens // 2)
        tok_out = tokens_out or (tokens - tokens // 2)
        self._current.tokens_in    = tok_in
        self._current.tokens_out   = tok_out
        self._current.tokens_total = tok_in + tok_out
        self._current.sionna_calls = sionna_calls
        self._current.reward       = reward
        self._current.bler         = bler
        self._current.throughput_mbps = throughput
        self._current.usd_cost     = (tok_in + tok_out) * self._cost_per_token
        self._current.error        = error

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self) -> Path:
        """Write all rows to CSV. Returns the file path."""
        if not self._rows:
            return self._path
        fieldnames = list(asdict(CostRow()).keys())
        file_exists = self._path.exists()
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in self._rows:
                writer.writerow(asdict(row))
        self._rows.clear()
        return self._path

    # -----------------------------------------------------------------------
    # Summary (printed / logged at end of each scenario run)
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        """Return summary stats over all rows saved so far."""
        import csv as _csv
        if not self._path.exists():
            return {}
        rows = []
        with open(self._path) as f:
            reader = _csv.DictReader(f)
            for r in reader:
                if r["experiment"] == self.experiment:
                    rows.append(r)
        if not rows:
            return {}
        total_wall   = sum(float(r["wall_s"])        for r in rows)
        total_tokens = sum(int(r["tokens_total"])     for r in rows)
        total_usd    = sum(float(r["usd_cost"])       for r in rows)
        total_sionna = sum(int(r["sionna_calls"])     for r in rows)
        avg_reward   = sum(float(r["reward"])         for r in rows) / len(rows)
        avg_bler     = sum(float(r["bler"])           for r in rows) / len(rows)
        return {
            "experiment":    self.experiment,
            "steps":         len(rows),
            "total_wall_s":  round(total_wall, 2),
            "total_tokens":  total_tokens,
            "total_usd":     round(total_usd, 6),
            "sionna_calls":  total_sionna,
            "avg_reward":    round(avg_reward, 4),
            "avg_bler":      round(avg_bler, 4),
        }
