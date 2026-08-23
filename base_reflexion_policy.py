"""
base_reflexion_policy.py  —  Shared Reflexion agent parent class.

Fixes addressed:
  #13 — The identical Reflexion lesson-generation pattern that was duplicated
         across Scenarios 2, 3, and 4 is now a single parent class.
         Each scenario subclasses it and overrides only:
           - kpi_thresholds   (dict of KPI name → threshold value)
           - _lesson_from_kpi (str template per KPI violation)
           - build_prompt     (scenario-specific observation → prompt)

Usage (Scenario 2):
    from base_reflexion_policy import BaseReflexionPolicy

    class ReActPolicy(BaseReflexionPolicy):
        kpi_thresholds = {"bler": 0.10, "throughput": 5.0}

        def _lesson_from_kpi(self, kpi, value):
            if kpi == "bler" and value > self.kpi_thresholds["bler"]:
                return f"BLER {value:.3f} exceeded target; prefer lower MCS."
            if kpi == "throughput" and value < self.kpi_thresholds["throughput"]:
                return f"Throughput {value:.1f} Mbps below target; increase MCS if BLER allows."
            return ""

        def build_prompt(self, obs, history):
            ...
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from llm_client import LLMClient, LLMResponse
from config_loader import CFG


class BaseReflexionPolicy(ABC):
    """
    Abstract parent for all Reflexion-style LLM policies.

    Subclasses must implement:
        kpi_thresholds : dict  — {"kpi_name": threshold_value, ...}
        _lesson_from_kpi(kpi, value) → str | ""
        build_prompt(obs, history)   → str
    """

    # Override in subclass
    kpi_thresholds: dict = {}

    def __init__(self, memory_window: int | None = None, scenario_key: str | None = None):
        """
        memory_window : explicit override, takes priority if given.
        scenario_key  : e.g. "scenario3" - looks up CFG[scenario_key]["memory_window"].
                        Falls back to CFG["scenario2"]["memory_window"] only if
                        neither is given and scenario2's key happens to exist,
                        so old call sites keep working, but every real subclass
                        should pass its own scenario_key (fix: this used to be
                        hardcoded to scenario2 for every subclass).
        """
        self._client  = LLMClient()
        self._lessons: list[str] = []
        self._history: list[dict] = []        # rolling episode KPI records
        if memory_window is not None:
            self._memory_window = memory_window
        elif scenario_key is not None:
            self._memory_window = CFG.get(scenario_key, {}).get("memory_window", 5)
        else:
            self._memory_window = CFG.get("scenario2", {}).get("memory_window", 5)
        self._adapt_flags: dict   = {}        # modulate next-episode behaviour

    # -----------------------------------------------------------------------
    # Core Reflexion loop
    # -----------------------------------------------------------------------

    def on_episode_start(self, episode_num: int) -> None:
        """No-op by default; override if a subclass needs per-episode setup.
        Exists so BaseReflexionPolicy satisfies the same on_episode_start/
        on_episode_end lifecycle every other policy class in this framework
        uses, even when used standalone (not multiply-inherited with a
        BasePolicy that happened to already provide this)."""
        pass

    def act(self, obs: Any) -> tuple[dict, LLMResponse]:
        """
        Build prompt → call LLM → return (action, response).
        Lessons and history are automatically injected into the prompt.
        """
        prompt = self._compose_prompt(obs)
        response = self._client.generate(prompt)
        return response.action, response

    def on_episode_end(self, episode_kpis: dict) -> None:
        """
        Call at the end of every episode with a dict of average KPIs.
        Generates lessons, updates memory window, sets adaptation flags.

        Example:
            policy.on_episode_end({"bler": 0.12, "throughput": 4.8})
        """
        # Generate lessons for any KPI that violates its threshold
        new_lessons: list[str] = []
        for kpi, value in episode_kpis.items():
            lesson = self._lesson_from_kpi(kpi, value)
            if lesson:
                new_lessons.append(lesson)

        # Keep only unique lessons (avoid repetition)
        for lesson in new_lessons:
            if lesson not in self._lessons:
                self._lessons.append(lesson)

        # Trim lessons list to avoid prompt overflow (keep 10 most recent)
        self._lessons = self._lessons[-10:]

        # Update rolling history window
        self._history.append(episode_kpis)
        self._history = self._history[-self._memory_window:]

        # Update adaptation flags based on KPI trends
        self._update_adapt_flags(episode_kpis)

    # -----------------------------------------------------------------------
    # Must implement in subclass
    # -----------------------------------------------------------------------

    @abstractmethod
    def _lesson_from_kpi(self, kpi: str, value: float) -> str:
        """
        Return a lesson string if `value` violates the threshold for `kpi`,
        otherwise return "".
        """
        ...

    @abstractmethod
    def build_prompt(self, obs: Any, history: list[dict]) -> str:
        """
        Construct the full LLM prompt string from the current observation
        and rolling history.  Lessons will be injected automatically.
        """
        ...

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _compose_prompt(self, obs: Any) -> str:
        """Inject lessons and history into the subclass-built prompt."""
        base = self.build_prompt(obs, self._history)
        if self._lessons:
            lesson_block = "\n".join(f"  • {l}" for l in self._lessons)
            base += f"\n\nLessons from previous episodes:\n{lesson_block}"
        return base

    def _update_adapt_flags(self, kpis: dict) -> None:
        """
        Set scenario-agnostic adaptation flags that the subclass prompt
        can inspect via self._adapt_flags.
        """
        bler = kpis.get("bler", 0.0)
        tp   = kpis.get("throughput", float("inf"))
        thr  = self.kpi_thresholds

        self._adapt_flags["bler_critical"]        = bler > thr.get("bler", 0.15) * 1.5
        self._adapt_flags["throughput_critical"]  = tp   < thr.get("throughput", 0.0) * 0.5
        self._adapt_flags["shift_detected"]       = self._adapt_flags.get("shift_detected", False)

    def notify_shift(self) -> None:
        """Call when a channel shift is detected; injects a shift lesson."""
        lesson = "Channel model shift detected — reset MCS to conservative value and re-explore."
        if lesson not in self._lessons:
            self._lessons.append(lesson)
        self._adapt_flags["shift_detected"] = True
