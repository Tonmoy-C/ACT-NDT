"""
llm_client.py  —  Real LLM integration via Ollama (local, free, Apple Silicon).

Fixes addressed:
  #1  — Wire real LLM API; remove _simple_heuristic_response()
  #10 — Structured JSON output instead of fragile regex action parser

Setup (one-time, on your Mac):
    brew install ollama
    ollama serve          # runs on localhost:11434
    ollama pull llama3    # ~4.7 GB; or use mistral / phi3 for smaller footprint

The client falls back gracefully:
  1. Try Ollama (local)
  2. Try OpenAI  (if OPENAI_API_KEY is set)
  3. Try Anthropic (if ANTHROPIC_API_KEY is set)
  4. Log warning and return safe heuristic (for CI / offline smoke tests only)
"""

from __future__ import annotations
import json
import os
import time
import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass

from config_loader import CFG

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    action: dict           # {"mcs": int, "pmi": int, "power_db": float}
    tokens_used: int = 0
    wall_s: float = 0.0
    provider: str = "none"
    error: str = ""


# ---------------------------------------------------------------------------
# Structured action parser  (fixes #10 — no more regex on freeform text)
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = CFG["llm"]["action_schema"]
_MCS_MIN   = _ACTION_SCHEMA["mcs"]["min"]
_MCS_MAX   = _ACTION_SCHEMA["mcs"]["max"]
_PMI_MIN   = _ACTION_SCHEMA["pmi"]["min"]
_PMI_MAX   = _ACTION_SCHEMA["pmi"]["max"]
_PWR_MIN   = _ACTION_SCHEMA["power_db"]["min"]
_PWR_MAX   = _ACTION_SCHEMA["power_db"]["max"]

_SAFE_ACTION = {
    "mcs": 10,
    "pmi": 0,
    "power_db": 20.0,
}

_JSON_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object and nothing else — no explanation, "
    "no markdown fences.  The object must contain exactly these keys:\n"
    '  {"mcs": <int 0-28>, "pmi": <int 0-15>, "power_db": <float 10-30>}\n'
    "Values outside the ranges will be clipped automatically."
)


def _parse_action(raw: str) -> tuple[dict, str]:
    """
    Extract a valid action dict from the model's raw text response.
    Priority:
      1. Parse entire response as JSON.
      2. Find first {...} block and parse that.
      3. Fall back to safe default (never crash).

    Returns (action_dict, error_message_or_"")
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```[a-z]*", "", raw).strip().strip("`").strip()

    # Attempt 1: whole string
    try:
        obj = json.loads(cleaned)
        return _clip_action(obj), ""
    except json.JSONDecodeError:
        pass

    # Attempt 2: first {...} block
    m = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            return _clip_action(obj), ""
        except json.JSONDecodeError:
            pass

    # Fallback
    err = f"Could not parse JSON action from: {raw[:120]!r}"
    log.warning(err)
    return dict(_SAFE_ACTION), err


def _clip_action(obj: dict) -> dict:
    """Clip action values to valid ranges from config."""
    return {
        "mcs":      int(min(max(int(obj.get("mcs",      _SAFE_ACTION["mcs"])),      _MCS_MIN), _MCS_MAX)),
        "pmi":      int(min(max(int(obj.get("pmi",      _SAFE_ACTION["pmi"])),      _PMI_MIN), _PMI_MAX)),
        "power_db": float(min(max(float(obj.get("power_db", _SAFE_ACTION["power_db"])), _PWR_MIN), _PWR_MAX)),
    }


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str) -> LLMResponse:
    url     = CFG["llm"]["base_url"] + "/api/generate"
    model   = CFG["llm"]["model"]
    timeout = CFG["llm"]["timeout_s"]
    full_prompt = prompt + _JSON_INSTRUCTION
    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": CFG["llm"]["temperature"],
            "num_predict": CFG["llm"]["max_tokens"],
        }
    }).encode()

    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = data.get("response", "")
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        wall = time.perf_counter() - t0
        action, err = _parse_action(text)
        return LLMResponse(text=text, action=action, tokens_used=tokens,
                           wall_s=wall, provider="ollama", error=err)
    except Exception as exc:
        wall = time.perf_counter() - t0
        return LLMResponse(text="", action=dict(_SAFE_ACTION),
                           wall_s=wall, provider="ollama", error=str(exc))


def _call_openai(prompt: str) -> LLMResponse:
    """Requires: pip install openai   +   export OPENAI_API_KEY=sk-..."""
    try:
        import openai  # type: ignore
        client = openai.OpenAI()
        full_prompt = prompt + _JSON_INSTRUCTION
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=CFG["llm"]["temperature"],
            max_tokens=CFG["llm"]["max_tokens"],
            response_format={"type": "json_object"},   # native JSON mode
        )
        wall = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        action, err = _parse_action(text)
        return LLMResponse(text=text, action=action, tokens_used=tokens,
                           wall_s=wall, provider="openai", error=err)
    except Exception as exc:
        return LLMResponse(text="", action=dict(_SAFE_ACTION),
                           wall_s=0.0, provider="openai", error=str(exc))


def _call_anthropic(prompt: str) -> LLMResponse:
    """Requires: pip install anthropic   +   export ANTHROPIC_API_KEY=sk-ant-..."""
    try:
        import anthropic as _ant  # type: ignore
        client = _ant.Anthropic()
        full_prompt = prompt + _JSON_INSTRUCTION
        t0 = time.perf_counter()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=CFG["llm"]["max_tokens"],
            messages=[{"role": "user", "content": full_prompt}],
        )
        wall = time.perf_counter() - t0
        text = msg.content[0].text if msg.content else ""
        tokens = (msg.usage.input_tokens or 0) + (msg.usage.output_tokens or 0)
        action, err = _parse_action(text)
        return LLMResponse(text=text, action=action, tokens_used=tokens,
                           wall_s=wall, provider="anthropic", error=err)
    except Exception as exc:
        return LLMResponse(text="", action=dict(_SAFE_ACTION),
                           wall_s=0.0, provider="anthropic", error=str(exc))


def _heuristic_fallback(obs_text: str) -> LLMResponse:
    """
    OFFLINE FALLBACK — used only when no LLM backend is reachable.
    Labelled clearly so it never silently masks LLM failures in production.
    """
    log.warning("[LLMClient] All backends unreachable — using heuristic fallback.")
    return LLMResponse(
        text="[HEURISTIC FALLBACK — no LLM reachable]",
        action=dict(_SAFE_ACTION),
        provider="heuristic",
        error="no_backend",
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client.  Provider priority:
      config.yaml llm.provider  →  env-var override  →  auto-detect

    Usage:
        client = LLMClient()
        response = client.generate(prompt_text)
        action   = response.action          # {"mcs": int, "pmi": int, "power_db": float}
        tokens   = response.tokens_used
    """

    def __init__(self):
        self._provider = os.environ.get("LLM_PROVIDER", CFG["llm"]["provider"])

    def generate(self, prompt: str) -> LLMResponse:
        """Generate an action from the LLM given a prompt string."""
        if self._provider == "ollama":
            resp = _call_ollama(prompt)
            if not resp.error:
                return resp
            # Cascade to cloud if Ollama is down
            log.warning(f"Ollama failed ({resp.error}), trying cloud …")

        if os.environ.get("OPENAI_API_KEY"):
            resp = _call_openai(prompt)
            if not resp.error:
                return resp

        if os.environ.get("ANTHROPIC_API_KEY"):
            resp = _call_anthropic(prompt)
            if not resp.error:
                return resp

        return _heuristic_fallback(prompt)
