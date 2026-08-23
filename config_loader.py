"""
config_loader.py  —  Load config.yaml into a dict available as CFG.

Usage (in any scenario file):
    from config_loader import CFG, get

    episodes = CFG["training"]["episodes"]
    seed     = get("global.seed", default=42)
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path


_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


CFG: dict = _load()


def get(dotpath: str, default=None):
    """
    Read a nested config value using dot-notation.

    Example:
        get("scenario5.fitness.lam_tp")   → 0.55
        get("llm.model")                  → "llama3"
    """
    parts = dotpath.split(".")
    node = CFG
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def reload() -> None:
    """Force reload from disk (useful in tests / interactive sessions)."""
    global CFG
    CFG = _load()
