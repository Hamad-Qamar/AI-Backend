"""
Loads the cost-estimation assistant knowledge tables (coefficient / range reference)
from `config/cost_estimation_assistant_knowledge.md` for chat policy and future LLM context.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _knowledge_path() -> Path:
    p = Path(__file__).resolve().parent.parent / "config" / "cost_estimation_assistant_knowledge.md"
    if p.exists():
        return p
    return Path(os.getcwd()) / "config" / "cost_estimation_assistant_knowledge.md"


@lru_cache(maxsize=1)
def load_cost_estimation_assistant_knowledge() -> str:
    path = _knowledge_path()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def clear_cost_assistant_knowledge_cache() -> None:
    """For tests or hot-reload of the markdown file."""
    load_cost_estimation_assistant_knowledge.cache_clear()
