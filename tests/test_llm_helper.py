"""
Tests for LLMHelper. The HF pipeline is mocked so tests stay fast and
hermetic — no model download, no CPU inference.
"""

from typing import Any, Dict, List

import pytest

from ai_modules.llm_helper import LLMHelper, _SmallLRU


# ── pipeline mocks ───────────────────────────────────────────────────────────


class _FakePipeline:
    """Minimal stand-in that records prompts and returns a canned reply."""

    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.prompts: List[str] = []

    def __call__(self, prompt: str, **kwargs: Any) -> List[Dict[str, str]]:
        self.prompts.append(prompt)
        return [{"generated_text": self.reply}]


class _FailingPipeline:
    def __call__(self, *args: Any, **kwargs: Any):
        raise RuntimeError("simulated model failure")


# ── helper to build a helper with the pipeline pre-injected ──────────────────


def _make_helper(pipe: Any) -> LLMHelper:
    h = LLMHelper(preload=False)
    h._pipe = pipe
    return h


# ── _SmallLRU ────────────────────────────────────────────────────────────────


def test_small_lru_eviction_policy():
    lru = _SmallLRU(max_size=2)
    lru.put("x", "1")
    lru.put("y", "2")
    lru.get("x")          # promote x
    lru.put("z", "3")     # should evict y

    assert lru.get("x") == "1"
    assert lru.get("y") is None
    assert lru.get("z") == "3"


# ── _generate ────────────────────────────────────────────────────────────────


def test_generate_returns_text_and_caches_prompt():
    pipe = _FakePipeline(reply="hello world")
    helper = _make_helper(pipe)

    out1 = helper._generate("anything")
    out2 = helper._generate("anything")

    assert out1 == "hello world"
    assert out2 == "hello world"
    # The pipeline must only be invoked once because the second call hit cache.
    assert len(pipe.prompts) == 1


def test_generate_returns_empty_string_when_pipeline_fails():
    helper = _make_helper(_FailingPipeline())
    assert helper._generate("anything") == ""


def test_generate_returns_empty_string_for_empty_prompt():
    helper = _make_helper(_FakePipeline())
    assert helper._generate("") == ""


def test_generate_collapses_inline_whitespace_but_preserves_newlines():
    """
    Newlines are intentionally preserved (the tip parser uses them as
    a structural separator). Inline runs of spaces/tabs are collapsed.
    """
    pipe = _FakePipeline(reply="  multiple   spaces \n and\nnewlines  ")
    helper = _make_helper(pipe)
    assert helper._generate("p") == "multiple spaces\nand\nnewlines"


# ── explain_recommendation ───────────────────────────────────────────────────


def test_explain_recommendation_returns_empty_when_no_items():
    helper = _make_helper(_FakePipeline(reply="x"))
    assert helper.explain_recommendation("Bricks", "Standard", "Lahore", []) == ""


def test_explain_recommendation_includes_category_and_city_in_prompt():
    pipe = _FakePipeline(reply="Suitable.")
    helper = _make_helper(pipe)
    items = [{"item_name": "Khaprail brick", "brand": "Local"}]

    out = helper.explain_recommendation("Bricks", "Premium", "Karachi", items)

    assert out == "Suitable."
    assert len(pipe.prompts) == 1
    prompt_lower = pipe.prompts[0].lower()
    assert "bricks" in prompt_lower
    assert "karachi" in prompt_lower
    assert "premium" in prompt_lower
    assert "khaprail brick" in prompt_lower


# ── rewrite_kb_answer ────────────────────────────────────────────────────────


def test_rewrite_kb_answer_falls_back_to_original_on_empty_output():
    pipe = _FakePipeline(reply="")
    helper = _make_helper(pipe)

    original = "BuildHive verifies suppliers via NTN and bank checks."
    out = helper.rewrite_kb_answer("How are suppliers verified?", original)

    assert out == original


def test_rewrite_kb_answer_uses_llm_output_when_non_empty():
    pipe = _FakePipeline(reply="Polite rewrite.")
    helper = _make_helper(pipe)

    out = helper.rewrite_kb_answer("q", "original text")
    assert out == "Polite rewrite."


# ── cost_saving_tips ─────────────────────────────────────────────────────────


def test_cost_saving_tips_zero_total_returns_empty():
    helper = _make_helper(_FakePipeline(reply="anything"))
    assert helper.cost_saving_tips(0, "Lahore", "standard") == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1. Bulk-buy materials\n2. Compare 3 supplier quotes",
         ["Bulk-buy materials", "Compare 3 supplier quotes"]),
        ("- Reuse formwork - Negotiate labour rates",
         ["Reuse formwork", "Negotiate labour rates"]),
        ("Bulk buy cement, source bricks locally, hire experienced labour",
         ["Bulk buy cement", "source bricks locally"]),
    ],
)
def test_cost_saving_tips_permissive_parser(raw, expected):
    helper = _make_helper(_FakePipeline(reply=raw))
    tips = helper.cost_saving_tips(1_000_000, "Lahore", "standard", max_tips=2)
    assert tips == expected
