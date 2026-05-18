"""
LLMHelper — strict, throttled wrapper around the optional Flan-T5 pipeline.

Per the BuildHive plan, the LLM is used ONLY for:
  - Recommendation: 1-line "why these materials" explanation per top category
  - Chatbot: friendly rewrite of a KB answer
  - Cost: 1-2 cost-saving tips for a project total

It is NEVER used for:
  - Search / retrieval
  - Filtering
  - Calculations / BOQ
  - Core logic decisions

Default state is OFF. The pipeline is loaded lazily on first use and
shared via shared_models.get_llm_pipeline(). All generations are wrapped
in try/except — any failure returns an empty string, never raises, and
callers always have a graceful fallback to rule-based output.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .shared_models import get_llm_pipeline, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)


class _SmallLRU:
    """Tiny in-process LRU so we don't re-prompt the model for the same input."""

    def __init__(self, max_size: int = 128):
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._max = max_size

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)


class LLMHelper:
    """
    Thin wrapper over a HuggingFace text2text-generation pipeline.

    Usage policy:
      - Caller decides whether to use it (`use_llm=True`).
      - Helper is safe to call when the LLM is unavailable: it returns "".
      - Cached: identical prompts hit the in-process LRU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        preload: bool = False,
    ):
        self.model_name = model_name
        self._pipe = None
        self._cache = _SmallLRU(max_size=256)
        if preload:
            self._ensure()

    # ── infra ────────────────────────────────────────────────────────────────

    def _ensure(self):
        if self._pipe is None:
            self._pipe = get_llm_pipeline(self.model_name, enabled=True)
        return self._pipe

    @property
    def is_available(self) -> bool:
        try:
            return self._ensure() is not None
        except Exception:
            return False

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        do_sample: bool = False,
    ) -> str:
        """Single generate call with caching and full safety net."""
        if not prompt:
            return ""
        cached = self._cache.get(prompt)
        if cached is not None:
            return cached

        try:
            pipe = self._ensure()
            if pipe is None:
                return ""
            out = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_return_sequences=1,
            )
            text = (out[0].get("generated_text") or "").strip()
            # Collapse runs of spaces/tabs but PRESERVE newlines so the
            # tip parser can use line breaks as a structural separator.
            text = re.sub(r"[ \t]+", " ", text)
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if text:
                self._cache.put(prompt, text)
            return text
        except Exception as exc:
            logger.warning("LLMHelper generate failed: %s", exc)
            return ""

    # ── plan-approved use cases ──────────────────────────────────────────────

    def explain_recommendation(
        self,
        category: str,
        quality: str,
        city: Optional[str],
        items: List[Dict[str, Any]],
        max_items: int = 3,
    ) -> str:
        """
        ONE-line explanation for why a top-category set is suitable.
        Strictly opinion/explanation — never used to pick or filter products.

        Few-shot prompt: Flan-T5 small follows examples better than open
        instructions. The example fixes the shape ("Why X is suitable: …")
        so we get a complete sentence, not a prompt echo.
        """
        if not items:
            return ""

        item_summary = ", ".join(
            f"{it.get('item_name', 'N/A')}".strip()
            for it in items[:max_items]
        )
        location = f"in {city}" if city else "in Pakistan"
        quality_norm = (quality or "Standard").lower()

        prompt = (
            "Task: write one short sentence (under 25 words) explaining why "
            "the listed building materials suit the project. Do not list "
            "the items again. Do not repeat the question.\n\n"
            "Example:\n"
            "Project: a Standard cement house in Lahore.\n"
            "Materials: DG Khan Cement, Maple Leaf Cement.\n"
            "Reason: These cement brands are widely available in Lahore and "
            "meet Pakistan Building Code strength requirements at a "
            "mid-range price.\n\n"
            f"Project: a {quality_norm} {category.lower()} build {location}.\n"
            f"Materials: {item_summary}.\n"
            "Reason:"
        )
        return self._generate(prompt, max_new_tokens=64)

    def rewrite_kb_answer(self, question: str, answer: str) -> str:
        """
        Rewrite a KB answer to be concise, structured, and actionable.
        Rules: BuildHive context only, max 5 lines, no vague filler.
        Returns the original answer unchanged if the LLM is unavailable.
        """
        if not answer:
            return answer
        if not self.is_available:
            return answer

        # Truncate long inputs so Flan-T5 small doesn't get overwhelmed.
        short_answer = answer[:350] if len(answer) > 350 else answer

        prompt = (
            "You are a BuildHive marketplace assistant. Rules:\n"
            "- Answer ONLY about BuildHive and construction topics.\n"
            "- Be concise: max 5 lines.\n"
            "- If 'how to', give numbered steps.\n"
            "- End with one short tip starting with 'Tip:'.\n"
            "- Do NOT repeat the question or add unrelated info.\n\n"
            "Example:\n"
            "Question: How to add a listing?\n"
            "Answer: Go to Seller Dashboard. Click Add New Listing. Fill details and publish.\n"
            "Rewrite: To add a listing: 1. Open Seller Dashboard 2. Click Add New Listing "
            "3. Fill in details and publish. Tip: Add photos to get more views.\n\n"
            f"Question: {question}\n"
            f"Answer: {short_answer}\n"
            "Rewrite:"
        )
        result = self._generate(prompt, max_new_tokens=96)
        return result or answer

    def cost_saving_tips(
        self,
        total_pkr: float,
        city: str,
        grade: str,
        max_tips: int = 2,
    ) -> List[str]:
        """
        Generate up to N short, generic cost-reduction suggestions.
        Used purely as commentary on top of rule-based tips.
        """
        if total_pkr <= 0 or not self.is_available:
            return []

        # Few-shot: give the model a concrete example so it produces tips
        # in the same shape rather than echoing the instruction. Each tip
        # is short, on its own line, and starts with an action verb.
        prompt = (
            "Task: list practical ways to reduce the cost of a construction "
            "project in Pakistan. Each tip on its own line. Each tip starts "
            "with an action verb (Buy, Use, Hire, Compare, Reuse, Negotiate, "
            "Source). No numbering. No prompt echo.\n\n"
            "Example:\n"
            "Project: 5 marla economy house in Multan, total PKR 3,200,000.\n"
            "Tips:\n"
            "Compare quotes from at least three brick suppliers before ordering.\n"
            "Buy cement in bulk at the start of the project to lock today's price.\n\n"
            f"Project: {grade} construction in {city}, Pakistan, "
            f"total PKR {int(total_pkr):,}.\n"
            f"Tips:\n"
        )
        text = self._generate(prompt, max_new_tokens=160)
        if not text:
            return []

        # Accept several output styles from a small model: numbered lines,
        # bulleted lines, " - " separated, ";" separated, or a single
        # comma-separated sentence.
        raw_parts: List[str] = []
        for line in text.splitlines():
            # Within a line, also break on " - ", ";" and ",". Avoid
            # splitting on bare "-" (which would chop hyphenated words like
            # "Bulk-buy"); require whitespace on both sides.
            for sub in re.split(r"\s+-\s+|;|,", line):
                raw_parts.append(sub)

        tips: List[str] = []
        seen: set = set()
        for part in raw_parts:
            cleaned = part.strip(" -•\t.;:")
            if not cleaned:
                continue
            # Strip leading numbering like "1." or "1)" or "1:".
            m = re.match(r"^\d+[.)\:]\s*", cleaned)
            if m:
                cleaned = cleaned[m.end():].strip()
            if len(cleaned) < 4:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            tips.append(cleaned)
            if len(tips) >= max_tips:
                break
        return tips
