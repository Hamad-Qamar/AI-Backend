"""
shared_models.py — Cross-module singletons.

Why this exists:
    The plan requires "load once, share everywhere" so that we don't
    pay the (slow) SentenceTransformer load + CPU memory cost more than
    once per process. Every module that needs embeddings or the optional
    LLM should obtain it via this module.

Usage:
    from .shared_models import get_embedding_model
    model = get_embedding_model()           # default all-MiniLM-L6-v2
    model = get_embedding_model("custom")   # override if ever needed
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Internal cache: model_name -> instance
_EMBEDDING_CACHE: Dict[str, "SentenceTransformer"] = {}  # type: ignore[name-defined]
_EMBEDDING_LOCK = threading.Lock()

# Internal cache for the optional LLM (Flan-T5 small / base)
_LLM_CACHE: Dict[str, object] = {}
_LLM_LOCK = threading.Lock()

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-small"


def get_embedding_model(name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Return a process-wide cached SentenceTransformer.

    Loads the model lazily on first call. Subsequent calls return the
    same instance, so all modules share one set of weights in memory.
    """
    cached = _EMBEDDING_CACHE.get(name)
    if cached is not None:
        return cached

    with _EMBEDDING_LOCK:
        cached = _EMBEDDING_CACHE.get(name)
        if cached is not None:
            return cached

        logger.info("shared_models: loading embedding model '%s' (one-time)", name)
        from sentence_transformers import SentenceTransformer
        instance = SentenceTransformer(name)
        _EMBEDDING_CACHE[name] = instance
        return instance


# Sentinel stored in _LLM_CACHE to remember a previous load failure and
# avoid retry storms (every request would otherwise re-trigger a download).
_LLM_FAILED = object()


class _Seq2SeqWrapper:
    """
    Thin wrapper exposing the same call signature as a HuggingFace
    text2text-generation pipeline. We build it manually because newer
    `transformers` releases dropped the "text2text-generation" task alias
    from `pipeline(task=...)`. Loading the seq2seq model + tokenizer
    directly is forward-compatible.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        prompt,
        max_new_tokens: int = 80,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        **_: object,
    ):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [{"generated_text": text} for text in decoded]


def get_llm_pipeline(name: str = DEFAULT_LLM_MODEL, enabled: bool = True):
    """
    Return a process-wide cached seq2seq generation wrapper, or None if
    the LLM is disabled or fails to load.

    The plan demands strict, throttled use of the LLM. We don't pre-load
    it — callers should request it only when they actually want
    explanation/refinement output. A previous load failure is cached so
    we don't retry on every request.
    """
    if not enabled:
        return None

    cached = _LLM_CACHE.get(name)
    if cached is _LLM_FAILED:
        return None
    if cached is not None:
        return cached

    with _LLM_LOCK:
        cached = _LLM_CACHE.get(name)
        if cached is _LLM_FAILED:
            return None
        if cached is not None:
            return cached

        try:
            logger.info("shared_models: loading LLM pipeline '%s' (one-time)", name)
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            instance = _Seq2SeqWrapper(model=model, tokenizer=tokenizer)
            _LLM_CACHE[name] = instance
            logger.info("shared_models: LLM '%s' ready", name)
            return instance
        except Exception as exc:  # pragma: no cover — runtime safety net
            logger.warning(
                "shared_models: failed to load LLM '%s' (will not retry): %s",
                name, exc,
            )
            _LLM_CACHE[name] = _LLM_FAILED
            return None
