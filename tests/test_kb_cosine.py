"""
Sanity tests for the KB cosine pattern (Step M): build a tiny normalized
IndexFlatIP, search it, and verify (a) self-search returns cos≈1, and
(b) results are sorted by descending cosine similarity. This locks in
the normalize_L2 + IndexFlatIP idiom we use in ChatBotModule.
"""

import faiss
import numpy as np
import pytest


def _build_normalized_index(vectors: np.ndarray) -> faiss.Index:
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, vectors


def test_self_search_returns_cosine_one():
    rng = np.random.default_rng(seed=0)
    raw = rng.standard_normal((5, 16)).astype("float32")
    index, normalized = _build_normalized_index(raw)

    query = normalized[2:3].copy()  # already normalized
    sims, idxs = index.search(query, k=5)

    assert idxs[0][0] == 2
    assert sims[0][0] == pytest.approx(1.0, abs=1e-5)


def test_results_are_sorted_by_descending_similarity():
    rng = np.random.default_rng(seed=1)
    raw = rng.standard_normal((20, 32)).astype("float32")
    index, _ = _build_normalized_index(raw)

    q = rng.standard_normal((1, 32)).astype("float32")
    faiss.normalize_L2(q)
    sims, _ = index.search(q, k=10)

    # Strictly non-increasing.
    for i in range(len(sims[0]) - 1):
        assert sims[0][i] >= sims[0][i + 1] - 1e-6


def test_normalized_inner_product_equals_cosine():
    # If a · b are normalized, dot(a, b) is true cosine similarity.
    a = np.array([[3.0, 4.0]], dtype="float32")
    b = np.array([[1.0, 0.0]], dtype="float32")

    a_norm = a.copy()
    b_norm = b.copy()
    faiss.normalize_L2(a_norm)
    faiss.normalize_L2(b_norm)

    expected = float(np.dot(a_norm[0], b_norm[0]))
    index = faiss.IndexFlatIP(2)
    index.add(b_norm)
    sims, _ = index.search(a_norm, k=1)
    assert float(sims[0][0]) == pytest.approx(expected, abs=1e-6)
    # 3/5 = 0.6
    assert float(sims[0][0]) == pytest.approx(0.6, abs=1e-6)
