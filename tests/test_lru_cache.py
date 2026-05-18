"""Pure-logic tests for the small LRU cache used by RecommendationModule."""

from ai_modules.recommendation_module import _LRUCache


def test_lru_get_missing_returns_none():
    cache = _LRUCache(max_size=4)
    assert cache.get("missing") is None


def test_lru_put_then_get_returns_value():
    cache = _LRUCache(max_size=4)
    cache.put("k", 42)
    assert cache.get("k") == 42


def test_lru_evicts_oldest_when_full():
    cache = _LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # should evict "a"

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_get_promotes_to_most_recently_used():
    cache = _LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")            # promote "a"
    cache.put("c", 3)         # should evict "b" now, not "a"

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_lru_put_existing_key_updates_value_and_promotes():
    cache = _LRUCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 99)        # update + promote
    cache.put("c", 3)         # should evict "b"

    assert cache.get("a") == 99
    assert cache.get("b") is None
    assert cache.get("c") == 3
