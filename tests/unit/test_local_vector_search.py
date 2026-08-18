"""Tests for SQLite-compatible local semantic retrieval helpers."""

import pytest

from app.infrastructure.search.hybrid_search import HybridSearch

pytestmark = pytest.mark.unit


def test_embedding_values_accept_json_and_reject_invalid_values():
    assert HybridSearch._embedding_values("[1, 2.5, -3]") == [1.0, 2.5, -3.0]
    assert HybridSearch._embedding_values([1, 2.5]) == [1.0, 2.5]
    assert HybridSearch._embedding_values("not json") is None
    assert HybridSearch._embedding_values({"embedding": [1]}) is None


def test_cosine_similarity_ranks_matching_vectors_highest():
    assert HybridSearch._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert HybridSearch._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert HybridSearch._cosine_similarity([1.0], [1.0, 0.0]) is None
    assert HybridSearch._cosine_similarity([0.0], [0.0]) is None
