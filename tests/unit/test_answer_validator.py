import pytest

from app.domain.services.answer_validator import unsupported_numbers
from app.evaluation.metrics import ndcg_at_k

pytestmark = pytest.mark.unit


def test_numbers_present_in_evidence_are_supported():
    rows = [{"count": 1200, "total": "3,400.50"}]
    answer = "There were 1,200 cases totalling 3400.5. [Source 1]"
    assert unsupported_numbers(answer, [v for r in rows for v in r.values()]) == []


def test_invented_number_is_reported():
    assert unsupported_numbers("There were 42 cases.", [7]) == ["42"]


def test_list_markers_and_source_markers_are_ignored():
    assert unsupported_numbers("1. Alpha [Source 3]\n2. Beta", []) == []


def test_numbers_in_the_question_are_allowed():
    assert unsupported_numbers("Declined in 2026: 5", ["declined in 2026", 5]) == []


def test_ndcg_perfect_and_reversed_ranking():
    relevance = {1: 2, 2: 1}
    assert ndcg_at_k([1, 2], relevance) == 1.0
    assert 0 < ndcg_at_k([2, 1], relevance) < 1.0


def test_ndcg_hand_computed_value():
    # DCG = 1/log2(3) = 0.6309; IDCG = 1 -> 0.6309
    assert ndcg_at_k([9, 1], {1: 1}, k=2) == pytest.approx(0.6309, abs=1e-4)


def test_ndcg_without_relevant_documents_is_zero():
    assert ndcg_at_k([1, 2], {}, k=2) == 0.0
