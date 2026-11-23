import json

import pytest

from app.evaluation.golden import Anchor, load_golden, mean_scores, score_retrieval

pytestmark = pytest.mark.unit

ANCHORS = [Anchor("48 months", 2), Anchor("pre-existing", 1)]


def test_perfect_retrieval_scores_one():
    scores = score_retrieval(["Pre-existing: waiting period is 48 months", "other"], ANCHORS, 5)
    assert scores["recall"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["hit_rate"] == 1.0
    assert scores["ndcg"] == 1.0


def test_missing_evidence_lowers_recall_and_ndcg():
    scores = score_retrieval(["unrelated", "mentions pre-existing only"], ANCHORS, 5)
    assert scores["recall"] == 0.5
    assert scores["mrr"] == 0.5
    assert scores["precision"] == 0.5
    assert 0 < scores["ndcg"] < 1


def test_no_relevant_passage_scores_zero():
    scores = score_retrieval(["nothing", "useful"], ANCHORS, 5)
    assert scores == {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "hit_rate": 0.0, "ndcg": 0.0}


def test_only_top_k_passages_count():
    scores = score_retrieval(["nothing", "48 months pre-existing"], ANCHORS, 1)
    assert scores["recall"] == 0.0


def test_load_golden_filters_by_split(tmp_path):
    path = tmp_path / "g.jsonl"
    rows = [
        {"query": "a", "anchors": [{"contains": "x"}], "split": "dev"},
        {"query": "b", "anchors": [{"contains": "y", "grade": 2}], "split": "test"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert [c.query for c in load_golden(path, "test")] == ["b"]
    assert len(load_golden(path)) == 2


def test_mean_scores():
    assert mean_scores([{"a": 1.0}, {"a": 0.0}]) == {"a": 0.5}
    assert mean_scores([]) == {}
