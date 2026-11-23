import pytest

from app.domain.entities.models import DocumentChunk
from app.infrastructure.search import reranker as reranker_module
from app.infrastructure.search.reranker import CrossEncoderReranker, LocalReranker, get_reranker

pytestmark = pytest.mark.unit


def _chunks(count: int) -> list[DocumentChunk]:
    return [DocumentChunk(id=i, contract_id=1, content=f"passage {i}") for i in range(count)]


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [float(i) for i in range(len(pairs))]  # last chunk scores highest


def test_default_is_local_reranker():
    assert isinstance(get_reranker(), LocalReranker)


def test_few_candidates_skip_the_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda name: model)
    result = CrossEncoderReranker("m", min_candidates=10).rerank("q", _chunks(5), 3)
    assert model.calls == 0
    assert len(result) == 3


def test_enough_candidates_use_the_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(reranker_module, "_load_cross_encoder", lambda name: model)
    result = CrossEncoderReranker("m", min_candidates=4).rerank("q", _chunks(6), 2)
    assert [chunk.id for chunk in result] == [5, 4]


def test_missing_dependency_falls_back_to_local(monkeypatch):
    def boom(name):
        raise ImportError("sentence_transformers")

    monkeypatch.setattr(reranker_module, "_load_cross_encoder", boom)
    reranker = CrossEncoderReranker("m", min_candidates=2)
    assert len(reranker.rerank("q", _chunks(4), 2)) == 2
    assert reranker._unavailable is True
