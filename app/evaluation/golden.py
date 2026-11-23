"""Golden-set retrieval scoring with text anchors, so labels survive re-chunking and re-ingest."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from app.evaluation.metrics import ndcg_at_k


@dataclass
class Anchor:
    contains: str
    grade: int = 1


@dataclass
class GoldenCase:
    query: str
    anchors: list[Anchor]
    contract_id: Optional[int] = None
    split: str = "dev"
    reviewed: bool = False
    extra: dict = field(default_factory=dict)


def load_golden(path: Path, split: Optional[str] = None) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case = GoldenCase(
            query=raw["query"],
            anchors=[Anchor(a["contains"], int(a.get("grade", 1))) for a in raw["anchors"]],
            contract_id=raw.get("contract_id"),
            split=raw.get("split", "dev"),
            reviewed=bool(raw.get("reviewed", False)),
        )
        if split is None or case.split == split:
            cases.append(case)
    return cases


def score_retrieval(contents: Sequence[str], anchors: Sequence[Anchor], k: int) -> dict[str, float]:
    """Score ranked passage texts; an anchor is covered when a top-k passage contains it."""
    top = [content.casefold() for content in contents[:k]]
    grades = [
        max((a.grade for a in anchors if a.contains.casefold() in text), default=0) for text in top
    ]
    covered = [any(a.contains.casefold() in text for text in top) for a in anchors]

    relevance = {rank: grade for rank, grade in enumerate(grades) if grade > 0}
    # Uncovered anchors stay in the ideal ranking, so missing evidence lowers NDCG.
    relevance.update({-(i + 1): a.grade for i, a in enumerate(anchors) if not covered[i]})

    first_relevant = next((rank for rank, grade in enumerate(grades, 1) if grade > 0), None)
    return {
        "recall": sum(covered) / len(anchors) if anchors else 1.0,
        "precision": sum(1 for g in grades if g > 0) / len(grades) if grades else 0.0,
        "mrr": 1 / first_relevant if first_relevant else 0.0,
        "hit_rate": 1.0 if first_relevant else 0.0,
        "ndcg": ndcg_at_k(list(range(len(grades))), relevance, k),
    }


def mean_scores(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: round(sum(row[key] for row in rows) / len(rows), 4) for key in rows[0]}
