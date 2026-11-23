"""Score retrieval against a golden set and optionally gate on a stored baseline.

    PYTHONPATH=. python scripts/eval_retrieval.py --split dev --k 5 --out reports/retrieval.json
    PYTHONPATH=. python scripts/eval_retrieval.py --split test --baseline reports/baseline.json

Compare configurations by setting env vars per run (RERANKER, CONTEXT_NEIGHBOR_PASSAGES,
TOP_K_RETRIEVAL, ...). Exit code is 1 when NDCG or recall drops more than --tolerance
below the baseline.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.application.agents.rag_agent import RAGAgent
from app.evaluation.golden import load_golden, mean_scores, score_retrieval
from app.evaluation.metrics import compute_latency_metrics
from app.infrastructure.database.postgres import async_session_factory, init_db

GATED_METRICS = ("ndcg", "recall")


async def evaluate(golden_path: Path, split: Optional[str], k: int) -> dict:
    cases = load_golden(golden_path, split)
    if not cases:
        raise SystemExit("No golden cases matched.")
    await init_db()
    rows: list[dict[str, float]] = []
    latencies: list[int] = []
    async with async_session_factory() as session:
        agent = RAGAgent(session)
        for case in cases:
            started = perf_counter()
            result = await agent.retrieve(case.query, contract_id=case.contract_id, top_k=k)
            latencies.append(int((perf_counter() - started) * 1000))
            rows.append(score_retrieval([c.content for c in result.chunks], case.anchors, k))
    unreviewed = sum(1 for case in cases if not case.reviewed)
    return {
        "k": k,
        "split": split,
        "cases": len(cases),
        "unreviewed_cases": unreviewed,
        "metrics": mean_scores(rows),
        "latency_ms": compute_latency_metrics(latencies),
    }


def gate(report: dict, baseline_path: Path, tolerance: float) -> list[str]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
    return [
        f"{name}: {report['metrics'][name]:.4f} < baseline {baseline[name]:.4f} - {tolerance}"
        for name in GATED_METRICS
        if report["metrics"][name] < baseline[name] - tolerance
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_v1.jsonl"))
    parser.add_argument("--split", choices=["dev", "test"], default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    report = asyncio.run(evaluate(args.golden, args.split, args.k))
    print(json.dumps(report, indent=2))
    if report["unreviewed_cases"]:
        print(f"WARNING: {report['unreviewed_cases']} case(s) not human-reviewed", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.baseline:
        failures = gate(report, args.baseline, args.tolerance)
        for failure in failures:
            print(f"REGRESSION {failure}", file=sys.stderr)
        sys.exit(1 if failures else 0)
