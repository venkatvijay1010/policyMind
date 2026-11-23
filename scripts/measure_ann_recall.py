"""Measure Qdrant HNSW recall against exact search for several hnsw_ef values.

    PYTHONPATH=. python scripts/measure_ann_recall.py --queries 100 --limit 10

Queries are stored vectors, so the exact top hit is the vector itself; recall is the overlap of
the approximate top-k with the exact top-k. Small corpora return 1.0 at any ef; the numbers
become informative as the collection grows.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.infrastructure.search.vector_store import close_vector_store, get_vector_store


async def run(queries: int, limit: int, ef_values: list[int]) -> None:
    store = get_vector_store()
    if store is None:
        raise SystemExit("VECTOR_BACKEND is not 'qdrant'; set it (and QDRANT_URL) first.")
    vectors = await store.sample_vectors(queries)
    if not vectors:
        raise SystemExit("No vectors in the collection.")
    exact = [
        {hit.passage_id for hit in await store.search(v, None, limit, exact=True)} for v in vectors
    ]
    print(f"{len(vectors)} queries, top-{limit}")
    print("hnsw_ef  recall  avg_ms")
    for ef in ef_values:
        overlap = 0.0
        started = perf_counter()
        for vector, truth in zip(vectors, exact):
            found = {hit.passage_id for hit in await store.search(vector, None, limit, hnsw_ef=ef)}
            overlap += len(found & truth) / max(len(truth), 1)
        elapsed_ms = (perf_counter() - started) * 1000 / len(vectors)
        print(f"{ef:7d}  {overlap / len(vectors):.4f}  {elapsed_ms:.1f}")
    await close_vector_store()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = parser.parse_args()
    asyncio.run(run(args.queries, args.limit, args.ef))
