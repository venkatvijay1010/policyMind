"""Re-embed passages whose vectors came from a different model than EMBEDDING_MODEL.

Run after changing EMBEDDING_MODEL or EMBEDDING_DIMENSION. Run from the repository root with
PYTHONPATH set to it:

    python scripts/reembed_passages.py --dry-run
    python scripts/reembed_passages.py
    python scripts/reembed_passages.py --include-unrecorded --scope-key 3

Rows with no recorded model (indexed before it was tracked) are only touched with
--include-unrecorded. Each scope commits on its own and already-current rows are skipped, so
the script is safe to repeat. With VECTOR_BACKEND=qdrant the scope is re-synced afterwards.
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import or_, select

from app.config import settings
from app.infrastructure.database.models import ContractPassage
from app.infrastructure.database.postgres import async_session_factory, init_db
from app.infrastructure.llm.embeddings import get_embedding_service
from app.infrastructure.search.vector_store import (
    VectorStoreError,
    close_vector_store,
    get_vector_store,
)

BATCH_SIZE = 100


async def reembed(dry_run: bool, scope_key: Optional[int], include_unrecorded: bool) -> Counter:
    await init_db()
    model = settings.embedding_model
    stale = ContractPassage.embedding_model != model
    if include_unrecorded:
        stale = or_(stale, ContractPassage.embedding_model.is_(None))
    else:
        stale = ContractPassage.embedding_model.is_not(None) & stale

    stats: Counter = Counter()
    embedder = get_embedding_service()
    store = get_vector_store()
    async with async_session_factory() as session:
        scope_query = (
            select(ContractPassage.contract_id)
            .where(stale)
            .distinct()
            .order_by(ContractPassage.contract_id)
        )
        if scope_key is not None:
            scope_query = scope_query.where(ContractPassage.contract_id == scope_key)
        scope_ids = (await session.scalars(scope_query)).all()
        if not scope_ids:
            print(f"No passages need re-embedding for {model}")
        for scope_id in scope_ids:
            passages = (
                await session.scalars(
                    select(ContractPassage)
                    .where(ContractPassage.contract_id == scope_id, stale)
                    .order_by(ContractPassage.id)
                )
            ).all()
            stats["scopes"] += 1
            stats["passages"] += len(passages)
            print(f"  scope {scope_id}: {len(passages)} passages")
            if dry_run:
                continue
            try:
                for start in range(0, len(passages), BATCH_SIZE):
                    batch = passages[start : start + BATCH_SIZE]
                    vectors = await embedder.embed_batch([passage.content for passage in batch])
                    if len(vectors) != len(batch):
                        raise RuntimeError("Embedding service returned an incomplete response")
                    for passage, vector in zip(batch, vectors):
                        passage.embedding = vector
                        passage.embedding_model = model
                await session.commit()
            except Exception as exc:
                await session.rollback()
                stats["scopes_failed"] += 1
                print(f"  scope {scope_id}: failed ({exc})")
                continue
            if store is not None:
                rows = await session.execute(
                    select(ContractPassage.id, ContractPassage.embedding)
                    .where(ContractPassage.contract_id == scope_id)
                    .order_by(ContractPassage.id)
                )
                items = [(pid, vector) for pid, vector in rows if vector]
                try:
                    await store.replace_scope(scope_id, items)
                except VectorStoreError as exc:
                    stats["scopes_failed"] += 1
                    print(f"  scope {scope_id}: index sync failed ({exc.__cause__ or exc})")
    return stats


async def run(dry_run: bool, scope_key: Optional[int], include_unrecorded: bool) -> Counter:
    try:
        return await reembed(dry_run, scope_key, include_unrecorded)
    finally:
        await close_vector_store()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would change")
    parser.add_argument("--scope-key", type=int, help="Only re-embed this scope")
    parser.add_argument(
        "--include-unrecorded",
        action="store_true",
        help="Also re-embed rows with no recorded embedding model",
    )
    arguments = parser.parse_args()
    stats = asyncio.run(run(arguments.dry_run, arguments.scope_key, arguments.include_unrecorded))
    print("\nDry run, nothing written:" if arguments.dry_run else "\nDone:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    if stats["scopes_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
