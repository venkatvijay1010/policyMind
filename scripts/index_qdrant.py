"""Re-sync the Qdrant collection from the passage embeddings stored in the database.

Run once after enabling VECTOR_BACKEND=qdrant (new uploads are indexed automatically), and
again if an upload failed after its vector step. Run from the repository root with
PYTHONPATH set to it:

    python scripts/index_qdrant.py --dry-run
    python scripts/index_qdrant.py
    python scripts/index_qdrant.py --scope-key 3

Each scope is replaced as a whole, so the script is safe to repeat.
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from app.config import settings
from app.infrastructure.database.models import ContractPassage
from app.infrastructure.database.postgres import async_session_factory, init_db
from app.infrastructure.search.hybrid_search import HybridSearch
from app.infrastructure.search.vector_store import (
    VectorStoreError,
    close_vector_store,
    get_vector_store,
)


async def index_scopes(dry_run: bool, scope_key: Optional[int]) -> Counter:
    store = get_vector_store()
    if store is None:
        raise SystemExit("VECTOR_BACKEND is not 'qdrant'; set it (and QDRANT_URL) first.")
    await init_db()
    stats: Counter = Counter()
    async with async_session_factory() as session:
        scope_query = (
            select(ContractPassage.contract_id).distinct().order_by(ContractPassage.contract_id)
        )
        if scope_key is not None:
            scope_query = scope_query.where(ContractPassage.contract_id == scope_key)
        scope_ids = (await session.scalars(scope_query)).all()
        if not scope_ids:
            print("No scopes with passages found")
        for scope_id in scope_ids:
            rows = await session.execute(
                select(ContractPassage.id, ContractPassage.embedding)
                .where(ContractPassage.contract_id == scope_id)
                .order_by(ContractPassage.id)
            )
            items: list[tuple[int, list[float]]] = []
            skipped = 0
            for passage_id, embedding in rows:
                values = HybridSearch._embedding_values(embedding)
                if values:
                    items.append((passage_id, values))
                else:
                    skipped += 1
            stats["scopes"] += 1
            stats["passages_indexed"] += len(items)
            stats["passages_skipped_malformed"] += skipped
            print(f"  scope {scope_id}: {len(items)} indexed, {skipped} skipped")
            if dry_run:
                continue
            try:
                await store.replace_scope(scope_id, items)
            except VectorStoreError as exc:
                stats["scopes_failed"] += 1
                print(f"  scope {scope_id}: failed ({exc.__cause__ or exc})")
    return stats


async def run(dry_run: bool, scope_key: Optional[int]) -> Counter:
    try:
        return await index_scopes(dry_run, scope_key)
    finally:
        await close_vector_store()


def main() -> None:
    if settings.vector_backend != "qdrant":
        raise SystemExit("VECTOR_BACKEND is not 'qdrant'; set it (and QDRANT_URL) first.")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would be indexed")
    parser.add_argument("--scope-key", type=int, help="Only re-index this scope")
    arguments = parser.parse_args()
    stats = asyncio.run(run(arguments.dry_run, arguments.scope_key))
    print("\nDry run, nothing written:" if arguments.dry_run else "\nDone:")
    for name, value in sorted(stats.items()):
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
