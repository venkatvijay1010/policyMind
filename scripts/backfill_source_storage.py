"""Give every knowledge scope a stored, openable source document.

Run once after upgrading, and again after enabling STORAGE_BACKEND=s3:

    python scripts/backfill_source_storage.py --dry-run
    python scripts/backfill_source_storage.py

Steps (each is safe to repeat):
1. Scopes that were indexed without a source file get one rebuilt from their passages.
2. PDF elements recorded before text offsets existed get their offsets, so evidence
   highlights the exact paragraph on the page.
3. With S3 enabled, every source document not yet in the bucket is uploaded and its
   bucket/key/ETag are saved on the row.
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.application.source_documents import (
    attach_text_source,
    locate_texts,
    reconstruct_text,
    store_original,
)
from app.infrastructure.database.models import (
    ContractPassage,
    DocumentElement,
    PassageSource,
    Policy,
    SourceDocument,
)
from app.infrastructure.database.postgres import async_session_factory, init_db
from app.infrastructure.storage.object_storage import get_object_storage


async def _ordered_passages(session: AsyncSession, statement) -> list[ContractPassage]:
    result = await session.execute(
        statement.order_by(ContractPassage.passage_order, ContractPassage.id).options(
            defer(ContractPassage.embedding)
        )
    )
    return list(result.scalars().all())


async def _attach_missing_sources(session: AsyncSession, dry_run: bool, stats: Counter) -> None:
    scope_ids = (
        await session.scalars(
            select(Policy.id).where(
                exists().where(ContractPassage.contract_id == Policy.id),
                ~exists().where(SourceDocument.contract_id == Policy.id),
            )
        )
    ).all()
    for scope_id in scope_ids:
        passages = await _ordered_passages(
            session, select(ContractPassage).where(ContractPassage.contract_id == scope_id)
        )
        rebuilt = reconstruct_text([(p.content, p.source_offset_start or 0) for p in passages])
        if rebuilt is None:
            stats["scopes_unrecoverable"] += 1
            print(f"  scope {scope_id}: passages overlap inconsistently; skipped")
            continue
        stats["scopes_attached"] += 1
        if dry_run:
            continue
        text_content, starts = rebuilt
        for passage, start in zip(passages, starts):
            passage.source_offset_start = start
            passage.source_offset_end = start + len(passage.content)
        title = await session.scalar(select(Policy.contract_title).where(Policy.id == scope_id))
        await attach_text_source(
            session,
            scope_id,
            text_content,
            f"{title or f'scope-{scope_id}'}.txt",
            [passage.id for passage in passages],
            origin="reconstructed_from_passages",
        )
        await session.commit()


async def _fill_element_offsets(session: AsyncSession, dry_run: bool, stats: Counter) -> None:
    document_ids = (
        await session.scalars(
            select(DocumentElement.source_document_id)
            .where(DocumentElement.text_offset_start.is_(None))
            .distinct()
        )
    ).all()
    for document_id in document_ids:
        passages = await _ordered_passages(
            session,
            select(ContractPassage)
            .join(PassageSource, PassageSource.passage_id == ContractPassage.id)
            .where(PassageSource.source_document_id == document_id)
            .distinct(),
        )
        rebuilt = reconstruct_text([(p.content, p.source_offset_start or 0) for p in passages])
        if rebuilt is None:
            stats["documents_without_offsets"] += 1
            continue
        elements = (
            await session.scalars(
                select(DocumentElement)
                .where(DocumentElement.source_document_id == document_id)
                .order_by(DocumentElement.id)
            )
        ).all()
        spans = locate_texts(rebuilt[0], [element.text_content for element in elements])
        located = 0
        for element, span in zip(elements, spans):
            if span is None:
                continue
            located += 1
            if not dry_run:
                element.text_offset_start, element.text_offset_end = span
        stats["elements_located"] += located
        stats["elements_unlocated"] += len(elements) - located
        if not dry_run:
            await session.commit()


async def _upload_missing_objects(session: AsyncSession, dry_run: bool, stats: Counter) -> None:
    storage = get_object_storage()
    if storage is None:
        print("  STORAGE_BACKEND is not s3; skipping uploads")
        return
    document_ids = (
        await session.scalars(select(SourceDocument.id).where(SourceDocument.storage_key.is_(None)))
    ).all()
    for document_id in document_ids:
        stats["objects_to_upload"] += 1
        if dry_run:
            continue
        source = await session.get(SourceDocument, document_id)
        await store_original(source, storage)
        await session.commit()
        stats["objects_uploaded"] += 1


async def backfill(dry_run: bool) -> Counter:
    # Adds any missing columns so older databases can be read and updated.
    await init_db()
    stats: Counter = Counter()
    async with async_session_factory() as session:
        print("1. Attaching source documents to scopes that have none")
        await _attach_missing_sources(session, dry_run, stats)
        print("2. Locating PDF elements in the passage text")
        await _fill_element_offsets(session, dry_run, stats)
        print("3. Uploading source documents to object storage")
        await _upload_missing_objects(session, dry_run, stats)
        await session.rollback()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Report what would change")
    arguments = parser.parse_args()
    stats = asyncio.run(backfill(arguments.dry_run))
    print("\nDry run, nothing written:" if arguments.dry_run else "\nDone:")
    for name, value in sorted(stats.items()):
        print(f"  {name}: {value}")
    if not stats:
        print("  nothing to do")


if __name__ == "__main__":
    main()
