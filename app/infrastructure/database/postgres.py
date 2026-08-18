"""Database connection and async session management.

The module name remains ``postgres`` for import compatibility, but the default
development database is SQLite. PostgreSQL can still be selected through
``DATABASE_URL`` when a deployment needs it.
"""

from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import MetaData, event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    metadata = metadata


database_backend = make_url(settings.database_url).get_backend_name()
_engine_options: dict[str, object] = {
    "echo": settings.debug,
    "pool_pre_ping": True,
}
if database_backend != "sqlite":
    _engine_options.update(pool_size=5, max_overflow=10)

# Create async engine. SQLite's aiosqlite driver is used by default so FastAPI
# can retain the existing AsyncSession-based request handling.
engine = create_async_engine(settings.database_url, **_engine_options)

if database_backend == "sqlite":

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
        """SQLite disables foreign-key enforcement unless it is enabled per connection."""
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Create the configured database and all application tables if needed."""
    if database_backend == "sqlite":
        database_path = make_url(settings.database_url).database
        if database_path and database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    # Import registers every ORM model with Base.metadata before create_all runs.
    from app.infrastructure.database import models  # noqa: F401

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Alias for backward compatibility
get_db_session = get_session
