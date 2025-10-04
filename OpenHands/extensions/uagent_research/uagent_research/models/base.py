"""Base database models and session management"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import MetaData
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)

# SQLAlchemy metadata
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Global session factory
_async_session_factory = None
_engine = None


async def init_database(database_url: str, echo: bool = False):
    """
    Initialize database connection and create tables.

    Args:
        database_url: Database connection URL (e.g., postgresql+asyncpg://user:pass@host/db)
        echo: Whether to echo SQL statements (for debugging)
    """
    global _async_session_factory, _engine

    logger.info(f"Initializing database: {database_url}")

    # Create async engine
    # SQLite doesn't support pool_size and max_overflow
    if database_url.startswith("sqlite"):
        _engine = create_async_engine(
            database_url,
            echo=echo,
        )
    else:
        _engine = create_async_engine(
            database_url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )

    # Create session factory
    _async_session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create all tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized successfully")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session for dependency injection.

    Usage:
        async with get_session() as session:
            result = await session.execute(...)

    Or with FastAPI:
        @app.get("/endpoint")
        async def endpoint(session: AsyncSession = Depends(get_session)):
            ...
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_database():
    """Close database connections"""
    global _engine

    if _engine:
        await _engine.dispose()
        logger.info("Database connections closed")
