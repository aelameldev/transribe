import asyncpg
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/transcribeDB")

_pool: asyncpg.Pool | None = None

async def init_pool(min_size: int = 2, max_size: int = 10):
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=min_size,
            max_size=max_size,
        )
    return _pool

async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Pool not initialized. Call init_pool() first.")
    return _pool

async def close_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
