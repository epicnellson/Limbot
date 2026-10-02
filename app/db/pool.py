from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

# Every pooled connection is opened read only, so a malformed or injected query physically
# cannot write. The bot only ever asks questions about a student; it never changes a record.
SERVER_SETTINGS = {
    "default_transaction_read_only": "on",
    "application_name": "limbot",
    "statement_timeout": "5000",
}


class Database:
    """Asyncpg pool with a read-only guarantee, plus a probe that never raises.

    ``asyncpg`` is imported lazily so the process can start, serve webhooks and answer health
    checks when the driver is missing or Postgres is not up yet.
    """

    def __init__(self, settings: Settings, pool: Any = None) -> None:
        self._settings = settings
        self._pool = pool

    @property
    def configured(self) -> bool:
        return bool(self._settings.postgres_dsn_value)

    @property
    def connected(self) -> bool:
        return self._pool is not None

    @property
    def pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("database is not connected")
        return self._pool

    async def connect(self) -> None:
        if self._pool is not None:
            return
        if not self.configured:
            logger.info("POSTGRES_DSN is not set, student tools stay disabled")
            return

        import asyncpg

        self._pool = await asyncpg.create_pool(
            dsn=self._settings.postgres_dsn_value,
            min_size=self._settings.postgres_pool_min_size,
            max_size=self._settings.postgres_pool_max_size,
            command_timeout=self._settings.postgres_command_timeout_seconds,
            server_settings=SERVER_SETTINGS,
        )
        logger.info(
            "postgres pool created",
            extra={
                "context": {
                    "min_size": self._settings.postgres_pool_min_size,
                    "max_size": self._settings.postgres_pool_max_size,
                    "read_only": True,
                }
            },
        )

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        async with self.pool.acquire() as connection:
            return list(await connection.fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> Any:
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        async with self.pool.acquire() as connection:
            return await connection.fetchval(query, *args)

    async def ping(self) -> tuple[bool, float, str | None]:
        """Return (reachable, latency_ms, error) without letting probe errors escape."""
        if self._pool is None:
            return False, 0.0, "database is not connected"
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            async with asyncio.timeout(self._settings.postgres_command_timeout_seconds * 2):
                await self._pool.fetchval("SELECT 1")
        except Exception as exc:
            return False, (loop.time() - started) * 1000, f"{type(exc).__name__}: {exc}"
        return True, (loop.time() - started) * 1000, None

    def describe(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "connected": self.connected,
            "read_only": True,
            "pool_size": self._settings.postgres_pool_max_size,
        }
