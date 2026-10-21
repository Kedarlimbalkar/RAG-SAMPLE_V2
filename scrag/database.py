"""
PostgreSQL Connection Pool and Transaction Manager with pgvector support.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from scrag.config import Settings, settings as global_settings

logger = logging.getLogger(__name__)

# Conditional import of psycopg2 to allow module loading in pure parsing environments
try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    pool = None  # type: ignore


class DatabaseManager:
    """
    Thread-safe PostgreSQL connection pool manager supporting transactional
    operations and pgvector extension registrations.
    """

    def __init__(self, config: Optional[Settings] = None) -> None:
        self.cfg = config or global_settings
        self._pool: Optional[Any] = None

    def initialize_pool(self, minconn: int = 1, maxconn: Optional[int] = None) -> None:
        """Initialize the ThreadedConnectionPool."""
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "psycopg2 is required for DatabaseManager. "
                "Install with `pip install psycopg2-binary`."
            )

        max_connections = maxconn or self.cfg.postgres_max_connections
        logger.info(
            f"Initializing PostgreSQL connection pool ({minconn}-{max_connections} conns) "
            f"to {self.cfg.postgres_host}:{self.cfg.postgres_port}/{self.cfg.postgres_db}"
        )

        self._pool = pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=max_connections,
            host=self.cfg.postgres_host,
            port=self.cfg.postgres_port,
            dbname=self.cfg.postgres_db,
            user=self.cfg.postgres_user,
            password=self.cfg.postgres_password,
            connect_timeout=self.cfg.postgres_pool_timeout,
        )

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("PostgreSQL connection pool closed.")

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Context manager for acquiring a connection from the pool.
        Automatically returns connection to pool upon completion.
        """
        if self._pool is None:
            self.initialize_pool()

        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def get_cursor(
        self, commit: bool = True, cursor_factory: Any = None
    ) -> Generator[Any, None, None]:
        """
        Transactional cursor context manager.
        Commits upon exit if commit=True, rolls back on exception.
        """
        factory = cursor_factory or (RealDictCursor if PSYCOPG2_AVAILABLE else None)
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=factory)
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    def run_schema_migration(self, schema_file: Optional[Path] = None) -> None:
        """Execute DDL statements from schema.sql."""
        path = schema_file or Path("schema.sql")
        if not path.is_file():
            raise FileNotFoundError(f"Schema file not found at: {path}")

        sql_script = path.read_text(encoding="utf-8")
        with self.get_cursor(commit=True) as cursor:
            logger.info(f"Applying schema definitions from {path}...")
            cursor.execute(sql_script)
            logger.info("Database schema initialized successfully.")


# Global database manager instance
db = DatabaseManager()

