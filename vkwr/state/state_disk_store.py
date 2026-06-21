"""L3 disk persistence for request state via SQLite BLOB storage.

Stores state tensors (pickle serialized) keyed by (req_id, token_pos).
Writes are async (background thread) to avoid blocking the main scheduling loop.
"""

from __future__ import annotations

import logging
import os
import pickle
import queue
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)


class StateDiskStore:
    """L3 disk persistence for request state via SQLite BLOB storage.

    Stores state tensors (pickle serialized) keyed by (req_id, token_pos).
    Writes are async (background thread) to avoid blocking the main scheduling loop.
    """

    def __init__(self, db_path: str, max_size_mb: int = 10240):
        self.db_path = db_path
        self.max_size_bytes = max_size_mb * 1024 * 1024

        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._write_queue: queue.Queue[tuple] = queue.Queue()

        self._pending_lock = threading.Lock()
        self._pending_events: dict[str, threading.Event] = {}

        self._closed = False

        self._write_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._write_conn.execute("PRAGMA journal_mode=WAL")
        self._write_conn.execute("PRAGMA synchronous=NORMAL")
        self._create_table(self._write_conn)

        self._reader_conn = sqlite3.connect(db_path, check_same_thread=False)
        self._reader_conn.execute("PRAGMA journal_mode=WAL")

        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _create_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state_entries (
                req_id TEXT NOT NULL,
                token_pos INTEGER NOT NULL,
                data BLOB NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (req_id, token_pos)
            )
        """
        )
        conn.commit()

    def _mark_pending(self, req_id: str) -> None:
        with self._pending_lock:
            if req_id not in self._pending_events:
                self._pending_events[req_id] = threading.Event()
            self._pending_events[req_id].clear()

    def _mark_done(self, req_id: str) -> None:
        with self._pending_lock:
            evt = self._pending_events.pop(req_id, None)
            if evt:
                evt.set()

    def _wait_pending(self, req_id: str) -> None:
        while True:
            with self._pending_lock:
                evt = self._pending_events.get(req_id)
            if evt is None:
                break
            evt.wait(timeout=0.1)

    def save(self, req_id: str, state: list, token_pos: int) -> None:
        """Enqueue a save operation. Actual write happens on background thread."""
        if self._closed:
            return
        pickled_data = pickle.dumps(state)
        self._mark_pending(req_id)
        self._write_queue.put(("save", req_id, token_pos, pickled_data, time.monotonic()))

    def restore(self, req_id: str) -> list | None:
        """Restore the latest state for req_id. Blocks until any pending write for this req_id completes."""
        self._wait_pending(req_id)

        try:
            cur = self._reader_conn.execute(
                "SELECT data FROM state_entries WHERE req_id = ? ORDER BY token_pos DESC LIMIT 1",
                (req_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return pickle.loads(row[0])
        except Exception:
            logger.exception("Failed to restore state for req_id %s", req_id)
            return None

    def clear(self, req_id: str) -> None:
        """Remove all state entries for a request."""
        if self._closed:
            return
        self._wait_pending(req_id)
        self._mark_pending(req_id)
        self._write_queue.put(("delete", req_id))

    def flush(self) -> None:
        """Wait for all pending writes to complete."""
        done = threading.Event()
        self._write_queue.put(("flush", done))
        done.wait(timeout=30)

    def close(self) -> None:
        """Flush and close the background writer."""
        self._closed = True
        self._write_queue.put(("shutdown",))
        self._writer_thread.join(timeout=10)
        self._write_conn.close()
        self._reader_conn.close()

    def _writer_loop(self) -> None:
        while True:
            try:
                op = self._write_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            op_type = op[0]

            if op_type == "shutdown":
                self._flush_remaining()
                return

            if op_type == "save":
                _op_type, req_id, token_pos, data, created_at = op
                self._save_entry(req_id, token_pos, data, created_at)
                self._mark_done(req_id)

            elif op_type == "delete":
                _op_type, req_id = op
                self._delete_entries(req_id)
                self._mark_done(req_id)

            elif op_type == "flush":
                flush_event = op[1]
                flush_event.set()

    def _flush_remaining(self) -> None:
        while not self._write_queue.empty():
            try:
                op = self._write_queue.get_nowait()
                op_type = op[0]
                if op_type == "save":
                    _op_type, req_id, token_pos, data, created_at = op
                    self._save_entry(req_id, token_pos, data, created_at)
                    self._mark_done(req_id)
                elif op_type == "delete":
                    _op_type, req_id = op
                    self._delete_entries(req_id)
                    self._mark_done(req_id)
            except queue.Empty:
                break

    def _save_entry(self, req_id: str, token_pos: int, data: bytes, created_at: float) -> None:
        self._enforce_max_size()
        try:
            self._write_conn.execute(
                """
                INSERT OR REPLACE INTO state_entries (req_id, token_pos, data, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (req_id, token_pos, data, created_at),
            )
            self._write_conn.commit()
        except Exception:
            logger.exception("Failed to save state for req_id %s at pos %d", req_id, token_pos)

    def _delete_entries(self, req_id: str) -> None:
        try:
            self._write_conn.execute("DELETE FROM state_entries WHERE req_id = ?", (req_id,))
            self._write_conn.commit()
        except Exception:
            logger.exception("Failed to delete state for req_id %s", req_id)

    def _get_db_size(self, conn: sqlite3.Connection) -> int:
        try:
            pc = conn.execute("PRAGMA page_count").fetchone()[0]
            ps = conn.execute("PRAGMA page_size").fetchone()[0]
            return pc * ps
        except Exception:
            return 0

    def _enforce_max_size(self) -> None:
        current_size = self._get_db_size(self._write_conn)
        if current_size < self.max_size_bytes:
            return

        try:
            oldest_rows = self._write_conn.execute("SELECT req_id FROM state_entries ORDER BY created_at ASC LIMIT 10").fetchall()
            for (req_id,) in oldest_rows:
                self._write_conn.execute("DELETE FROM state_entries WHERE req_id = ?", (req_id,))
            self._write_conn.commit()
            current_size = self._get_db_size(self._write_conn)
            logger.warning(
                "Disk store evicted %d entries, DB size: %d bytes, limit: %d bytes",
                len(oldest_rows),
                current_size,
                self.max_size_bytes,
            )
        except Exception:
            logger.exception("Failed to evict oldest entries from disk store")
