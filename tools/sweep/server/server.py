#!/usr/bin/env python3
"""Minimal REST server for RL experiment orchestration.
"""

from __future__ import annotations

import http.server
import json
import os
import sqlite3
import urllib.parse
from datetime import datetime
from typing import Callable, TypeVar

DB_FILE = os.environ.get("JOB_SERVER_DB", "jobs.db")

# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

def init_db(path: str = DB_FILE) -> None:
    """Create tables if they do not exist."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
                                            id   INTEGER PRIMARY KEY,
                                            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks (
                                             id      INTEGER PRIMARY KEY,
                                             job_id  INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
            spec    TEXT    NOT NULL,
            status  TEXT    NOT NULL DEFAULT 'pending',
            result  TEXT,
            updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """
    )
    conn.commit()
    conn.close()


F = TypeVar("F", bound=Callable[..., object])

def with_connection(fn: F) -> F:  # type: ignore[override]
    """Decorator that injects a temporary SQLite connection **after** `self`."""

    def inner(self, *args, **kwargs):  # type: ignore[override]
        conn = sqlite3.connect(DB_FILE, timeout=30)
        try:
            return fn(self, conn, *args, **kwargs)
        finally:
            conn.close()

    return inner  # type: ignore[return-value]


class RequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "RLJobServer/0.2"

    def _send(self, status: int, body=None):
        self.send_response(status)
        if body is not None:
            payload = json.dumps(body).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.end_headers()

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length).decode() if length else ""

    def do_POST(self):  # noqa: N802
        """Handle all POST endpoints."""
        path_parts = urllib.parse.urlparse(self.path).path.strip("/").split("/")

        if not path_parts or path_parts[0] != "jobs":
            return self._send(404, {"error": "Unknown endpoint"})

        try:
            if len(path_parts) == 2:
                return self._handle_create_job(path_parts[1])
            if len(path_parts) == 3 and path_parts[2] == "tasks":
                return self._handle_take_task(path_parts[1])
            if len(path_parts) == 4 and path_parts[2] == "tasks":
                return self._handle_report_task(path_parts[1], int(path_parts[3]))
        except ValueError:
            return self._send(400, {"error": "Malformed URL"})

        return self._send(404, {"error": "Unknown endpoint"})

    @with_connection
    def _handle_create_job(self, conn: sqlite3.Connection, job_name: str):
        body = self._read_body()
        tasks = [line for line in body.splitlines() if line.strip()]
        if not tasks:
            return self._send(400, {"error": "No tasks provided"})

        cur = conn.cursor()
        cur.execute("INSERT OR IGNORE INTO jobs(name) VALUES (?)", (job_name,))
        cur.execute("SELECT id FROM jobs WHERE name = ?", (job_name,))
        job_id = cur.fetchone()[0]
        cur.executemany(
            "INSERT INTO tasks(job_id, spec, status) VALUES (?, ?, 'pending')",
            ((job_id, t) for t in tasks),
        )
        conn.commit()
        return self._send(201, {"created_tasks": len(tasks)})

    @with_connection
    def _handle_take_task(self, conn: sqlite3.Connection, job_name: str):
        conn.isolation_level = None  # manual transaction control
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")  # lock DB for atomic pop
        cur.execute("SELECT id FROM jobs WHERE name = ?", (job_name,))
        row = cur.fetchone()
        if not row:
            conn.execute("ROLLBACK")
            return self._send(404, {"error": "Job not found"})
        job_id = row[0]
        cur.execute(
            "SELECT id, spec FROM tasks WHERE job_id = ? AND status = 'pending' LIMIT 1",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            conn.execute("COMMIT")
            return self._send(204)  # no pending tasks
        task_id, spec_text = row
        cur.execute(
            "UPDATE tasks SET status = 'in_progress', updated = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), task_id),
        )
        conn.execute("COMMIT")
        return self._send(200, {"task_id": task_id, "spec": json.loads(spec_text)})

    @with_connection
    def _handle_report_task(self, conn: sqlite3.Connection, job_name: str, task_id: int):
        body = self._read_body() or "{}"
        try:
            result_json = json.loads(body)
        except json.JSONDecodeError:
            return self._send(400, {"error": "Result is not valid JSON"})

        cur = conn.cursor()
        cur.execute(
            """
            UPDATE tasks
            SET status = 'done', result = ?, updated = ?
            WHERE id = ?
              AND status = 'in_progress'
            """,
            (json.dumps(result_json), datetime.utcnow().isoformat(), task_id),
        )
        if cur.rowcount == 0:
            return self._send(404, {"error": "Task not found or not in progress"})
        conn.commit()
        return self._send(200, {"ok": True})

    def log_message(self, fmt, *args):  # noqa: D401, N802
        return


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n🚀 RL Job Server listening on http://0.0.0.0:{port}\n")
    http.server.ThreadingHTTPServer(("", port), RequestHandler).serve_forever()




