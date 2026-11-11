from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time


class QueueFull(Exception):
    pass


class JobStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "jobs.sqlite3"
        with self.connection() as database:
            database.execute("PRAGMA journal_mode=WAL")
            database.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    options TEXT NOT NULL, created_at REAL NOT NULL,
                    started_at REAL, finished_at REAL, updated_at REAL NOT NULL,
                    stage TEXT, error TEXT, summary TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
            """)

    @contextmanager
    def connection(self):
        database = sqlite3.connect(self.path, timeout=15)
        database.row_factory = sqlite3.Row
        try:
            with database:
                yield database
        finally:
            database.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        job = dict(row)
        job["options"] = json.loads(job["options"])
        job["summary"] = json.loads(job["summary"]) if job["summary"] else None
        return job

    def submit(self, job_id, name, options, max_pending=20):
        now = time.time()
        with self.connection() as database:
            database.execute("BEGIN IMMEDIATE")
            pending = database.execute("SELECT count(*) FROM jobs WHERE status IN ('queued', 'running')").fetchone()[0]
            if pending >= max_pending:
                raise QueueFull("The generation queue is full")
            database.execute("INSERT INTO jobs(id,name,status,options,created_at,updated_at) VALUES(?,?,'queued',?,?,?)",
                             (job_id, name, json.dumps(options), now, now))
        return self.get(job_id)

    def get(self, job_id):
        with self.connection() as database:
            return self.decode(database.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self, limit=50, offset=0):
        with self.connection() as database:
            return [self.decode(row) for row in database.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?", (limit, offset))]

    def counts(self):
        with self.connection() as database:
            return dict(database.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())

    def claim(self):
        with self.connection() as database:
            database.execute("BEGIN IMMEDIATE")
            if database.execute("SELECT 1 FROM jobs WHERE status='running' LIMIT 1").fetchone():
                return None
            row = database.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            database.execute("UPDATE jobs SET status='running',stage='initializing',started_at=?,updated_at=? WHERE id=?",
                             (now, now, row["id"]))
        return self.get(row["id"])

    def stage(self, job_id, stage):
        with self.connection() as database:
            database.execute("UPDATE jobs SET stage=?,updated_at=? WHERE id=? AND status='running'",
                             (stage, time.time(), job_id))

    def finish(self, job_id, summary=None, error=None):
        with self.connection() as database:
            now = time.time()
            database.execute("UPDATE jobs SET status=?,stage=?,finished_at=?,updated_at=?,summary=?,error=? "
                             "WHERE id=? AND status='running'",
                             ("failed" if error else "complete", "failed" if error else "complete", now, now,
                              json.dumps(summary) if summary else None, error, job_id))

    def cancel(self, job_id):
        with self.connection() as database:
            now = time.time()
            return database.execute("UPDATE jobs SET status='cancelled',stage='cancelled',finished_at=?,updated_at=? "
                                    "WHERE id=? AND status='queued'", (now, now, job_id)).rowcount == 1

    def interrupt_running(self, reason):
        with self.connection() as database:
            now = time.time()
            database.execute("UPDATE jobs SET status='failed',stage='interrupted',finished_at=?,updated_at=?,error=? "
                             "WHERE status='running'", (now, now, reason))
