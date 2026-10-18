from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

from repopilot.observability.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(StrEnum):
    """
    All valid states a task can be in.

    State machine:
        PENDING ──(claim)──► IN_PROGRESS ──(success)──► COMPLETED  (terminal)
                                    │
                                    ├──(failure, retries left)──► PENDING
                                    └──(failure, no retries)────► FAILED
        PENDING ──(dep failed)──► BLOCKED
        FAILED / BLOCKED / PENDING ──(approved re-plan)──► REPLACED  (terminal)

    Dropping a task is done through an approved re-plan (it becomes REPLACED),
    so the reason is always recorded and its dependents are re-planned too.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    REPLACED = "replaced"  # superseded by tasks from an approved re-plan


_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    goal        TEXT NOT NULL,
    plan_json   TEXT,                   -- the plan exactly as the user approved it (audit record)
    status      TEXT DEFAULT 'planning',-- approved → completed | failed | abandoned
    created_at  REAL DEFAULT (unixepoch()),
    approved_at REAL                    -- when the user approved the plan
);
"""

# JSON list columns (depends_on, output_files, acceptance_criteria) are decoded
# at the point of use with agents.task_spec.json_list(); status is a TaskStatus value.
_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT NOT NULL,          -- task_001…, unique within a project
    project_id           TEXT REFERENCES projects(id),
    title                TEXT NOT NULL,
    description          TEXT NOT NULL,
    task_type            TEXT NOT NULL,          -- TaskType: decides the agent's tools
    status               TEXT DEFAULT 'pending',
    depends_on           TEXT DEFAULT '[]',      -- task IDs that must complete first
    output_files         TEXT DEFAULT '[]',      -- files the task must produce (checked)
    acceptance_criteria  TEXT DEFAULT '[]',      -- what the judge checks
    verify_command       TEXT,                   -- must exit 0 for the task to pass
    result               TEXT,                   -- agent's summary, passed to dependents
    error                TEXT,                   -- last failure reason, shown to the retry
    retry_count          INTEGER DEFAULT 0,
    max_retries          INTEGER DEFAULT 3,
    execution_order      INTEGER DEFAULT 0,      -- position in the plan, for display order
    created_at           REAL DEFAULT (unixepoch()),
    started_at           REAL,
    completed_at         REAL,
    PRIMARY KEY (project_id, id)
);
"""


class SQLiteTaskStore:
    """
    Authoritative persistence layer for all task state.

    Design rules:
    - WAL mode: crash-safe, allows concurrent reads while the orchestrator writes
    - Every connection is opened and closed per operation — no shared long-lived connection
    - Every state transition is a single atomic UPDATE — write to DB before acting on state
    - Execution is single-process and serial, so any task left IN_PROGRESS when the
      process starts up again is unconditionally orphaned (no heartbeat needed)

    DB layout:
        projects table  — one row per /plan invocation
        tasks table     — 10-40 rows per project, one per subtask, keyed by
                          (project_id, id): every plan numbers its tasks task_001…
    """

    def __init__(self, db_path: str = ".repopilot/tasks.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_db()
        logger.info(f"SQLiteTaskStore initialised at {db_path}")

    @contextmanager
    def _conn(self):
        """
        Open a fresh connection, enable WAL + foreign keys, yield, commit, close.
        Using a context manager per operation avoids shared-connection threading issues.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _setup_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_PROJECTS_TABLE + _TASKS_TABLE)
            self._migrate(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id, status)")

    @staticmethod
    def _migrate(conn) -> None:
        """
        Upgrade databases created by older versions:
          - task id alone was the primary key, so a second /plan in the same
            folder failed (every plan numbers its tasks task_001…). Rebuild the
            table keyed by (project_id, id), keeping all rows.
          - add columns introduced later.
        """
        columns = {row["name"]: row["pk"] for row in conn.execute("PRAGMA table_info(tasks)")}
        if columns.get("project_id", 0) == 0:
            logger.info("Migrating tasks table to per-project task ids")
            conn.execute("ALTER TABLE tasks RENAME TO tasks_old")
            conn.execute("DROP INDEX IF EXISTS idx_tasks_project")
            conn.executescript(_TASKS_TABLE)
            new_columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
            shared = ", ".join(c for c in columns if c in new_columns)
            conn.execute(f"INSERT INTO tasks ({shared}) SELECT {shared} FROM tasks_old")
            conn.execute("DROP TABLE tasks_old")
        elif "verify_command" not in columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN verify_command TEXT")

    # ------------------------------------------------------------------
    # Project operations
    # ------------------------------------------------------------------

    def create_project(self, goal: str, plan) -> str:
        """
        Persist an approved ExecutionPlan as a project + task rows.
        Returns the new project_id (UUID).
        """
        project_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO projects(id, name, goal, plan_json, status, approved_at) VALUES (?,?,?,?,?, unixepoch())",
                (project_id, plan.project_name, goal, plan.model_dump_json(), "approved"),
            )
            self._insert_tasks(conn, project_id, plan.tasks, first_order=0)
        logger.info(f"Project {project_id} created with {len(plan.tasks)} tasks")
        return project_id

    @staticmethod
    def _insert_tasks(conn, project_id: str, planned_tasks, first_order: int) -> None:
        for i, pt in enumerate(planned_tasks):
            conn.execute(
                """INSERT INTO tasks
                   (id, project_id, title, description, task_type, depends_on,
                    output_files, acceptance_criteria, verify_command, execution_order)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    pt.id,
                    project_id,
                    pt.title,
                    pt.description,
                    pt.task_type.value,
                    json.dumps(pt.depends_on),
                    json.dumps(pt.output_files),
                    json.dumps(pt.acceptance_criteria),
                    (getattr(pt, "verify_command", None) or "").strip() or None,
                    first_order + i,
                ),
            )

    def apply_replan(self, project_id: str, replaced_ids: list[str], new_tasks, reason: str) -> None:
        """
        Swap in an approved re-plan, atomically: the tasks being replaced become
        REPLACED (kept for the record) and the new tasks are appended as PENDING.
        """
        with self._conn() as conn:
            for task_id in replaced_ids:
                conn.execute(
                    "UPDATE tasks SET status='replaced', error=? WHERE project_id=? AND id=?",
                    (f"Replaced by re-plan: {reason}", project_id, task_id),
                )
            last = conn.execute(
                "SELECT COALESCE(MAX(execution_order), -1) FROM tasks WHERE project_id=?", (project_id,)
            ).fetchone()[0]
            self._insert_tasks(conn, project_id, new_tasks, first_order=last + 1)
        logger.info(f"Project {project_id} re-planned: replaced {replaced_ids}, added {[t.id for t in new_tasks]}")

    def get_latest_approved_project(self) -> str | None:
        """
        Return the most recent project that is still unfinished (status 'approved'),
        or None. Finished projects are marked 'completed' / 'failed' / 'abandoned'
        by finish_project(), so they are never resumed.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM projects WHERE status='approved' ORDER BY created_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        return row["id"] if row else None

    def get_latest_project(self) -> str | None:
        """Return the most recent project_id regardless of status (used by /task_status)."""
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM projects ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def get_project(self, project_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT id, name, goal, status FROM projects WHERE id=?", (project_id,)).fetchone()
        return dict(row) if row else None

    def finish_project(self, project_id: str, status: str) -> None:
        """Mark a project terminal: 'completed', 'failed' or 'abandoned'."""
        with self._conn() as conn:
            conn.execute("UPDATE projects SET status=? WHERE id=?", (status, project_id))
        logger.info(f"Project {project_id} marked {status}")

    # ------------------------------------------------------------------
    # Atomic state transitions
    # ------------------------------------------------------------------

    def claim_task(self, project_id: str, task_id: str) -> bool:
        """
        Atomically transition a task from PENDING → IN_PROGRESS.
        Returns True if the claim succeeded, False if already claimed.
        """
        with self._conn() as conn:
            cur = conn.execute(
                """UPDATE tasks
                   SET status     = 'in_progress',
                       started_at = unixepoch()
                   WHERE project_id = ? AND id = ? AND status = 'pending'""",
                (project_id, task_id),
            )
        return cur.rowcount == 1

    def complete_task(self, project_id: str, task_id: str, result: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET status       = 'completed',
                       result       = ?,
                       completed_at = unixepoch()
                   WHERE project_id = ? AND id = ?""",
                (result, project_id, task_id),
            )
        logger.info(f"Task {task_id} completed")

    def fail_task(self, project_id: str, task_id: str, error: str) -> None:
        """
        Record failure and apply retry logic in a single atomic UPDATE.
        SQL CASE: retries left → 'pending', exhausted → 'failed'.
        """
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET retry_count = retry_count + 1,
                       error       = ?,
                       status      = CASE
                           WHEN retry_count + 1 < max_retries THEN 'pending'
                           ELSE 'failed'
                       END,
                       started_at  = NULL
                   WHERE project_id = ? AND id = ?""",
                (error, project_id, task_id),
            )
        logger.info(f"Task {task_id} failed: {error[:80]}")

    def release_task(self, project_id: str, task_id: str) -> None:
        """
        IN_PROGRESS → PENDING without using a retry: the task was interrupted
        (token budget used up), not judged, so the next /plan run starts it again.
        """
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks SET status = 'pending', started_at = NULL
                   WHERE project_id = ? AND id = ? AND status = 'in_progress'""",
                (project_id, task_id),
            )

    def block_task(self, project_id: str, task_id: str, reason: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status='blocked', error=? WHERE project_id=? AND id=?", (reason, project_id, task_id)
            )

    def block_unreachable_tasks(self, project_id: str) -> int:
        """
        Mark every PENDING task BLOCKED if any of its dependencies is FAILED or
        BLOCKED. Repeats until nothing changes, so blocking propagates down the
        whole dependency chain. Returns the number of tasks blocked.
        """
        blocked_total = 0
        while True:
            tasks = self.get_all_tasks(project_id)
            dead = {t["id"] for t in tasks if t["status"] in ("failed", "blocked")}
            newly = []
            for t in tasks:
                if t["status"] != "pending":
                    continue
                bad = [d for d in json.loads(t["depends_on"] or "[]") if d in dead]
                if bad:
                    newly.append((t["id"], f"Dependency failed: {', '.join(bad)}"))
            if not newly:
                return blocked_total
            for task_id, reason in newly:
                self.block_task(project_id, task_id, reason)
                logger.info(f"Task {task_id} blocked — {reason}")
            blocked_total += len(newly)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    # get ready tasks - get tasks that are in pending state and
    # their dependents are all done
    def get_ready_tasks(self, project_id: str) -> list[dict]:
        """Return PENDING tasks whose every dependency is COMPLETED."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM tasks
                   WHERE project_id = ? AND status = 'pending'
                   ORDER BY execution_order""",
                (project_id,),
            ).fetchall()

        # list of pending tasks

        ready = []
        for row in rows:
            # for each of the pending tasks
            deps = json.loads(row["depends_on"])
            if self._all_deps_done(project_id, deps):
                ready.append(dict(row))
        return ready

    # checking if dependencies of a task are done - true or false
    def _all_deps_done(self, project_id: str, dep_ids: list[str]) -> bool:
        dep_ids = sorted(set(dep_ids))
        if not dep_ids:
            return True
        placeholders = ",".join(["?"] * len(dep_ids))
        with self._conn() as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) FROM tasks
                    WHERE project_id = ? AND id IN ({placeholders})
                      AND status = 'completed'""",
                [project_id, *dep_ids],
            ).fetchone()
        # every dependency must exist and be done — a missing id never counts as done
        return row[0] == len(dep_ids)

    def get_all_tasks(self, project_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id=? ORDER BY execution_order", (project_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_progress(self, project_id: str) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM tasks WHERE project_id=? GROUP BY status", (project_id,)
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def get_dep_results(self, project_id: str, dep_ids: list[str]) -> list[dict]:
        """
        Fetch title + result of completed dependency tasks.
        Injected into each agent's system prompt so it knows what prior tasks produced.
        """
        if not dep_ids:
            return []
        placeholders = ",".join(["?"] * len(dep_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT id, title, result FROM tasks
                    WHERE project_id = ? AND id IN ({placeholders})
                      AND status = 'completed'""",
                [project_id, *dep_ids],
            ).fetchall()
        return [dict(r) for r in rows]
