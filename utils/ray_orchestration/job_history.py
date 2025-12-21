"""
Job History Storage

SQLite-based persistent storage for job history with support for
querying, filtering, and analytics.
"""

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator
from uuid import UUID

from models.job import Job, JobState, JobType, JobPriority

logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path.home() / ".seisproc" / "job_history.db"


class JobHistoryStorage:
    """
    SQLite-based persistent storage for job history.

    Provides:
    - Persistent storage of completed/failed/cancelled jobs
    - Efficient querying by state, type, date range
    - Analytics aggregation (counts, durations, error rates)
    - Automatic cleanup of old entries

    Thread-safe with connection pooling per thread.

    Usage
    -----
    >>> storage = JobHistoryStorage()
    >>> storage.save_job(completed_job)
    >>> jobs = storage.query_jobs(states=[JobState.COMPLETED], limit=10)
    >>> stats = storage.get_statistics()
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_history_days: int = 30,
        auto_cleanup: bool = True,
    ):
        """
        Initialize job history storage.

        Parameters
        ----------
        db_path : Path, optional
            Path to SQLite database file. Defaults to ~/.seisproc/job_history.db
        max_history_days : int
            Maximum age of jobs to keep (default 30 days)
        auto_cleanup : bool
            Whether to automatically cleanup old entries on startup
        """
        self._db_path = db_path or DEFAULT_DB_PATH
        self._max_history_days = max_history_days
        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_database()

        # Cleanup old entries
        if auto_cleanup:
            self.cleanup_old_entries()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Get thread-local database connection.

        Yields
        ------
        sqlite3.Connection
            Database connection for this thread
        """
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row

        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    queued_at TEXT,
                    started_at TEXT,
                    paused_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    error_traceback TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    config TEXT,
                    result TEXT,
                    parent_id TEXT,
                    child_ids TEXT,
                    tags TEXT,
                    metadata TEXT,
                    ray_task_id TEXT,
                    worker_id TEXT,
                    duration_seconds REAL
                )
            """)

            # Indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_state
                ON jobs(state)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_type
                ON jobs(job_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                ON jobs(created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_completed_at
                ON jobs(completed_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_parent_id
                ON jobs(parent_id)
            """)

            conn.commit()
            logger.debug(f"Database initialized at {self._db_path}")

    def save_job(self, job: Job) -> None:
        """
        Save a job to history.

        Parameters
        ----------
        job : Job
            Job to save (usually a terminal state job)
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Calculate duration
                duration = job.duration_seconds

                cursor.execute("""
                    INSERT OR REPLACE INTO jobs (
                        id, name, job_type, state, priority,
                        created_at, queued_at, started_at, paused_at, completed_at,
                        error_message, error_traceback, retry_count, max_retries,
                        config, result, parent_id, child_ids, tags, metadata,
                        ray_task_id, worker_id, duration_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(job.id),
                    job.name,
                    job.job_type.name,
                    job.state.name,
                    job.priority.name,
                    job.created_at.isoformat(),
                    job.queued_at.isoformat() if job.queued_at else None,
                    job.started_at.isoformat() if job.started_at else None,
                    job.paused_at.isoformat() if job.paused_at else None,
                    job.completed_at.isoformat() if job.completed_at else None,
                    job.error_message,
                    job.error_traceback,
                    job.retry_count,
                    job.max_retries,
                    json.dumps(job.config) if job.config else None,
                    json.dumps(job.result) if job.result else None,
                    str(job.parent_id) if job.parent_id else None,
                    json.dumps([str(c) for c in job.child_ids]) if job.child_ids else None,
                    json.dumps(job.tags) if job.tags else None,
                    json.dumps(job.metadata) if job.metadata else None,
                    job.ray_task_id,
                    job.worker_id,
                    duration,
                ))

                conn.commit()
                logger.debug(f"Saved job {job.id} to history")

    def get_job(self, job_id: UUID) -> Optional[Job]:
        """
        Get a job by ID.

        Parameters
        ----------
        job_id : UUID
            Job ID to retrieve

        Returns
        -------
        Job or None
            Job if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (str(job_id),))
            row = cursor.fetchone()

            if row:
                return self._row_to_job(row)
            return None

    def query_jobs(
        self,
        states: Optional[List[JobState]] = None,
        job_types: Optional[List[JobType]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        parent_id: Optional[UUID] = None,
        search_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "completed_at",
        order_desc: bool = True,
    ) -> List[Job]:
        """
        Query jobs with filters.

        Parameters
        ----------
        states : list, optional
            Filter by job states
        job_types : list, optional
            Filter by job types
        start_date : datetime, optional
            Filter jobs created after this date
        end_date : datetime, optional
            Filter jobs created before this date
        tags : list, optional
            Filter by tags (any match)
        parent_id : UUID, optional
            Filter by parent job ID
        search_name : str, optional
            Search in job name (case-insensitive)
        limit : int
            Maximum results to return (default 100)
        offset : int
            Offset for pagination
        order_by : str
            Column to order by (default "completed_at")
        order_desc : bool
            Order descending (default True)

        Returns
        -------
        list
            List of matching jobs
        """
        conditions = []
        params = []

        if states:
            placeholders = ",".join("?" for _ in states)
            conditions.append(f"state IN ({placeholders})")
            params.extend(s.name for s in states)

        if job_types:
            placeholders = ",".join("?" for _ in job_types)
            conditions.append(f"job_type IN ({placeholders})")
            params.extend(t.name for t in job_types)

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())

        if parent_id:
            conditions.append("parent_id = ?")
            params.append(str(parent_id))

        if search_name:
            conditions.append("name LIKE ?")
            params.append(f"%{search_name}%")

        # Build query
        query = "SELECT * FROM jobs"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Validate order_by to prevent SQL injection
        valid_columns = {"id", "name", "job_type", "state", "created_at", "completed_at", "duration_seconds"}
        if order_by not in valid_columns:
            order_by = "completed_at"

        order = "DESC" if order_desc else "ASC"
        query += f" ORDER BY {order_by} {order}"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            jobs = []
            for row in rows:
                job = self._row_to_job(row)
                # Post-filter by tags if specified
                if tags:
                    if not any(tag in job.tags for tag in tags):
                        continue
                jobs.append(job)

            return jobs

    def get_recent_jobs(self, limit: int = 20) -> List[Job]:
        """
        Get most recent completed jobs.

        Parameters
        ----------
        limit : int
            Maximum jobs to return

        Returns
        -------
        list
            List of recent jobs, newest first
        """
        return self.query_jobs(
            states=[JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED],
            limit=limit,
            order_by="completed_at",
            order_desc=True,
        )

    def get_failed_jobs(self, limit: int = 20) -> List[Job]:
        """
        Get recent failed jobs.

        Parameters
        ----------
        limit : int
            Maximum jobs to return

        Returns
        -------
        list
            List of failed jobs
        """
        return self.query_jobs(
            states=[JobState.FAILED],
            limit=limit,
            order_by="completed_at",
            order_desc=True,
        )

    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get job statistics.

        Parameters
        ----------
        start_date : datetime, optional
            Start of period
        end_date : datetime, optional
            End of period

        Returns
        -------
        dict
            Statistics including counts, durations, error rates
        """
        conditions = []
        params = []

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total counts by state
            cursor.execute(f"""
                SELECT state, COUNT(*) as count
                FROM jobs
                {where_clause}
                GROUP BY state
            """, params)

            state_counts = {}
            for row in cursor.fetchall():
                state_counts[row["state"]] = row["count"]

            # Counts by type
            cursor.execute(f"""
                SELECT job_type, COUNT(*) as count
                FROM jobs
                {where_clause}
                GROUP BY job_type
            """, params)

            type_counts = {}
            for row in cursor.fetchall():
                type_counts[row["job_type"]] = row["count"]

            # Duration statistics
            duration_where = where_clause + " AND duration_seconds IS NOT NULL" if where_clause else "WHERE duration_seconds IS NOT NULL"
            cursor.execute(f"""
                SELECT
                    AVG(duration_seconds) as avg_duration,
                    MIN(duration_seconds) as min_duration,
                    MAX(duration_seconds) as max_duration,
                    SUM(duration_seconds) as total_duration
                FROM jobs
                {duration_where}
            """, params)

            duration_row = cursor.fetchone()

            # Error rate
            total_completed = sum(
                state_counts.get(s, 0) for s in
                ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]
            )
            failed_count = state_counts.get("FAILED", 0)
            error_rate = (failed_count / total_completed * 100) if total_completed > 0 else 0

            return {
                "total_jobs": sum(state_counts.values()),
                "by_state": state_counts,
                "by_type": type_counts,
                "duration": {
                    "avg_seconds": duration_row["avg_duration"],
                    "min_seconds": duration_row["min_duration"],
                    "max_seconds": duration_row["max_duration"],
                    "total_seconds": duration_row["total_duration"],
                },
                "error_rate_percent": round(error_rate, 2),
                "period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
            }

    def get_daily_counts(
        self,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Get job counts grouped by day.

        Parameters
        ----------
        days : int
            Number of days to include

        Returns
        -------
        list
            Daily counts with date and counts by state
        """
        start_date = datetime.now() - timedelta(days=days)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    DATE(created_at) as date,
                    state,
                    COUNT(*) as count
                FROM jobs
                WHERE created_at >= ?
                GROUP BY DATE(created_at), state
                ORDER BY date DESC
            """, (start_date.isoformat(),))

            # Aggregate by date
            daily = {}
            for row in cursor.fetchall():
                date = row["date"]
                if date not in daily:
                    daily[date] = {"date": date, "total": 0, "by_state": {}}
                daily[date]["by_state"][row["state"]] = row["count"]
                daily[date]["total"] += row["count"]

            return list(daily.values())

    def delete_job(self, job_id: UUID) -> bool:
        """
        Delete a job from history.

        Parameters
        ----------
        job_id : UUID
            Job ID to delete

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs WHERE id = ?", (str(job_id),))
                deleted = cursor.rowcount > 0
                conn.commit()

                if deleted:
                    logger.debug(f"Deleted job {job_id} from history")
                return deleted

    def cleanup_old_entries(self) -> int:
        """
        Clean up entries older than max_history_days.

        Returns
        -------
        int
            Number of entries deleted
        """
        cutoff = datetime.now() - timedelta(days=self._max_history_days)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM jobs WHERE created_at < ?",
                    (cutoff.isoformat(),)
                )
                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old job history entries")
                return deleted

    def clear_all(self) -> int:
        """
        Clear all job history.

        Returns
        -------
        int
            Number of entries deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM jobs")
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"Cleared all {deleted} job history entries")
                return deleted

    def export_to_json(self, file_path: Path) -> int:
        """
        Export all jobs to JSON file.

        Parameters
        ----------
        file_path : Path
            Output file path

        Returns
        -------
        int
            Number of jobs exported
        """
        jobs = self.query_jobs(limit=10000)

        data = {
            "exported_at": datetime.now().isoformat(),
            "job_count": len(jobs),
            "jobs": [job.to_dict() for job in jobs],
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(jobs)} jobs to {file_path}")
        return len(jobs)

    def import_from_json(self, file_path: Path) -> int:
        """
        Import jobs from JSON file.

        Parameters
        ----------
        file_path : Path
            Input file path

        Returns
        -------
        int
            Number of jobs imported
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        count = 0
        for job_data in data.get("jobs", []):
            try:
                job = Job.from_dict(job_data)
                self.save_job(job)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import job: {e}")

        logger.info(f"Imported {count} jobs from {file_path}")
        return count

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert database row to Job object."""
        return Job(
            id=UUID(row["id"]),
            name=row["name"],
            job_type=JobType[row["job_type"]],
            state=JobState[row["state"]],
            priority=JobPriority[row["priority"]],
            created_at=datetime.fromisoformat(row["created_at"]),
            queued_at=datetime.fromisoformat(row["queued_at"]) if row["queued_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            paused_at=datetime.fromisoformat(row["paused_at"]) if row["paused_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            error_message=row["error_message"],
            error_traceback=row["error_traceback"],
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            config=json.loads(row["config"]) if row["config"] else {},
            result=json.loads(row["result"]) if row["result"] else None,
            parent_id=UUID(row["parent_id"]) if row["parent_id"] else None,
            child_ids=[UUID(c) for c in json.loads(row["child_ids"])] if row["child_ids"] else [],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            ray_task_id=row["ray_task_id"],
            worker_id=row["worker_id"],
        )

    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            del self._local.connection


# Singleton instance
_history_storage: Optional[JobHistoryStorage] = None
_history_lock = threading.Lock()


def get_job_history_storage(
    db_path: Optional[Path] = None,
    max_history_days: int = 30,
) -> JobHistoryStorage:
    """
    Get or create the singleton JobHistoryStorage instance.

    Parameters
    ----------
    db_path : Path, optional
        Path to database file
    max_history_days : int
        Maximum history age

    Returns
    -------
    JobHistoryStorage
        The singleton storage instance
    """
    global _history_storage

    with _history_lock:
        if _history_storage is None:
            _history_storage = JobHistoryStorage(
                db_path=db_path,
                max_history_days=max_history_days,
            )
        return _history_storage
