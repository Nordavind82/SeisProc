"""
Storage manager for processing sessions.

Manages temporary directories, disk space validation, and cleanup
for parallel processing operations.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ProcessingSession:
    """A processing session with isolated storage."""
    session_id: str
    session_dir: Path
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def temp_dir(self) -> Path:
        """Temporary directory for intermediate files."""
        d = self.session_dir / 'temp'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        """Output directory for processed data."""
        d = self.session_dir / 'output'
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cleanup_temp(self):
        """Remove temporary files, keep output."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def cleanup_all(self):
        """Remove entire session directory."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir, ignore_errors=True)

    def save_session_info(self):
        """Save session metadata for recovery."""
        info = {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'status': 'active'
        }
        info_path = self.session_dir / 'session_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

    def mark_complete(self):
        """Mark session as complete."""
        info_path = self.session_dir / 'session_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            info['status'] = 'complete'
            info['completed_at'] = datetime.now().isoformat()
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)


class ProcessingStorageManager:
    """
    Manages storage for processing operations.

    Handles:
    - Creating isolated session directories
    - Disk space validation
    - Cleanup of old/orphaned sessions
    - Crash recovery
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize storage manager.

        Args:
            base_dir: Base directory for storage. If None, uses default
                     from AppSettings (~/.seisproc/data)
        """
        if base_dir is None:
            from models.app_settings import AppSettings
            settings = AppSettings()
            self.base_dir = settings.get_effective_storage_directory()
        else:
            self.base_dir = Path(base_dir)

        self.processing_dir = self.base_dir / 'processing'
        self.processing_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, name: str) -> ProcessingSession:
        """
        Create a new processing session with isolated directory.

        Args:
            name: Base name for the session (e.g., dataset name)

        Returns:
            ProcessingSession object with temp and output directories
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_id = f"{name}_{timestamp}"

        # Sanitize session_id for filesystem
        session_id = "".join(c if c.isalnum() or c in '_-' else '_' for c in session_id)

        session_dir = self.processing_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        session = ProcessingSession(
            session_id=session_id,
            session_dir=session_dir
        )
        session.save_session_info()

        return session

    def get_disk_space_available(self, path: Optional[Path] = None) -> int:
        """
        Get available disk space in bytes.

        Args:
            path: Path to check. If None, uses base_dir.

        Returns:
            Available space in bytes
        """
        check_path = path or self.base_dir
        stat = os.statvfs(str(check_path))
        return stat.f_bavail * stat.f_frsize

    def get_disk_space_total(self, path: Optional[Path] = None) -> int:
        """Get total disk space in bytes."""
        check_path = path or self.base_dir
        stat = os.statvfs(str(check_path))
        return stat.f_blocks * stat.f_frsize

    def estimate_required_space(
        self,
        n_traces: int,
        n_samples: int,
        include_temp: bool = True
    ) -> int:
        """
        Estimate disk space required for processing.

        Args:
            n_traces: Number of traces
            n_samples: Samples per trace
            include_temp: Include space for temporary files

        Returns:
            Estimated space needed in bytes
        """
        # Trace data: float32 = 4 bytes
        trace_bytes = n_traces * n_samples * 4

        # Output Zarr (uncompressed)
        output_size = trace_bytes

        # Temporary files (processed zarr, maybe headers)
        temp_size = trace_bytes if include_temp else 0

        # Headers and metadata overhead (~1% of trace data)
        overhead = int(trace_bytes * 0.01)

        return output_size + temp_size + overhead

    def validate_disk_space(
        self,
        n_traces: int,
        n_samples: int,
        safety_margin: float = 1.2
    ) -> tuple:
        """
        Validate sufficient disk space exists.

        Args:
            n_traces: Number of traces to process
            n_samples: Samples per trace
            safety_margin: Multiplier for required space (default 1.2 = 20% extra)

        Returns:
            Tuple of (is_sufficient: bool, message: str)
        """
        available = self.get_disk_space_available()
        required = self.estimate_required_space(n_traces, n_samples)
        required_with_margin = int(required * safety_margin)

        available_gb = available / (1024 ** 3)
        required_gb = required_with_margin / (1024 ** 3)

        if available >= required_with_margin:
            return True, f"Sufficient space: {available_gb:.1f} GB available, {required_gb:.1f} GB required"
        else:
            return False, (
                f"Insufficient disk space!\n"
                f"Available: {available_gb:.1f} GB\n"
                f"Required: {required_gb:.1f} GB\n"
                f"Please free up at least {required_gb - available_gb:.1f} GB"
            )

    def list_sessions(self) -> List[ProcessingSession]:
        """
        List all processing sessions.

        Returns:
            List of ProcessingSession objects
        """
        sessions = []

        if not self.processing_dir.exists():
            return sessions

        for item in self.processing_dir.iterdir():
            if item.is_dir():
                info_path = item / 'session_info.json'
                if info_path.exists():
                    try:
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        session = ProcessingSession(
                            session_id=info['session_id'],
                            session_dir=item,
                            created_at=datetime.fromisoformat(info['created_at'])
                        )
                        sessions.append(session)
                    except (json.JSONDecodeError, KeyError):
                        # Orphaned directory
                        session = ProcessingSession(
                            session_id=item.name,
                            session_dir=item
                        )
                        sessions.append(session)

        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than specified age.

        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        cleaned = 0

        for session in self.list_sessions():
            # Check session info for status
            info_path = session.session_dir / 'session_info.json'
            if info_path.exists():
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    # Don't clean up complete sessions - they might be in use
                    if info.get('status') == 'complete':
                        continue
                except:
                    pass

            # Clean up old active/orphaned sessions
            if session.created_at < cutoff:
                session.cleanup_all()
                cleaned += 1
                print(f"Cleaned up old session: {session.session_id}")

        return cleaned

    def cleanup_orphaned(self):
        """
        Remove orphaned session directories (no session_info.json).

        Returns:
            Number of directories cleaned
        """
        cleaned = 0

        if not self.processing_dir.exists():
            return cleaned

        for item in self.processing_dir.iterdir():
            if item.is_dir():
                info_path = item / 'session_info.json'
                if not info_path.exists():
                    # Orphaned directory - remove it
                    try:
                        shutil.rmtree(item)
                        cleaned += 1
                        print(f"Cleaned up orphaned directory: {item.name}")
                    except Exception as e:
                        print(f"Warning: Could not remove {item}: {e}")

        return cleaned

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        sessions = self.list_sessions()
        total_size = 0

        for session in sessions:
            if session.session_dir.exists():
                for f in session.session_dir.rglob('*'):
                    if f.is_file():
                        total_size += f.stat().st_size

        return {
            'n_sessions': len(sessions),
            'total_size_bytes': total_size,
            'total_size_gb': total_size / (1024 ** 3),
            'disk_available_gb': self.get_disk_space_available() / (1024 ** 3),
            'disk_total_gb': self.get_disk_space_total() / (1024 ** 3),
        }
