"""Lightweight in-memory job manager for tracking long-running tasks.

Designed so we can later swap the backend to Redis or another store
without changing the call sites.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed
    progress: int = 0
    step: str = ""
    message: str = ""
    eta_seconds: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, *, step: str = "queued", message: str = "Preparing task") -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, status="queued", progress=0, step=step, message=message)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        step: Optional[str] = None,
        message: Optional[str] = None,
        eta_seconds: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(0, min(100, int(progress)))
            if step is not None:
                job.step = step
            if message is not None:
                job.message = message
            if eta_seconds is not None:
                job.eta_seconds = eta_seconds
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            job.updated_at = time.time()
            return job

    def start_job(self, job_id: str, *, step: str = "starting", message: str = "Starting task") -> Optional[Job]:
        return self.update_job(job_id, status="running", step=step, message=message, progress=1)

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> Optional[Job]:
        return self.update_job(job_id, status="completed", progress=100, step="completed", message="Completed", result=result)

    def fail_job(self, job_id: str, error: str) -> Optional[Job]:
        return self.update_job(job_id, status="failed", message="Failed", error=error)

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            # return a shallow copy to avoid accidental mutation
            return Job(**job.__dict__)

    def set_progress(self, job_id: str, progress: int, *, step: Optional[str] = None, message: Optional[str] = None, eta_seconds: Optional[int] = None) -> Optional[Job]:
        return self.update_job(job_id, progress=progress, step=step, message=message, eta_seconds=eta_seconds)


# Singleton manager for app-wide usage
job_manager = JobManager()
