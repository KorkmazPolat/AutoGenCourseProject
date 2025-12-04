from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    stage: str
    success: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackMonitor:
    """Capture approval/rejection metadata per pipeline stage."""

    def __init__(self) -> None:
        self._events: List[FeedbackEvent] = []

    def log_event(self, stage: str, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        event = FeedbackEvent(stage=stage, success=success, metadata=metadata or {})
        self._events.append(event)
        if not success:
            logger.warning("Feedback failure @%s -> %s", stage, event.metadata)

    def failure_trends(self) -> Dict[str, int]:
        failures: Dict[str, int] = {}
        for event in self._events:
            if not event.success:
                failures[event.stage] = failures.get(event.stage, 0) + 1
        return failures

    def summary(self) -> Dict[str, Any]:
        stage_totals: Dict[str, Dict[str, int]] = {}
        for event in self._events:
            stage_stats = stage_totals.setdefault(event.stage, {"approvals": 0, "rejections": 0})
            if event.success:
                stage_stats["approvals"] += 1
            else:
                stage_stats["rejections"] += 1

        return {
            "total_events": len(self._events),
            "stages": stage_totals,
            "failure_trends": self.failure_trends(),
            "recent_failures": [
                {
                    "stage": event.stage,
                    "metadata": event.metadata,
                    "ts": event.timestamp,
                }
                for event in self._events
                if not event.success
            ][:5],
        }


class PerformanceMonitor:
    """Record execution durations via context manager usage."""

    class _StageTimer:
        def __init__(self, monitor: PerformanceMonitor, stage: str) -> None:
            self._monitor = monitor
            self._stage = stage
            self._start: float | None = None

        def __enter__(self) -> PerformanceMonitor._StageTimer:
            self._start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, exc_tb) -> None:
            end = time.perf_counter()
            duration = end - (self._start or end)
            self._monitor._record(self._stage, duration)
            if exc:
                logger.exception("Stage %s failed after %.2fs", self._stage, duration)

    def __init__(self) -> None:
        self._durations: Dict[str, List[float]] = {}

    def track(self, stage: str) -> PerformanceMonitor._StageTimer:
        return PerformanceMonitor._StageTimer(self, stage)

    def _record(self, stage: str, duration: float) -> None:
        self._durations.setdefault(stage, []).append(duration)
        logger.debug("Stage %s duration %.2fs", stage, duration)

    def summary(self) -> Dict[str, Dict[str, float]]:
        snapshot: Dict[str, Dict[str, float]] = {}
        for stage, durations in self._durations.items():
            if not durations:
                continue
            snapshot[stage] = {
                "count": len(durations),
                "avg_seconds": sum(durations) / len(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
            }
        return snapshot


class WorkloadManager:
    """Manage background tasks (e.g., video rendering) with bounded concurrency."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: List[Tuple[str, Future]] = []
        self._queue_depths: List[int] = []
        self._task_log: List[Dict[str, Any]] = []

    def submit_task(self, task_name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        pending = sum(1 for _, future in self._tasks if not future.done())
        self._queue_depths.append(pending)

        def _runner() -> Dict[str, Any]:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                return {"task_name": task_name, "result": result, "success": True, "duration": duration, "error": None}
            except Exception as exc:  # noqa: BLE001
                duration = time.perf_counter() - start
                logger.exception("Async task %s failed: %s", task_name, exc)
                return {"task_name": task_name, "result": None, "success": False, "duration": duration, "error": str(exc)}

        future = self._executor.submit(_runner)
        self._tasks.append((task_name, future))
        return future

    def collect_results(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for task_name, future in self._tasks:
            data = future.result()
            results.append(data)
            self._task_log.append(data)
        self._tasks.clear()
        return results

    def summary(self) -> Dict[str, Any]:
        success_count = sum(1 for task in self._task_log if task.get("success"))
        failure_count = len(self._task_log) - success_count
        depths = self._queue_depths or [0]
        return {
            "tasks_recorded": len(self._task_log),
            "successes": success_count,
            "failures": failure_count,
            "max_queue_depth": max(depths),
            "avg_queue_depth": sum(depths) / len(depths),
            "active_tasks": sum(1 for _, future in self._tasks if not future.done()),
            "task_log": self._task_log[-10:],
        }

    def shutdown(self) -> None:
        for _, future in self._tasks:
            future.cancel()
        self._tasks.clear()
        self._executor.shutdown(wait=True)

    @property
    def current_depth(self) -> int:
        return sum(1 for _, future in self._tasks if not future.done())
