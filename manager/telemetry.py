from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
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
    """
    Captures approvals/rejections from different stages so we can surface trends.
    """

    def __init__(self) -> None:
        self.events: List[FeedbackEvent] = []

    def record(self, stage: str, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        event = FeedbackEvent(stage=stage, success=success, metadata=metadata or {})
        self.events.append(event)
        if not success:
            logger.warning("Feedback failure on %s: %s", stage, event.metadata)

    def failure_trends(self) -> Dict[str, int]:
        failures: Dict[str, int] = defaultdict(int)
        for event in self.events:
            if not event.success:
                failures[event.stage] += 1
        return dict(failures)

    def summary(self) -> Dict[str, Any]:
        success_count = sum(1 for event in self.events if event.success)
        failure_count = len(self.events) - success_count
        return {
            "total_events": len(self.events),
            "success_count": success_count,
            "failure_count": failure_count,
            "failures_by_stage": self.failure_trends(),
            "recent_failures": [
                event.metadata for event in self.events if not event.success
            ][-5:],
        }


class PerformanceMonitor:
    """
    Lightweight perf tracker that times stage execution and keeps aggregates.
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def track(self, stage: str) -> Any:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics[stage].append(duration)
            logger.debug("Stage %s took %.2fs", stage, duration)

    def capture(self, stage: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self.track(stage):
            return func(*args, **kwargs)

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for stage, durations in self.metrics.items():
            if not durations:
                continue
            summary[stage] = {
                "count": len(durations),
                "avg_seconds": sum(durations) / len(durations),
                "max_seconds": max(durations),
                "min_seconds": min(durations),
            }
        return summary


class WorkloadManager:
    """
    Queues heavy tasks (video rendering) and captures telemetry + queue depth.
    """

    def __init__(self, max_video_workers: int = 2) -> None:
        self.max_video_workers = max_video_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self.video_tasks: List[Tuple[str, Future]] = []
        self.queue_depth_history: List[int] = []
        self.video_durations: List[Dict[str, Any]] = []

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.max_video_workers)
        return self._executor

    def submit_video_task(self, lesson_name: str, script_payload: Dict[str, Any], generator: Any) -> None:
        pending = sum(1 for _, task in self.video_tasks if not task.done())
        self.queue_depth_history.append(pending)
        logger.info("Queueing video generation for '%s' (queue depth: %d)", lesson_name, pending)

        def _task() -> Dict[str, Any]:
            start = time.perf_counter()
            try:
                video_info = generator.generate(script_payload)
                success = True
                return {"lesson_name": lesson_name, "video_info": video_info, "duration": time.perf_counter() - start, "success": success}
            except Exception as exc:  # noqa: BLE001
                duration = time.perf_counter() - start
                logger.exception("Video generation failed for '%s': %s", lesson_name, exc)
                return {"lesson_name": lesson_name, "video_info": None, "duration": duration, "success": False, "error": str(exc)}

        executor = self._ensure_executor()
        future = executor.submit(_task)
        self.video_tasks.append((lesson_name, future))

    def collect_video_results(self) -> Tuple[List[Any], List[Dict[str, Any]]]:
        videos: List[Any] = []
        telemetry: List[Dict[str, Any]] = []
        for lesson_name, future in self.video_tasks:
            result = future.result()
            videos.append(result.get("video_info"))
            telemetry.append(
                {
                    "lesson_name": lesson_name,
                    "duration": result.get("duration"),
                    "success": result.get("success"),
                    "error": result.get("error"),
                }
            )
        self.video_durations.extend(telemetry)
        self.close()
        return videos, telemetry

    def close(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.video_tasks.clear()

    def summary(self) -> Dict[str, Any]:
        success_count = sum(1 for entry in self.video_durations if entry.get("success"))
        failure_count = len(self.video_durations) - success_count
        queue_depth = self.queue_depth_history or [0]
        return {
            "queue_depth_max": max(queue_depth),
            "queue_depth_avg": sum(queue_depth) / len(queue_depth),
            "success_count": success_count,
            "failure_count": failure_count,
            "tasks": self.video_durations,
        }
