from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_lesson(lesson: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    title = lesson.get("title", "")
    text = lesson.get("text", "")

    if len(title.strip()) < 3:
        issues.append("Lesson title is too short.")
    if len(text.strip()) < 20:
        issues.append("Lesson text is too short.")

    return (len(issues) == 0, issues)

