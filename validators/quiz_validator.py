from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_quiz(quiz: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    questions = quiz.get("questions") or []

    if len(questions) < 3:
        issues.append("Quiz must contain at least 3 questions.")

    for i, question in enumerate(questions):
        options = question.get("options") or []
        if len(options) < 2:
            issues.append(f"Question {i} must have at least 2 options.")
        correct_index = question.get("correct_index")
        if not isinstance(correct_index, int) or not 0 <= correct_index < len(options):
            issues.append(f"Question {i} has invalid correct_index.")

    return (len(issues) == 0, issues)

