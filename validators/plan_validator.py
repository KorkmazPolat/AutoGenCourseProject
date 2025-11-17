from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_plan(plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    modules = plan.get("modules")
    if not modules:
        issues.append("Course plan must contain at least one module.")
    else:
        for i, module in enumerate(modules):
            if not module.get("title"):
                issues.append(f"Module {i} is missing a title.")
            lessons = module.get("lessons")
            if not lessons:
                issues.append(f"Module {i} must contain at least one lesson.")

    return (len(issues) == 0, issues)

