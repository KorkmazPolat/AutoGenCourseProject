from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_script(script: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    scenes = script.get("scenes")

    if not scenes:
        issues.append("Video script must contain at least one scene.")
    else:
        for i, scene in enumerate(scenes):
            duration = scene.get("duration", 0)
            if duration <= 0:
                issues.append(f"Scene {i} has non-positive duration.")

    return (len(issues) == 0, issues)

