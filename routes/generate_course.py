from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from manager.agent_manager import AgentManager


router = APIRouter()


class GenerateCourseRequest(BaseModel):
    learning_outcomes: List[str]
    skip_video: bool = False


@router.post("/generate-course")
def generate_course(payload: GenerateCourseRequest) -> Dict[str, Any]:
    manager = AgentManager()
    result = manager.run(payload.learning_outcomes, skip_video=payload.skip_video)
    return result

