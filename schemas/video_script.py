from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class Scene(BaseModel):
    text: str
    duration: int


class VideoScript(BaseModel):
    lesson: str
    scenes: List[Scene]

    def to_json(self) -> Dict[str, Any]:
        return self.dict()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "VideoScript":
        return cls.parse_obj(data)

