from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel


class LessonContent(BaseModel):
    title: str
    text: str
    summary: str

    def to_json(self) -> Dict[str, Any]:
        return self.dict()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LessonContent":
        return cls.parse_obj(data)

