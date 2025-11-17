from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, validator


class Question(BaseModel):
    question: str
    options: List[str]
    correct_index: int

    @validator("correct_index")
    def validate_correct_index(cls, value: int, values: Dict[str, Any]) -> int:
        options = values.get("options") or []
        if not 0 <= value < len(options):
            raise ValueError("correct_index must reference an existing option")
        return value


class Quiz(BaseModel):
    lesson: str
    questions: List[Question]

    def to_json(self) -> Dict[str, Any]:
        return self.dict()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Quiz":
        return cls.parse_obj(data)

