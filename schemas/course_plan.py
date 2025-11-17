from __future__ import annotations

from typing import Any, Dict, List, Type, TypeVar

from pydantic import BaseModel


class Module(BaseModel):
    title: str
    lessons: List[str]


class CoursePlan(BaseModel):
    title: str
    modules: List[Module]

    def to_json(self) -> Dict[str, Any]:
        return self.dict()

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CoursePlan":
        return cls.parse_obj(data)


T = TypeVar("T", bound=BaseModel)


def model_from_json(model_cls: Type[T], data: Dict[str, Any]) -> T:
    return model_cls.parse_obj(data)

