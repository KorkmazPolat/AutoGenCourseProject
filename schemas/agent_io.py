from __future__ import annotations

from typing import List

from pydantic import BaseModel


class CoursePlannerInput(BaseModel):
    learning_outcomes: List[str]
    num_modules: int | None = None
    num_lessons: int | None = None
    feedback: str | None = None
    previous_plan: dict | None = None


class LessonWriterInput(BaseModel):
    module_name: str
    lesson_name: str
    learning_outcomes: List[str]


class VideoScriptInput(BaseModel):
    lesson: str


class QuizInput(BaseModel):
    lesson: str


