from typing import Any, Dict

import pytest

from agents.base_agent import LLMConfig
from agents.course_planner_agent import CoursePlannerAgent
from agents.lesson_writer_agent import LessonWriterAgent
from agents.quiz_agent import QuizAgent
from agents.validation_agent import ValidationAgent
from agents.video_generator_agent import VideoGeneratorAgent
from agents.video_script_agent import VideoScriptAgent


class DummyLLMMixin:
    llm_config = LLMConfig(api_key="dummy-key")  # satisfy BaseAgent check

    def call_llm(self, prompt: str) -> Dict[str, Any]:  # type: ignore[override]
        if "course plan" in prompt.lower():
            return {
                "title": "Demo Course",
                "modules": [{"title": "Module 1", "lessons": ["Lesson 1"]}],
            }
        if "full lesson" in prompt.lower():
            return {
                "title": "Lesson 1",
                "text": "This is a sufficiently long lesson text.",
                "summary": "Summary",
            }
        if "video script" in prompt.lower():
            return {
                "lesson": "Lesson 1",
                "scenes": [{"text": "Scene 1", "duration": 10}],
            }
        if "multiple-choice" in prompt.lower():
            return {
                "lesson": "Lesson 1",
                "questions": [
                    {
                        "question": "Q1",
                        "options": ["a", "b", "c"],
                        "correct_index": 1,
                    },
                    {
                        "question": "Q2",
                        "options": ["a", "b", "c"],
                        "correct_index": 2,
                    },
                    {
                        "question": "Q3",
                        "options": ["a", "b", "c"],
                        "correct_index": 0,
                    },
                ],
            }
        return {}


class DummyCoursePlanner(DummyLLMMixin, CoursePlannerAgent):
    pass


class DummyLessonWriter(DummyLLMMixin, LessonWriterAgent):
    pass


class DummyVideoScriptAgent(DummyLLMMixin, VideoScriptAgent):
    pass


class DummyQuizAgent(DummyLLMMixin, QuizAgent):
    pass


class DummyValidationAgent(DummyLLMMixin, ValidationAgent):
    def _run_llm_validation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        return content


def test_course_planner_agent():
    agent = DummyCoursePlanner()
    result = agent.generate({"learning_outcomes": ["Outcome 1"]})
    assert "modules" in result


def test_lesson_writer_agent():
    agent = DummyLessonWriter()
    result = agent.generate(
        {
            "module_name": "Module 1",
            "lesson_name": "Lesson 1",
            "learning_outcomes": ["Outcome 1"],
        }
    )
    assert "text" in result


def test_video_script_agent():
    agent = DummyVideoScriptAgent()
    lesson = {
        "title": "Lesson 1",
        "text": "This is a sufficiently long lesson text.",
        "summary": "Summary",
    }
    result = agent.generate(lesson)
    assert "scenes" in result


def test_quiz_agent():
    agent = DummyQuizAgent()
    lesson = {
        "title": "Lesson 1",
        "text": "This is a sufficiently long lesson text.",
        "summary": "Summary",
    }
    result = agent.generate(lesson)
    assert "questions" in result


def test_video_generator_agent():
    agent = VideoGeneratorAgent()
    script = {"lesson": "Lesson 1", "scenes": [{"text": "Scene 1", "duration": 10}]}
    result = agent.generate(script)
    assert "video_path" in result


def test_validation_agent_pass_through():
    agent = DummyValidationAgent()
    content = {"title": "Lesson 1", "text": "x" * 30, "summary": "S"}
    result = agent.generate(content)
    assert result["status"] in {"ok", "fixed", "fixed_with_issues"}
