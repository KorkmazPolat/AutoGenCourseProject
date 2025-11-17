from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.course_planner_agent import CoursePlannerAgent
from agents.lesson_writer_agent import LessonWriterAgent
from agents.quiz_agent import QuizAgent
from agents.validation_agent import ValidationAgent
from agents.video_generator_agent import VideoGeneratorAgent
from agents.video_script_agent import VideoScriptAgent
from schemas.course_plan import CoursePlan

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self) -> None:
        self.course_planner = CoursePlannerAgent()
        self.lesson_writer = LessonWriterAgent()
        self.video_script_agent = VideoScriptAgent()
        self.quiz_agent = QuizAgent()
        self.video_generator = VideoGeneratorAgent()
        self.validator = ValidationAgent()

    def run(self, learning_outcomes: List[str]) -> Dict[str, Any]:
        logger.info("Starting course generation pipeline.")
        course_plan_json = self.course_planner.generate(
            {"learning_outcomes": learning_outcomes}
        )
        plan_validation = self.validator.generate(course_plan_json)
        course_plan = CoursePlan.parse_obj(plan_validation["validated_content"])

        lessons: List[Dict[str, Any]] = []
        videos: List[Dict[str, Any]] = []
        quizzes: List[Dict[str, Any]] = []

        for module in course_plan.modules:
            logger.info("Processing module '%s'", module.title)
            for lesson_name in module.lessons:
                logger.info("Generating lesson '%s'", lesson_name)
                lesson_json = self.lesson_writer.generate(
                    {
                        "module_name": module.title,
                        "lesson_name": lesson_name,
                        "learning_outcomes": learning_outcomes,
                    }
                )
                lesson_validation = self.validator.generate(lesson_json)
                validated_lesson = lesson_validation["validated_content"]
                lessons.append(validated_lesson)

                logger.info("Generating video script for lesson '%s'", lesson_name)
                script_json = self.video_script_agent.generate(validated_lesson)
                script_validation = self.validator.generate(script_json)
                validated_script = script_validation["validated_content"]

                logger.info("Generating quiz for lesson '%s'", lesson_name)
                quiz_json = self.quiz_agent.generate(validated_lesson)
                quiz_validation = self.validator.generate(quiz_json)
                validated_quiz = quiz_validation["validated_content"]

                logger.info("Generating video for lesson '%s'", lesson_name)
                video_info = self.video_generator.generate(validated_script)

                videos.append(video_info)
                quizzes.append(validated_quiz)

        final_validation = self.validator.generate(
            {
                "course_plan": course_plan.to_json(),
                "lessons": lessons,
                "videos": videos,
                "quizzes": quizzes,
            }
        )

        logger.info("Course generation pipeline finished.")
        return {
            "course_plan": course_plan.to_json(),
            "lessons": lessons,
            "videos": videos,
            "quizzes": quizzes,
            "final_validation": final_validation,
        }
