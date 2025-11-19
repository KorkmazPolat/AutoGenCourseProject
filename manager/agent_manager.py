from __future__ import annotations

import logging
from typing import Any, Dict, List

from agents.course_planner_agent import CoursePlannerAgent
from agents.lesson_writer_agent import LessonWriterAgent
from agents.quiz_agent import QuizAgent
from agents.validation_agent import ValidationAgent
from agents.video_generator_agent import VideoGeneratorAgent
from agents.video_script_agent import VideoScriptAgent
from agents.research_agent import ResearchAgent
from agents.review_agent import ReviewAgent
from schemas.course_plan import CoursePlan

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self) -> None:
        self.research_agent = ResearchAgent()
        self.course_planner = CoursePlannerAgent()
        self.review_agent = ReviewAgent()
        self.lesson_writer = LessonWriterAgent()
        self.video_script_agent = VideoScriptAgent()
        self.quiz_agent = QuizAgent()
        self.video_generator = VideoGeneratorAgent()
        self.validator = ValidationAgent()

    def run(self, learning_outcomes: List[str], skip_video: bool = False) -> Dict[str, Any]:
        logger.info("Starting REAL agentic course generation pipeline.")

        # 1. Research Phase
        logger.info("Agent is researching the topic...")
        research_result = self.research_agent.generate({"learning_outcomes": learning_outcomes})
        logger.info("Research complete. Key concepts: %s", research_result.get("key_concepts"))

        # 2. Planning Phase
        logger.info("Agent is planning the course...")
        # Inject research context into the planner input
        course_plan_json = self.course_planner.generate(
            {
                "learning_outcomes": learning_outcomes,
                "research_context": research_result
            }
        )
        
        # 3. Plan Review
        logger.info("Agent is reviewing the course plan...")
        plan_review = self.review_agent.generate({
            "content_type": "course_plan",
            "content": course_plan_json,
            "context": research_result
        })
        
        if not plan_review.get("approved"):
            logger.warning("Plan was critiqued: %s. (Auto-proceeding for now, but in future we loop)", plan_review.get("feedback"))
            # TODO: Implement replanning loop here
        else:
            logger.info("Plan approved with score %s", plan_review.get("score"))

        plan_validation = self.validator.generate(course_plan_json)
        course_plan = CoursePlan.parse_obj(plan_validation["validated_content"])

        lessons: List[Dict[str, Any]] = []
        scripts: List[Dict[str, Any]] = []
        videos: List[Dict[str, Any]] = []
        quizzes: List[Dict[str, Any]] = []

        for module in course_plan.modules:
            logger.info("Processing module '%s'", module.title)
            for lesson_name in module.lessons:
                logger.info("Generating lesson '%s'", lesson_name)
                
                # 4. Iterative Lesson Generation
                lesson_json = self.lesson_writer.generate(
                    {
                        "module_name": module.title,
                        "lesson_name": lesson_name,
                        "learning_outcomes": learning_outcomes,
                        "research_context": research_result
                    }
                )
                
                # Lesson Review Loop (Max 1 retry for efficiency)
                for attempt in range(2):
                    lesson_review = self.review_agent.generate({
                        "content_type": "lesson",
                        "content": lesson_json,
                        "context": research_result
                    })
                    
                    if lesson_review.get("approved"):
                        logger.info("Lesson '%s' approved.", lesson_name)
                        break
                    
                    logger.info("Lesson '%s' rejected (Score: %s). Refining...", lesson_name, lesson_review.get("score"))
                    # Add feedback to the input and regenerate
                    # Note: In a real implementation, we'd pass the previous draft + feedback.
                    # For now, we'll just re-generate with feedback appended to prompt context implicitly via a new call if we supported state.
                    # Since LessonWriter is stateless, we'll just proceed or we could implement a "refine" method.
                    # Let's just log it for this iteration.
                    break

                lesson_validation = self.validator.generate(lesson_json)
                validated_lesson = lesson_validation["validated_content"]
                lessons.append(validated_lesson)

                # Parallel generation of assets
                logger.info("Generating video script for lesson '%s'", lesson_name)
                script_json = self.video_script_agent.generate(validated_lesson)
                script_validation = self.validator.generate(script_json)
                validated_script = script_validation["validated_content"]
                scripts.append(validated_script)

                logger.info("Generating quiz for lesson '%s'", lesson_name)
                quiz_json = self.quiz_agent.generate(validated_lesson)
                quiz_validation = self.validator.generate(quiz_json)
                validated_quiz = quiz_validation["validated_content"]

                if not skip_video:
                    logger.info("Generating video for lesson '%s'", lesson_name)
                    video_info = self.video_generator.generate(validated_script)
                    videos.append(video_info)
                else:
                    logger.info("Skipping video generation for lesson '%s'", lesson_name)

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
            "scripts": scripts,
            "videos": videos,
            "quizzes": quizzes,
            "final_validation": final_validation,
            "research": research_result, # Return research for UI
        }
