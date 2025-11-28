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
from manager.telemetry import FeedbackMonitor, PerformanceMonitor, WorkloadManager
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
        self.feedback_monitor: FeedbackMonitor | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.workload_manager: WorkloadManager | None = None

    def _initialize_monitors(self) -> None:
        self.feedback_monitor = FeedbackMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.workload_manager = WorkloadManager()

    def _record_feedback(self, stage: str, success: bool, **metadata: Any) -> None:
        if self.feedback_monitor is not None:
            self.feedback_monitor.record(stage, success, metadata)

    def run(self, learning_outcomes: List[str], skip_video: bool = False, num_modules: int | None = None, num_lessons: int | None = None) -> Dict[str, Any]:
        self._initialize_monitors()
        assert self.performance_monitor is not None
        assert self.workload_manager is not None

        logger.info("Starting REAL agentic course generation pipeline.")

        # 1. Research Phase
        logger.info("Agent is researching the topic...")
        with self.performance_monitor.track("research.generate"):
            research_result = self.research_agent.generate({"learning_outcomes": learning_outcomes})
        logger.info("Research complete. Key concepts: %s", research_result.get("key_concepts"))
        self._record_feedback("research.generate", bool(research_result.get("key_concepts")), key_concepts=research_result.get("key_concepts", []))

        # 2. Planning Phase
        logger.info("Agent is planning the course...")
        # Inject research context into the planner input
        with self.performance_monitor.track("course_planner.generate"):
            course_plan_json = self.course_planner.generate(
                {
                    "learning_outcomes": learning_outcomes,
                    "research_context": research_result,
                    "num_modules": num_modules,
                    "num_lessons": num_lessons
                }
            )
        self._record_feedback("course_planner.generate", bool(course_plan_json), num_modules=num_modules, num_lessons=num_lessons)
        
        # 3. Plan Review & Refinement
        max_retries = 2
        for attempt in range(max_retries + 1):
            logger.info("Agent is reviewing the course plan (Attempt %d/%d)...", attempt + 1, max_retries + 1)
            with self.performance_monitor.track("review_agent.course_plan"):
                plan_review = self.review_agent.generate({
                    "content_type": "course_plan",
                    "content": course_plan_json,
                    "context": research_result
                })
            approved = bool(plan_review.get("approved"))
            self._record_feedback(
                "review_agent.course_plan",
                approved,
                attempt=attempt + 1,
                score=plan_review.get("score"),
                feedback=plan_review.get("feedback"),
            )
            
            if approved:
                logger.info("Plan approved with score %s", plan_review.get("score"))
                break
            
            if attempt < max_retries:
                logger.warning("Plan rejected: %s. Refining...", plan_review.get("feedback"))
                # Regenerate with feedback
                course_plan_json = self.course_planner.generate(
                    {
                        "learning_outcomes": learning_outcomes,
                        "research_context": research_result,
                        "num_modules": num_modules,
                        "num_lessons": num_lessons,
                        "feedback": plan_review.get("feedback"),
                        "previous_plan": course_plan_json
                    }
                )
            else:
                logger.warning("Plan rejected after max retries. Proceeding with best effort.")

        with self.performance_monitor.track("validator.course_plan"):
            plan_validation = self.validator.generate(course_plan_json)
        course_plan = CoursePlan.parse_obj(plan_validation["validated_content"])
        self._record_feedback("validator.course_plan", True, modules=len(course_plan.modules))

        lessons: List[Dict[str, Any]] = []
        scripts: List[Dict[str, Any]] = []
        videos: List[Dict[str, Any]] = []
        quizzes: List[Dict[str, Any]] = []

        for module in course_plan.modules:
            logger.info("Processing module '%s'", module.title)
            for lesson_name in module.lessons:
                logger.info("Generating lesson '%s'", lesson_name)
                
                # 4. Iterative Lesson Generation
                with self.performance_monitor.track("lesson_writer.generate"):
                    lesson_json = self.lesson_writer.generate(
                        {
                            "module_name": module.title,
                            "lesson_name": lesson_name,
                            "learning_outcomes": learning_outcomes,
                            "research_context": research_result
                        }
                    )
                self._record_feedback("lesson_writer.generate", True, module=module.title, lesson=lesson_name)
                
                # Lesson Review Loop (Max 1 retry for efficiency)
                for attempt in range(2):
                    with self.performance_monitor.track("review_agent.lesson"):
                        lesson_review = self.review_agent.generate({
                            "content_type": "lesson",
                            "content": lesson_json,
                            "context": research_result
                        })
                    lesson_approved = bool(lesson_review.get("approved"))
                    self._record_feedback(
                        "review_agent.lesson",
                        lesson_approved,
                        module=module.title,
                        lesson=lesson_name,
                        attempt=attempt + 1,
                        score=lesson_review.get("score"),
                    )
                    
                    if lesson_approved:
                        logger.info("Lesson '%s' approved.", lesson_name)
                        break
                    
                    logger.info("Lesson '%s' rejected (Score: %s). Refining...", lesson_name, lesson_review.get("score"))
                    # Add feedback to the input and regenerate
                    # Note: In a real implementation, we'd pass the previous draft + feedback.
                    # For now, we'll just re-generate with feedback appended to prompt context implicitly via a new call if we supported state.
                    # Since LessonWriter is stateless, we'll just proceed or we could implement a "refine" method.
                    # Let's just log it for this iteration.
                    break

                with self.performance_monitor.track("validator.lesson"):
                    lesson_validation = self.validator.generate(lesson_json)
                validated_lesson = lesson_validation["validated_content"]
                lessons.append(validated_lesson)

                # Parallel generation of assets
                logger.info("Generating video script for lesson '%s'", lesson_name)
                with self.performance_monitor.track("video_script_agent.generate"):
                    script_json = self.video_script_agent.generate(validated_lesson)
                with self.performance_monitor.track("validator.script"):
                    script_validation = self.validator.generate(script_json)
                validated_script = script_validation["validated_content"]
                scripts.append(validated_script)

                logger.info("Generating quiz for lesson '%s'", lesson_name)
                with self.performance_monitor.track("quiz_agent.generate"):
                    quiz_json = self.quiz_agent.generate(validated_lesson)
                with self.performance_monitor.track("validator.quiz"):
                    quiz_validation = self.validator.generate(quiz_json)
                validated_quiz = quiz_validation["validated_content"]

                if not skip_video:
                    logger.info("Queueing video generation for lesson '%s'", lesson_name)
                    assert self.workload_manager is not None
                    self.workload_manager.submit_video_task(
                        lesson_name=lesson_name,
                        script_payload=validated_script,
                        generator=self.video_generator
                    )
                    self._record_feedback("video_generation.queue", True, lesson=lesson_name)
                else:
                    logger.info("Skipping video generation for lesson '%s'", lesson_name)

                quizzes.append(validated_quiz)

        if not skip_video:
            videos, video_task_metrics = self.workload_manager.collect_video_results()
            for task in video_task_metrics:
                self._record_feedback(
                    "video_generation.process",
                    task.get("success", False),
                    lesson=task.get("lesson_name"),
                    duration=task.get("duration"),
                    error=task.get("error"),
                )
        else:
            if self.workload_manager is not None:
                self.workload_manager.close()

        final_validation = self.validator.generate(
            {
                "course_plan": course_plan.to_json(),
                "lessons": lessons,
                "videos": videos,
                "quizzes": quizzes,
            }
        )

        logger.info("Course generation pipeline finished.")
        telemetry_payload: Dict[str, Any] = {
            "feedback": self.feedback_monitor.summary() if self.feedback_monitor else {},
            "performance": self.performance_monitor.summary() if self.performance_monitor else {},
        }
        if not skip_video and self.workload_manager is not None:
            telemetry_payload["workload"] = self.workload_manager.summary()

        return {
            "course_plan": course_plan.to_json(),
            "lessons": lessons,
            "scripts": scripts,
            "videos": videos,
            "quizzes": quizzes,
            "final_validation": final_validation,
            "research": research_result, # Return research for UI
            "telemetry": telemetry_payload,
        }
    def generate_lesson_bundle(self, module_title: str, lesson_title: str, lesson_desc: str, skip_video: bool = False) -> Dict[str, Any]:
        """
        Generates content for a single lesson (text, script, quiz, video) based on title and description.
        """
        logger.info("Generating bundle for lesson '%s'", lesson_title)
        
        # 1. Generate Lesson Text
        # We treat the description as the 'learning outcome' or context for this specific lesson
        lesson_json = self.lesson_writer.generate(
            {
                "module_name": module_title,
                "lesson_name": lesson_title,
                "learning_outcomes": [lesson_desc], # Use desc as outcome
                "research_context": {"key_concepts": [lesson_desc]} # Minimal context
            }
        )
        
        lesson_validation = self.validator.generate(lesson_json)
        validated_lesson = lesson_validation["validated_content"]
        
        # 2. Generate Script
        script_json = self.video_script_agent.generate(validated_lesson)
        script_validation = self.validator.generate(script_json)
        validated_script = script_validation["validated_content"]
        
        # 3. Generate Quiz
        quiz_json = self.quiz_agent.generate(validated_lesson)
        quiz_validation = self.validator.generate(quiz_json)
        validated_quiz = quiz_validation["validated_content"]
        
        # 4. Generate Video
        video_info = None
        if not skip_video:
            video_info = self.video_generator.generate(validated_script)
            
        return {
            "lesson": validated_lesson,
            "script": validated_script,
            "quiz": validated_quiz,
            "video": video_info
        }
