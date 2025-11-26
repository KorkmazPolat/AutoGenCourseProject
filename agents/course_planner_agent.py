from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from schemas.agent_io import CoursePlannerInput
from schemas.course_plan import CoursePlan, Module


class CoursePlannerAgent(BaseAgent):
    name = "course_planner"

    def _normalize_llm_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt common LLM plan formats into the strict CoursePlan schema.

        Expected target schema:
        {
          "title": str,
          "modules": [
            {"title": str, "lessons": [str, ...]},
            ...
          ]
        }
        """
        # Already looks like our schema
        if "title" in payload and "modules" in payload:
            return payload

        title = payload.get("title") or payload.get("courseTitle") or payload.get("course_title") or "Course"
        modules_raw: List[Dict[str, Any]] = payload.get("modules") or payload.get("courseModules") or []
        modules: List[Module] = []

        for module in modules_raw:
            module_title = module.get("title") or module.get("moduleTitle") or "Module"
            lessons_raw = module.get("lessons") or module.get("moduleLessons") or []
            lesson_titles: List[str] = []
            for lesson in lessons_raw:
                if isinstance(lesson, str):
                    lesson_titles.append(lesson)
                elif isinstance(lesson, dict):
                    name = (
                        lesson.get("title")
                        or lesson.get("lessonTitle")
                        or lesson.get("name")
                    )
                    if name:
                        lesson_titles.append(name)
            if not lesson_titles:
                continue
            modules.append(Module(title=module_title, lessons=lesson_titles))

        return CoursePlan(title=title, modules=modules).to_json()

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        planner_input = CoursePlannerInput.parse_obj(input_json)
        template = self.load_template("course_plan.jinja")
        prompt = template.render(
            learning_outcomes=planner_input.learning_outcomes,
            num_modules=planner_input.num_modules,
            num_lessons=planner_input.num_lessons,
            feedback=planner_input.feedback,
            previous_plan=planner_input.previous_plan
        )
        llm_result = self.call_llm(prompt)
        raw_payload = self.validate_json(llm_result)
        normalized_payload = self._normalize_llm_plan(raw_payload)
        course_plan = CoursePlan.parse_obj(normalized_payload)
        return course_plan.to_json()
