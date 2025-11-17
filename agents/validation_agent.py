from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from agents.base_agent import BaseAgent
from validators.lesson_validator import validate_lesson
from validators.plan_validator import validate_plan
from validators.quiz_validator import validate_quiz
from validators.script_validator import validate_script

logger = logging.getLogger(__name__)


class ValidationAgent(BaseAgent):
    name = "validation"
    max_fix_iterations: int = 2

    def _run_structural_validators(self, content: Dict[str, Any]) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        ok = True

        if "modules" in content:
            plan_ok, plan_issues = validate_plan(content)
            ok = ok and plan_ok
            issues.extend(plan_issues)

        if {"title", "text"} <= content.keys():
            lesson_ok, lesson_issues = validate_lesson(content)
            ok = ok and lesson_ok
            issues.extend(lesson_issues)

        if "scenes" in content:
            script_ok, script_issues = validate_script(content)
            ok = ok and script_ok
            issues.extend(script_issues)

        if "questions" in content:
            quiz_ok, quiz_issues = validate_quiz(content)
            ok = ok and quiz_ok
            issues.extend(quiz_issues)

        return ok, issues

    def _run_llm_validation(self, content: Dict[str, Any]) -> Dict[str, Any]:
        template = self.load_template("validation.jinja")
        prompt = template.render(content=content)
        llm_result = self.call_llm(prompt)
        return self.validate_json(llm_result)

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        current = input_json
        all_issues: List[str] = []

        for iteration in range(self.max_fix_iterations + 1):
            structural_ok, structural_issues = self._run_structural_validators(current)
            all_issues.extend(structural_issues)
            if structural_ok:
                status = "ok" if iteration == 0 else "fixed"
                return {
                    "status": status,
                    "validated_content": current,
                    "issues": all_issues,
                }

            if iteration >= self.max_fix_iterations:
                logger.warning("Validation still failing after max iterations.")
                return {
                    "status": "fixed_with_issues",
                    "validated_content": current,
                    "issues": all_issues,
                }

            logger.info("Validation iteration %s failed, sending to LLM for correction", iteration)
            current = self._run_llm_validation(current)

