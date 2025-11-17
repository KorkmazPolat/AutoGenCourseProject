from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from schemas.lesson_content import LessonContent
from schemas.quiz import Question, Quiz


class QuizAgent(BaseAgent):
    name = "quiz"

    def _normalize_llm_quiz(self, payload: Any, lesson_title: str) -> Dict[str, Any]:
        """
        Adapt common LLM quiz formats into the strict Quiz schema:
        {
          "lesson": str,
          "questions": [
            {"question": str, "options": [str, ...], "correct_index": int},
            ...
          ]
        }
        """
        # If the model returned a bare list of questions, wrap it.
        if isinstance(payload, list):
            payload = {"questions": payload}

        if isinstance(payload, dict) and "lesson" in payload and "questions" in payload:
            return payload

        lesson_name = (
            payload.get("lesson") if isinstance(payload, dict) else None
        ) or (payload.get("lessonTitle") if isinstance(payload, dict) else None) or lesson_title
        questions_raw: Any = []
        if isinstance(payload, dict):
            questions_raw = (
                payload.get("questions")
                or payload.get("items")
                or payload.get("quiz")
                or []
            )
        questions: List[Question] = []

        if isinstance(questions_raw, list):
            for q in questions_raw:
                if isinstance(q, str):
                    # No options provided; skip or create dummy options
                    options = ["True", "False"]
                    questions.append(Question(question=q, options=options, correct_index=0))
                elif isinstance(q, dict):
                    text = (
                        q.get("question")
                        or q.get("prompt")
                        or q.get("text")
                        or ""
                    )
                    options = q.get("options") or q.get("choices") or q.get("answers") or []
                    if isinstance(options, dict):
                        # e.g. {"A": "...", "B": "..."} -> ["...", "..."]
                        options = list(options.values())
                    options = [str(o) for o in options]

                    correct_index = q.get("correct_index")
                    if correct_index is None:
                        # Try to infer from "correctOption", "answer", etc.
                        correct_option = (
                            q.get("correctOption")
                            or q.get("answer")
                            or q.get("correctAnswer")
                        )
                        if correct_option in options:
                            correct_index = options.index(correct_option)
                        else:
                            correct_index = 0
                    try:
                        correct_index_int = int(correct_index)
                    except (TypeError, ValueError):
                        correct_index_int = 0

                    if text and options:
                        questions.append(
                            Question(
                                question=str(text),
                                options=options,
                                correct_index=correct_index_int % len(options),
                            )
                        )

        if not questions:
            # Fallback: at least one placeholder question
            questions.append(
                Question(
                    question=f"What is the main idea of '{lesson_name}'?",
                    options=["It explains the lesson topic.", "It is unrelated."],
                    correct_index=0,
                )
            )

        quiz = Quiz(lesson=str(lesson_name), questions=questions)
        return quiz.to_json()

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        lesson = LessonContent.parse_obj(input_json)
        template = self.load_template("quiz.jinja")
        prompt = template.render(lesson=lesson.dict())
        llm_result = self.call_llm(prompt)
        try:
            raw_payload = self.validate_json(llm_result)
            normalized_payload = self._normalize_llm_quiz(raw_payload, lesson.title)
            quiz = Quiz.parse_obj(normalized_payload)
            return quiz.to_json()
        except Exception:
            # Fallback: synthesize a simple quiz directly from the lesson content
            fallback_questions = [
                Question(
                    question=f"What is the main goal of the lesson '{lesson.title}'?",
                    options=[
                        "To understand the key concepts described in the lesson.",
                        "To learn an unrelated topic.",
                        "To memorize random facts.",
                        "There is no clear goal.",
                    ],
                    correct_index=0,
                ),
                Question(
                    question="Which statement best summarizes the lesson?",
                    options=[
                        lesson.summary,
                        "It briefly mentions the topic without details.",
                        "It focuses on an unrelated domain.",
                        "It only lists questions.",
                    ],
                    correct_index=0,
                ),
                Question(
                    question="How should a learner apply this lesson?",
                    options=[
                        "By practicing the concepts in real examples.",
                        "By ignoring the ideas in practice.",
                        "By only reading without interaction.",
                        "By using it in an unrelated field.",
                    ],
                    correct_index=0,
                ),
            ]
            quiz = Quiz(lesson=lesson.title, questions=fallback_questions)
            return quiz.to_json()
