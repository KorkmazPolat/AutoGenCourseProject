from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from schemas.agent_io import LessonWriterInput
from schemas.lesson_content import LessonContent


class LessonWriterAgent(BaseAgent):
    name = "lesson_writer"

    def _normalize_llm_lesson(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt common LLM lesson formats into the strict LessonContent schema.

        Target schema:
        {
          "title": str,
          "text": str,
          "summary": str
        }
        """
        if all(key in payload for key in ("title", "text", "summary")):
            return payload

        title = (
            payload.get("title")
            or payload.get("lessonTitle")
            or payload.get("lesson_title")
            or payload.get("name")
            or payload.get("lesson")
            or "Lesson"
        )

        # Build main text from possible fields
        text_parts: List[str] = []

        direct_body = (
            payload.get("text")
            or payload.get("body")
            or payload.get("content")
            or payload.get("lessonText")
        )
        if isinstance(direct_body, str):
            text_parts.append(direct_body)

        # If there are sections, concatenate their text
        sections = payload.get("sections") or payload.get("lessonSections")
        if isinstance(sections, list):
            for section in sections:
                if isinstance(section, dict):
                    heading = section.get("title") or section.get("heading")
                    body = section.get("text") or section.get("body") or section.get("content")
                    piece_parts: List[str] = []
                    if heading:
                        piece_parts.append(str(heading))
                    if body:
                        piece_parts.append(str(body))
                    if piece_parts:
                        text_parts.append("\n\n".join(piece_parts))

        text = "\n\n".join(part for part in text_parts if part).strip()

        summary = (
            payload.get("summary")
            or payload.get("recap")
            or payload.get("overview")
            or payload.get("abstract")
            or ""
        )

        # Ensure we always have reasonably rich lesson text so that downstream
        # video/reading views have real content to display.
        final_summary = summary or title
        final_text = text or final_summary or title

        # If the body is very short, expand it into a few simple paragraphs
        # based on the summary and title. This avoids empty lessons when the
        # LLM returns only a brief sentence.
        if len(final_text.strip()) < 200 and final_summary:
            paragraphs: List[str] = []
            paragraphs.append(f"In this lesson, you will explore: {final_summary}")
            paragraphs.append(
                f"We start by introducing {title}, then walk through practical examples "
                "and simple explanations so that even beginners can follow."
            )
            paragraphs.append(
                "By the end of this lesson, you should feel more confident applying "
                f"what you have learned about {title}."
            )
            final_text = "\n\n".join(paragraphs)

        normalized = {
            "title": title,
            "text": final_text,
            "summary": final_summary,
        }
        return normalized

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        writer_input = LessonWriterInput.parse_obj(input_json)
        template = self.load_template("lesson_writer.jinja")
        prompt = template.render(
            module_name=writer_input.module_name,
            lesson_name=writer_input.lesson_name,
            learning_outcomes=writer_input.learning_outcomes,
        )
        llm_result = self.call_llm(prompt)

        # Be defensive: if the LLM returns non-JSON text, fall back to a
        # minimal LessonContent instead of raising and breaking the pipeline.
        try:
            raw_payload = self.validate_json(llm_result)
            normalized_payload = self._normalize_llm_lesson(raw_payload)
            lesson = LessonContent.parse_obj(normalized_payload)
            return lesson.to_json()
        except Exception:
            # Fallback: treat raw text as the lesson body
            text = str(llm_result).strip()
            title = writer_input.lesson_name or "Lesson"
            summary = text.split("\n", 1)[0][:200] if text else title
            lesson = LessonContent(
                title=title,
                text=text or summary or title,
                summary=summary,
            )
            return lesson.to_json()
