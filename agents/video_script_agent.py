from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from schemas.lesson_content import LessonContent
from schemas.video_script import Scene, VideoScript


class VideoScriptAgent(BaseAgent):
    name = "video_script"

    def _normalize_llm_script(self, payload: Dict[str, Any], lesson_title: str) -> Dict[str, Any]:
        """
        Adapt common LLM video script formats into the strict VideoScript schema:
        {
          "lesson": str,
          "scenes": [{"text": str, "duration": int}, ...]
        }
        """
        if "lesson" in payload and "scenes" in payload:
            return payload

        lesson_name = payload.get("lesson") or payload.get("lessonTitle") or lesson_title

        scenes_raw = (
            payload.get("scenes")
            or payload.get("script")
            or payload.get("sections")
            or payload.get("beats")
            or []
        )
        scenes: List[Scene] = []

        if isinstance(scenes_raw, list):
            for idx, scene in enumerate(scenes_raw):
                if isinstance(scene, str):
                    scenes.append(Scene(text=scene, duration=10))
                elif isinstance(scene, dict):
                    text = (
                        scene.get("text")
                        or scene.get("content")
                        or scene.get("narration")
                        or scene.get("sceneText")
                        or ""
                    )
                    duration = scene.get("duration") or scene.get("seconds") or scene.get("durationSeconds")
                    try:
                        duration_int = int(duration) if duration is not None else 10
                    except (TypeError, ValueError):
                        duration_int = 10
                    if text:
                        scenes.append(Scene(text=str(text), duration=duration_int))

        if not scenes:
            # Fallback: single-scene script from any text field
            fallback_text = (
                payload.get("text")
                or payload.get("body")
                or payload.get("script")
                or ""
            )
            if fallback_text:
                scenes.append(Scene(text=str(fallback_text), duration=60))

        script = VideoScript(lesson=str(lesson_name), scenes=scenes or [Scene(text=lesson_name, duration=30)])
        return script.to_json()

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        lesson = LessonContent.parse_obj(input_json)
        template = self.load_template("video_script.jinja")
        prompt = template.render(lesson=lesson.dict())
        llm_result = self.call_llm(prompt)
        raw_payload = self.validate_json(llm_result)
        normalized_payload = self._normalize_llm_script(raw_payload, lesson.title)
        script = VideoScript.parse_obj(normalized_payload)
        return script.to_json()
