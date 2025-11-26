from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agents.base_agent import BaseAgent
from schemas.video_script import VideoScript


class VideoGeneratorAgent(BaseAgent):
    name = "video_generator"

    def generate(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        script = VideoScript.parse_obj(input_json)
        # Ensure the static videos directory exists
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "course_material_service" / "static" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique filename
        import uuid
        filename = f"lesson_{uuid.uuid4().hex[:8]}.mp4"
        video_path = output_dir / filename
        
        # Call the video builder
        # Note: We need an OpenAI client for TTS. BaseAgent has one.
        if not self._llm_client:
             return {"error": "Video generation requires an OpenAI client (LLM config)."}

        try:
            from course_material_service.video_builder import generate_video_from_script
            
            # Map VideoScript schema to what video_builder expects
            # VideoScript: { "lesson": str, "scenes": [{"text": str, "duration": int}] }
            # video_builder expects: { "presentation": [...], "narration": [...] }
            # We will synthesize a simple presentation from the scenes.
            
            presentation = []
            narration = []
            
            for i, scene in enumerate(script.scenes, 1):
                presentation.append({
                    "page": i,
                    "heading": script.lesson,
                    "content": scene.text[:100] + "..." # Show snippet on slide
                })
                narration.append({
                    "script": scene.text
                })
                
            video_payload = {
                "presentation": presentation,
                "narration": narration,
                "hook": f"Lesson: {script.lesson}",
                "call_to_action": "Thanks for watching!"
            }
            
            generate_video_from_script(
                video_payload=video_payload,
                output_path=video_path,
                client=self._llm_client,
                voice="alloy", # Default
                tts_model="gpt-4o-mini-tts" # Default
            )
            
            # Return the absolute path, but the service will need to serve it relative to static
            return {
                "video_file": str(video_path),
                "captions_file": str(video_path.with_suffix(".vtt")),
                "chapters_file": str(video_path.with_name(video_path.stem + ".chapters.vtt"))
            }
            
        except Exception as e:
            # Log error and return empty/placeholder to avoid crashing the whole pipeline
            print(f"Video generation failed: {e}")
            return {"error": str(e)}

