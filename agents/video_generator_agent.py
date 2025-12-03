from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

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
            
            # Use LLM to design the visual presentation based on the script
            # This ensures variety and professional content structure
            design_prompt = f"""
            You are a professional presentation designer.
            Create a JSON structure for a video presentation based on the following narration script.
            
            Lesson Title: {script.lesson}
            
            Script Sections:
            {json.dumps([s.text for s in script.scenes], indent=2)}
            
            For EACH script section, design a corresponding slide.
            Return a JSON object with a "slides" key containing a list of slide objects.
            
            Each slide object must have:
            - "heading": A short, punchy headline (max 5 words).
            - "layout": One of ["bullets", "quote", "callout", "checklist", "example"].
            - "content_blocks": A list of blocks to render.
                - For "bullets": {{ "type": "bullets", "items": ["point 1", "point 2"] }}
                - For "quote": {{ "type": "quote", "text": "The quote text" }}
                - For "callout": {{ "type": "callout", "text": "Important tip or note" }}
                - For "checklist": {{ "type": "checklist", "items": ["step 1", "step 2"] }}
                - For "example": {{ "type": "example", "title": "Example Title", "text": "The example content" }}
            
            Rules:
            1. Vary the layouts! Do not just use bullets every time.
            2. Summarize the script into key points. Do NOT copy the full script onto the slide.
            3. Keep text concise and readable.
            """
            
            try:
                design_completion = self._llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": design_prompt}],
                    response_format={"type": "json_object"}
                )
                design_data = json.loads(design_completion.choices[0].message.content)
                designed_slides = design_data.get("slides", [])
            except Exception as e:
                print(f"Slide design failed, falling back to simple mode: {e}")
                designed_slides = []

            presentation = []
            narration = []
            
            for i, scene in enumerate(script.scenes, 1):
                # Get designed slide or fallback
                if i <= len(designed_slides):
                    slide_design = designed_slides[i-1]
                    presentation.append({
                        "page": i,
                        "heading": slide_design.get("heading", script.lesson),
                        "content": "", # Legacy field
                        "content_blocks": slide_design.get("content_blocks", [])
                    })
                else:
                    # Fallback
                    presentation.append({
                        "page": i,
                        "heading": script.lesson,
                        "content": scene.text[:100] + "..."
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

