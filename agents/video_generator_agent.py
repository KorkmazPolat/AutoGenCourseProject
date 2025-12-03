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
            
            # Combine full script to let AI decide segmentation
            full_script_text = "\n".join([s.text for s in script.scenes])
            
            # Use LLM to design the visual presentation AND re-segment narration
            design_prompt = f"""
            You are a professional video producer and instructional designer.
            
            Input Script:
            "{full_script_text}"
            
            Task:
            1. Re-organize this script into an engaging video presentation.
            2. Determine the OPTIMAL number of slides. 
               - Aim for 1 slide per 45-60 seconds of narration (approx 100-130 words).
               - Avoid creating too many short slides. Group related ideas together.
               - Minimum 3 slides, Maximum 10 slides for this length.
            3. For each slide:
               a. Provide the 'narration' text (you can slightly smooth/edit the original script for better flow, but keep the core message).
               b. Design the 'slide' visuals (heading, layout, content_blocks).
            
            Output JSON structure:
            {{
              "segments": [
                {{
                  "narration": "The spoken script for this segment...",
                  "slide": {{
                    "heading": "Short Punchy Title",
                    "content_blocks": [
                      {{ "type": "bullets", "items": ["Key point 1", "Key point 2"] }}
                    ]
                  }}
                }}
              ]
            }}
            
            Slide Design Rules:
            - "content_blocks" MUST NEVER BE EMPTY.
            - Use "bullets" (list of strings), "quote" (text), "callout" (text), or "checklist" (list).
            - Content must be concise (max 8 words per bullet).
            """
            
            try:
                design_completion = self._llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": design_prompt}],
                    response_format={"type": "json_object"}
                )
                design_data = json.loads(design_completion.choices[0].message.content)
                segments = design_data.get("segments", [])
            except Exception as e:
                print(f"Slide design failed, falling back to simple mode: {e}")
                segments = []

            presentation = []
            narration = []
            
            # Fallback if AI fails to generate segments
            if not segments:
                # Create one big segment or split by paragraphs
                segments = [{"narration": full_script_text, "slide": {"heading": script.lesson, "content_blocks": []}}]

            for i, segment in enumerate(segments, 1):
                slide_design = segment.get("slide", {})
                script_text = segment.get("narration", "")
                
                # Ensure content blocks
                blocks = slide_design.get("content_blocks", [])
                if not blocks:
                    # Smart fallback
                    sentences = [s.strip() for s in script_text.split('.') if len(s.strip()) > 10]
                    items = sentences[:4] if sentences else [script_text[:50] + "..."]
                    blocks = [{"type": "bullets", "items": items}]

                presentation.append({
                    "page": i,
                    "heading": slide_design.get("heading", script.lesson),
                    "content": "", 
                    "content_blocks": blocks
                })
                
                narration.append({
                    "script": script_text
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
                voice="alloy", 
                tts_model="gpt-4o-mini-tts" 
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

