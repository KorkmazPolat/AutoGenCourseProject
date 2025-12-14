from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import os

from agents.base_agent import BaseAgent
from schemas.video_script import VideoScript

# Gemini Import
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class VideoGeneratorAgent(BaseAgent):
    name = "video_generator"

    def generate(self, input_json: Dict[str, Any], engine: str = "openai") -> Dict[str, Any]:
        """
        Generates a video from the input script. 
        'engine' can be "openai" or "gemini".
        """
        script = VideoScript.parse_obj(input_json)
        
        project_root = Path(__file__).resolve().parents[1]
        output_dir = project_root / "course_material_service" / "static" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        import uuid
        filename = f"lesson_{uuid.uuid4().hex[:8]}.mp4"
        video_path = output_dir / filename
        
        if not self._llm_client:
             return {"error": "Video generation requires an OpenAI client (LLM config)."}

        try:
            from course_material_service.video_builder import generate_video_from_script
            
            full_script_text = "\n".join([s.text for s in script.scenes])
            
            # IMPROVED DESIGN PROMPT
            design_prompt = f"""
            You are a Hollywood-grade video presentation designer.
            
            Input Script:
            "{full_script_text}"
            
            GOAL: Create a VISUALLY DENSE, professional presentation.  
            
            RULES:
            1. Title Slide: Create a specific first slide with `layout: "hero"` or simply a big Heading.
            2. Visual Density: Do NOT create empty slides. Every slide must have rich `content_blocks`.
            3. Structure: 
               - Aim for 1 slide per 40-60 seconds of speech.
               - Group related concepts.
            4. Content Blocks:
               - Use 'key_takeaway' for main points (it renders as a glowing box).
               - Use 'subheading' to divide sections on a single slide.
               - Use 'diagram' placeholders for visual concepts.
               - Use 'code' for technical concepts.
               - Use 'table' for comparisons.
            
            Supported content_block types: 
            - 'bullets': {{ "type": "bullets", "items": [...] }}
            - 'subheading': {{ "type": "subheading", "text": "..." }} 
            - 'key_takeaway': {{ "type": "key_takeaway", "text": "..." }} 
            - 'callout': {{ "type": "callout", "text": "...", "style": "info|warning|tip" }}
            - 'diagram': {{ "type": "diagram", "caption": "..." }} 
            - 'quote': {{ "type": "quote", "text": "..." }}
            - 'code': {{ "type": "code", "code": "...", "language": "python" }}
            - 'table': {{ "type": "table", "rows": [["Col1", "Col2"], ["Val1", "Val2"]] }}

            Output JSON structure:
            {{
              "segments": [
                {{
                  "narration": "Script text...",
                  "slide": {{
                    "heading": "Slide Title",
                    "content_blocks": [ ... ]
                  }}
                }}
              ]
            }}
            """
            
            segments = []
            
            # --- ENGINE SELECTION ---
            if engine == "gemini" and HAS_GEMINI:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    print("Gemini API Key missing, falling back to OpenAI.")
                    # Fallback logic below
                else:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel("gemini-2.0-flash-exp") 
                        # Or 1.5-flash if 2.0 not available, user asked for 2.0 optimization before
                        
                        response = model.generate_content(
                            design_prompt,
                            generation_config={"response_mime_type": "application/json"}
                        )
                        design_data = json.loads(response.text)
                        segments = design_data.get("segments", [])
                        print("SUCCESS: Used Gemini for Video Design")
                    except Exception as e:
                        print(f"Gemini generation failed: {e}. Falling back to OpenAI.")
            
            # Fallback or Default to OpenAI
            if not segments:
                try:
                    design_completion = self._llm_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": design_prompt}],
                        response_format={"type": "json_object"}
                    )
                    content = design_completion.choices[0].message.content
                    design_data = json.loads(content)
                    segments = design_data.get("segments", [])
                except Exception as e:
                    print(f"Slide design failed: {e}")
                    segments = []

            presentation = []
            narration = []
            
            if not segments:
                 segments = [{"narration": full_script_text, "slide": {"heading": script.lesson, "content_blocks": [{"type": "bullets", "items": ["Key Concept 1", "Key Concept 2"]}]}}]

            for i, segment in enumerate(segments, 1):
                slide_design = segment.get("slide", {})
                script_text = segment.get("narration", "")
                blocks = slide_design.get("content_blocks", [])
                
                # Sanity check for empty blocks
                if not blocks:
                    blocks = [{"type": "bullets", "items": ["Key Point"]}]

                presentation.append({
                    "page": i,
                    "heading": slide_design.get("heading", "Lesson Part " + str(i)),
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
                "call_to_action": "End of Section"
            }
            
            generate_video_from_script(
                video_payload=video_payload,
                output_path=video_path,
                client=self._llm_client,
                voice="alloy", 
                tts_model="gpt-4o-mini-tts" 
            )
            
            return {
                "video_file": str(video_path),
                "captions_file": str(video_path.with_suffix(".vtt")),
                "chapters_file": str(video_path.with_name(video_path.stem + ".chapters.vtt"))
            }
            
        except Exception as e:
            print(f"Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
