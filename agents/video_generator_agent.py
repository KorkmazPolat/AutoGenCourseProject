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

    def generate(self, input_json: Dict[str, Any], engine: str = "openai", duration_minutes: int = 5) -> Dict[str, Any]:
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
            from course_material_service.video_builder import generate_slides_and_audio
            
            full_script_text = "\n".join([s.text for s in script.scenes])
            
            # Calculate target slides based on 3 slides per minute rule
            # Ensure a reasonable bounds (e.g. for 2 mins = 6 slides)
            target_slides = max(3, duration_minutes * 3)
            
            # IMPROVED DESIGN PROMPT
            design_prompt = f"""
            You are a Hollywood-grade video presentation designer using the advanced capabilities of Gemini 2.0.
            
            Input Script:
            "{full_script_text}"
            
            Target Duration: {duration_minutes} minutes.
            MANDATORY CONSTRAINT: You MUST generate EXACTLY {target_slides} slides. No more, no less.
            
            GOAL: Create a VISUALLY DENSE, professional presentation with "amazing design".  
            
            RULES:
            1. Title Slide: Slide 1 MUST be the title.
            2. Pacing: You have exactly {target_slides} slides to cover the entire script. 
               - Do NOT cram too much text on one slide if it violates the pacing.
               - Do NOT make 20 slides for a short script. Stick to {target_slides}.
            3. Visual Density: Do NOT create empty slides. Every slide must have rich `content_blocks`.
            4. Structure: 
               - Distribute the script evenly across the {target_slides} slides.
               - Ensure every slide has distinct narration segments.
            5. Content Blocks:
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
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Gemini/Google API Key missing for Video Generator.")

                # FORCE REST transport to avoid gRPC/DNS issues
                genai.configure(api_key=api_key, transport="rest")
                model = genai.GenerativeModel("gemini-2.0-flash-exp") 
                
                response = model.generate_content(
                    design_prompt,
                    generation_config=genai.types.GenerationConfig(
                        candidate_count=1,
                        max_output_tokens=8192,
                        temperature=0.7,
                        response_mime_type="application/json"
                    )
                )
                text_content = response.text
                
                # Basic cleanup
                if text_content.strip().startswith("```json"):
                    text_content = text_content.strip()[7:-3]
                elif text_content.strip().startswith("```"):
                    text_content = text_content.strip()[3:-3]
                    
                design_data = json.loads(text_content)
                segments = design_data.get("segments", [])
                print("SUCCESS: Used Gemini for Video Design")

            else:
                # Default to OpenAI
                design_completion = self._llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": design_prompt}],
                    response_format={"type": "json_object"}
                )
                content = design_completion.choices[0].message.content
                design_data = json.loads(content)
                segments = design_data.get("segments", [])


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
            
            # --- SWITCH TO SLIDE + AUDIO GENERATION ---
            # Create a directory for this lesson
            lesson_id = uuid.uuid4().hex[:8]
            lesson_dir = output_dir / f"lesson_{lesson_id}"
            
            result = generate_slides_and_audio(
                video_payload=video_payload,
                output_dir=lesson_dir,
                client=self._llm_client,
                voice="alloy", 
                tts_model="gpt-4o-mini-tts"
            )
            
            # result structure: {"slides": [...], "base_dir": "lesson_..."}
            
            return {
                "video_mode": "slideshow",
                "slides_data": result,
                # For compatibility, we can point video_file to index 1 or directory
                "video_file": str(lesson_dir) 
            }
            
        except Exception as e:
            print(f"Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
