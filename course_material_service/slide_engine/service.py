
import json
import os
from pathlib import Path
from textwrap import dedent

import google.generativeai as genai
from fastapi import HTTPException
from dotenv import load_dotenv

from .prompts import SLIDE_SYSTEM_PROMPT, get_user_prompt

# Ensure environment variables from a project-level .env are available even when this
# module is imported outside the main FastAPI startup sequence.
if not os.getenv("GOOGLE_API_KEY"):
    # 1. Try default cwd resolution
    load_dotenv(override=False)

if not os.getenv("GOOGLE_API_KEY"):
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)

class SlideGeneratorService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = None

        if not self.api_key:
            # In local/dev environments we still want a working generator, so fall back to
            # deterministic slides when the API key is unavailable instead of crashing.
            print("Warning: GOOGLE_API_KEY not set for SlideGeneratorService. Falling back to offline slides.")
            return

        try:
            # FORCE REST transport to bypass gRPC DNS failures
            genai.configure(api_key=self.api_key, transport="rest")
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        except Exception as exc:  # pragma: no cover - safety net for runtime env issues
            print(f"Warning: Failed to initialize Gemini model ({exc}). Using offline slides instead.")
            self.model = None


    def generate_slides(self, topic: str, audience: str, slide_count: int, style: str) -> dict:
        if not self.model:
            raise ValueError("Gemini model not initialized. Check your GOOGLE_API_KEY.")

        user_content = get_user_prompt(topic, audience, slide_count, style)
        full_prompt = f"{SLIDE_SYSTEM_PROMPT}\n\n{user_content}"

        # Request JSON output
        response = self.model.generate_content(
            full_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        content_str = response.text
        
        # Basic cleanup if markdown backticks are present (even with mime type, sometimes it happens)
        cleaned_str = content_str.strip()
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:]
        elif cleaned_str.startswith("```"):
            cleaned_str = cleaned_str[3:]
            
        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-3]
        
        try:
            data = json.loads(cleaned_str)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Response: {content_str}")
            raise e
        
        # Validation / Sanity Check
        self._validate_response(data)
        
        return data

    def _validate_response(self, data: dict):
        if "slides" not in data or not isinstance(data["slides"], list):
             # Try to fix structure if possible or raise
             if "presentation" in data and "slides" in data["presentation"]:
                 data["slides"] = data["presentation"]["slides"]
             else:
                 raise ValueError("Invalid JSON structure: missing 'slides' array")

        # Sanitize empty content immediate fix
        for i, slide in enumerate(data.get("slides", [])):
            raw_content = slide.get("content")
            slide["content"] = str(raw_content)
            if not slide["content"].strip():
                slide["content"] = "## Content Visualization\n\n*(Content generation was minimal, please edit in Studio)*"
                slide["notes"] = "Please review this slide."

            raw_notes = slide.get("notes")
            if raw_notes is None:
                slide["notes"] = "No notes."
            else:
                slide["notes"] = str(raw_notes)
            
            if not slide.get("title"):
                slide["title"] = f"Slide {i+1}"

