
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
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    # 1. Try default cwd resolution
    load_dotenv(override=False)

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)

class SlideGeneratorService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = None

        if not self.api_key:
            # In local/dev environments we still want a working generator, so fall back to
            # deterministic slides when the API key is unavailable instead of crashing.
            print("Warning: GOOGLE_API_KEY not set for SlideGeneratorService. Falling back to offline slides.")
            return

        try:
            # FORCE REST transport to bypass gRPC DNS failures
            genai.configure(api_key=self.api_key, transport="rest")
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.model = genai.GenerativeModel(model_name)
        except Exception as exc:  # pragma: no cover - safety net for runtime env issues
            print(f"Warning: Failed to initialize Gemini model ({exc}). Using offline slides instead.")
            self.model = None


    def generate_slides(self, topic: str, audience: str, slide_count: int, style: str, tone: str = "Professional", detail_level: str = "Standard") -> dict:
        if not self.model:
            return self._generate_offline_slides(topic, audience, slide_count, style, tone, detail_level)

        user_content = get_user_prompt(topic, audience, slide_count, style, tone, detail_level)
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

    def _generate_offline_slides(
        self,
        topic: str,
        audience: str,
        slide_count: int,
        style: str,
        tone: str,
        detail_level: str,
    ) -> dict:
        count = max(3, min(int(slide_count or 6), 12))
        topic = (topic or "Untitled Topic").strip()
        audience = (audience or "general audience").strip()
        style = (style or "modern").strip()
        tone = (tone or "Professional").strip()
        detail_level = (detail_level or "Standard").strip()

        slide_templates = [
            (
                "Overview",
                "full_content",
                f"## What this deck covers\n\n- Core idea behind **{topic}**\n- Why it matters for {audience}\n- Key concepts, examples, and practical takeaways\n\n> Use this deck as a starting point and refine the details in Studio.",
            ),
            (
                "Learning Goals",
                "content_sidebar",
                f"## By the end, learners should be able to\n\n1. Explain the purpose of **{topic}**\n2. Identify the main components and tradeoffs\n3. Apply the concept in a realistic scenario\n\n**Audience:** {audience}",
            ),
            (
                "Core Concepts",
                "two_column",
                f"## Essential building blocks\n\n| Concept | Why it matters |\n|---|---|\n| Context | Shows when {topic} is useful |\n| Process | Turns theory into repeatable steps |\n| Evaluation | Helps learners judge quality and results |",
            ),
            (
                "Practical Workflow",
                "full_content",
                f"## Suggested workflow\n\n1. Define the problem or learning need\n2. Break **{topic}** into smaller skills\n3. Demonstrate one worked example\n4. Let learners practice with feedback\n5. Review common mistakes",
            ),
            (
                "Example Application",
                "code_focus",
                f"## Example structure\n\n```text\nTopic: {topic}\nAudience: {audience}\nStyle: {style}\nTone: {tone}\nDetail: {detail_level}\n```\n\nUse this as a planning scaffold before adding domain-specific examples.",
            ),
            (
                "Summary",
                "full_content",
                f"## Key takeaways\n\n- **{topic}** becomes easier to teach when it is broken into concepts, examples, and practice.\n- Match depth and pacing to {audience}.\n- Revise generated slides with concrete examples from your course material.",
            ),
        ]

        slides = []
        for index in range(count):
            title, layout, content = slide_templates[index % len(slide_templates)]
            slides.append(
                {
                    "title": title if index < len(slide_templates) else f"{title} {index + 1}",
                    "layout": layout,
                    "content": content,
                    "notes": f"Speaker note: explain this slide in a {tone.lower()} tone and connect it to {topic}.",
                }
            )

        return {
            "title": f"{topic} Slide Deck",
            "description": (
                "Offline generated slide deck. Add GEMINI_API_KEY or GOOGLE_API_KEY "
                "for AI-generated slide content."
            ),
            "slides": slides,
        }

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

