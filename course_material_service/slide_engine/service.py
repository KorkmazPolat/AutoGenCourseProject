
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
            return self._generate_offline_slides(topic, audience, slide_count, style)

        user_content = get_user_prompt(topic, audience, slide_count, style)
        full_prompt = f"{SLIDE_SYSTEM_PROMPT}\n\n{user_content}"

        try:
            # Request JSON output
            response = self.model.generate_content(
                full_prompt
            )
            
            content_str = response.text
            
            # Basic cleanup if markdown backticks are present
            cleaned_str = content_str.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:]
            elif cleaned_str.startswith("```"):
                cleaned_str = cleaned_str[3:]
                
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3]
            
            data = json.loads(cleaned_str)
            
            # Validation / Sanity Check
            self._validate_response(data)
            
            return data

        except Exception as e:
            print(f"Slide Generation Failed: {e}")
            # Instead of bubbling the error to the UI, provide a deterministic offline deck so
            # the slide experience keeps working even without external API access.
            return self._generate_offline_slides(topic, audience, slide_count, style)

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

    def _generate_offline_slides(self, topic: str, audience: str, slide_count: int, style: str) -> dict:
        """Fallback slide generator that keeps the UI functional when Gemini is unavailable."""
        # Basic guard rails so we never create zero slides
        total_slides = max(1, min(slide_count or 10, 25))

        # Reusable talking points so offline decks still feel structured
        section_templates = [
            ("Kickoff", "Introduce why the topic matters right now."),
            ("Core Concepts", "Highlight the three main ideas learners must remember."),
            ("Deep Dive", "Connect the concept to a realistic scenario with concrete actions."),
            ("Toolbox", "List frameworks, formulas, or tactics that make the idea actionable."),
            ("Case Study", "Tell a short story that illustrates success or failure."),
            ("Checklist", "Provide a repeatable process people can run on their own."),
            ("Wrap-Up", "Summarize the momentum and give a clear next step.")
        ]

        slides = []
        for index in range(total_slides):
            section_title, guidance = section_templates[index % len(section_templates)]
            readable_index = index + 1

            content = dedent(f"""
            ## {section_title}
            - **Context**: {topic} for {audience or 'your audience'}.
            - **Style Inspiration**: {style.title()} aesthetics with bold callouts.
            - **What to Cover**: {guidance}

            ### Mini Framework
            1. Observation
            2. Insight
            3. Action

            > "Great presentations mix clarity with pace." – Studio Assistant
            """).strip()

            if readable_index % 3 == 0:
                table = dedent("""
                | Signal | Description |
                | --- | --- |
                | Challenge | Where learners stumble today |
                | Opportunity | How {topic} helps |
                | Metric | How to measure improvement |
                """).strip().format(topic=topic)
                content = f"{content}\n\n{table}"

            notes = (
                f"Slide {readable_index}: walk the audience through the talking points, "
                f"then ask a question to keep the session interactive."
            )

            slides.append({
                "title": f"{topic} – {section_title}",
                "layout": "full_content",
                "content": content,
                "notes": notes
            })

        description = (
            f"A practical walkthrough of {topic} tailored for {audience or 'a general audience'}. "
            "Generated offline to ensure the slide experience keeps working in restricted environments."
        )

        return {
            "title": f"{topic} Slide Deck",
            "description": description,
            "slides": slides
        }
