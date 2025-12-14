import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from course_material_service.slide_engine.service import SlideGeneratorService
from routes.generate_course import _prepare_slides_for_view


@pytest.fixture
def service():
    """Return a service instance without requiring a real Gemini key."""
    return SlideGeneratorService()


def test_validate_response_normalizes_non_string_fields(service):
    data = {
        "slides": [
            {"title": None, "content": 123, "notes": 456},
            {"title": "", "content": "", "notes": None},
        ]
    }

    # Should not raise and should coerce all content/notes/title fields to strings
    service._validate_response(data)

    first = data["slides"][0]
    assert first["title"] == "Slide 1"
    assert first["content"] == "123"
    assert first["notes"] == "456"

    second = data["slides"][1]
    assert second["title"] == "Slide 2"
    # Empty content is replaced with placeholder text
    assert "Content Visualization" in second["content"]
    assert second["notes"] == "Please review this slide."


def test_offline_generator_creates_requested_slide_count(service):
    fallback = service._generate_offline_slides("Test Topic", "Leaders", 3, "modern")

    assert fallback["title"] == "Test Topic Slide Deck"
    assert len(fallback["slides"]) == 3
    assert all(slide["content"] for slide in fallback["slides"])


def test_prepare_slides_for_view_renders_markdown():
    slides = [
        {"title": "Intro", "content": "## Heading\n\n**Bold** text.", "notes": "note"},
        {"title": "", "content": None, "notes": None}
    ]

    prepared = _prepare_slides_for_view(slides)

    assert prepared[0]["title"] == "Intro"
    assert "<strong>Bold</strong>" in prepared[0]["content_html"]
    assert prepared[1]["title"] == "Slide 2"
    assert "No content available" in prepared[1]["content"]
