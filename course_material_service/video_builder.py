from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

#from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips  # type: ignore[import]
from PIL import Image, ImageDraw, ImageFont  # type: ignore[import]


class VideoGenerationError(RuntimeError):
    """Raised when the automated video generation pipeline fails."""


@dataclass(frozen=True)
class SlideSpec:
    page: int
    heading: str
    content: str


def _resolve_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Attempt to load a truetype font, falling back to the default bitmap font."""
    if bold:
        font_candidates: Sequence[str] = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        )
    else:
        font_candidates = (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        )

    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue

    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wrap text so it fits within the desired width."""
    lines: List[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            left, _, right, _ = draw.textbbox((0, 0), test_line, font=font)
            if (right - left) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _create_slide_image(
    slide: SlideSpec,
    output_path: Path,
    size: Tuple[int, int] = (1280, 720),
    background_color: Tuple[int, int, int] = (18, 24, 38),
    heading_color: Tuple[int, int, int] = (255, 255, 255),
    content_color: Tuple[int, int, int] = (210, 220, 235),
) -> Path:
    """Render a single slide image containing the heading and bullet content."""
    image = Image.new("RGB", size, color=background_color)
    draw = ImageDraw.Draw(image)

    heading_font = _resolve_font(60, bold=True)
    content_font = _resolve_font(36)

    margin_x, margin_y = 80, 80
    content_width = size[0] - (2 * margin_x)

    heading_lines = _wrap_text(draw, slide.heading, heading_font, content_width)
    current_y = margin_y
    for line in heading_lines:
        draw.text((margin_x, current_y), line, font=heading_font, fill=heading_color)
        _, _, _, line_height = draw.textbbox((margin_x, current_y), line or " ", font=heading_font)
        current_y += line_height + 10

    current_y += 20

    body_lines = _wrap_text(draw, slide.content, content_font, content_width)
    for line in body_lines:
        draw.text((margin_x, current_y), line, font=content_font, fill=content_color)
        _, _, _, line_height = draw.textbbox((margin_x, current_y), line or " ", font=content_font)
        current_y += line_height + 6

    image.save(output_path, format="PNG")
    return output_path


def _synthesize_audio_clip(
    client: Any,
    script: str,
    output_path: Path,
    *,
    voice: str = "alloy",
    tts_model: str = "gpt-4o-mini-tts",
) -> Path:
    """Create an audio narration file for the provided script using OpenAI TTS."""
    cleaned_script = script.strip()
    if not cleaned_script:
        raise VideoGenerationError("Narration script is empty; cannot synthesize audio.")

    try:
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=voice,
            input=cleaned_script,
        ) as response:
            response.stream_to_file(output_path)
    except Exception as exc:  # pragma: no cover
        raise VideoGenerationError(f"Text-to-speech synthesis failed: {exc}") from exc

    return output_path


def _pair_presentation_and_narration(
    presentation: Sequence[Dict[str, Any]],
    narration: Sequence[Dict[str, Any]],
) -> List[Tuple[SlideSpec, str]]:
    """Align presentation pages with narration scripts by index order."""
    if not presentation or not narration:
        raise VideoGenerationError("Presentation and narration data must both be provided.")

    ordered_presentation = sorted(
        (SlideSpec(page=item["page"], heading=item["heading"], content=item["content"]) for item in presentation),
        key=lambda slide: slide.page,
    )

    if len(ordered_presentation) != len(narration):
        raise VideoGenerationError(
            f"Presentation pages ({len(ordered_presentation)}) and narration sections ({len(narration)}) differ."
        )

    pairs: List[Tuple[SlideSpec, str]] = []
    for idx, slide in enumerate(ordered_presentation):
        script = narration[idx].get("script", "")
        if not isinstance(script, str) or not script.strip():
            raise VideoGenerationError(f"Narration script for slide {slide.page} is missing or empty.")
        pairs.append((slide, script))
    return pairs


def generate_video_from_script(
    *,
    video_payload: Dict[str, Any],
    output_path: Path,
    client: Any,
    voice: str = "alloy",
    tts_model: str = "gpt-4o-mini-tts",
    fps: int = 30,
) -> Path:
    """Generate a narrated slide video from the structured video script payload."""
    presentation = video_payload.get("presentation", [])
    narration = video_payload.get("narration", [])
    pairs = _pair_presentation_and_narration(presentation, narration)

    output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp_dir:
        clips: List[ImageClip] = []
        for index, (slide, script) in enumerate(pairs, start=1):
            slide_path = Path(tmp_dir) / f"slide_{index:02d}.png"
            audio_path = Path(tmp_dir) / f"audio_{index:02d}.mp3"

            _create_slide_image(slide, slide_path)
            _synthesize_audio_clip(client, script, audio_path, voice=voice, tts_model=tts_model)

            audio_clip = AudioFileClip(str(audio_path))
            image_clip = ImageClip(str(slide_path)).set_duration(audio_clip.duration).set_audio(audio_clip)
            clips.append(image_clip)

        if not clips:
            raise VideoGenerationError("No clips were generated for the video.")

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

        for clip in clips:
            clip.close()
        final_clip.close()

    return output_path

