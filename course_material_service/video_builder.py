from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)  # type: ignore[import]
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
    background_color: Tuple[int, int, int] = (16, 22, 35),
    heading_color: Tuple[int, int, int] = (255, 255, 255),
    content_color: Tuple[int, int, int] = (220, 230, 245),
) -> Path:
    """Render a single slide image with improved visual design and bullets."""
    image = Image.new("RGB", size, color=background_color)
    draw = ImageDraw.Draw(image)

    # Accent color palette rotated by page index for visual variety
    palette: List[Tuple[int, int, int]] = [
        (99, 102, 241),   # indigo-500
        (16, 185, 129),   # emerald-500
        (59, 130, 246),   # blue-500
        (234, 179, 8),    # amber-500
        (236, 72, 153),   # pink-500
    ]
    accent = palette[(slide.page - 1) % len(palette)] if slide.page else palette[0]

    heading_font = _resolve_font(56, bold=True)
    content_font = _resolve_font(34)

    W, H = size
    margin_x, margin_y = 72, 68
    content_width = W - (2 * margin_x)

    # Top accent bar
    draw.rectangle([(0, 0), (W, 6)], fill=accent)

    # Heading with subtle shadow
    heading_lines = _wrap_text(draw, slide.heading, heading_font, content_width)
    current_y = margin_y
    for line in heading_lines:
        # shadow
        draw.text((margin_x + 2, current_y + 2), line, font=heading_font, fill=(0, 0, 0))
        draw.text((margin_x, current_y), line, font=heading_font, fill=heading_color)
        _, _, _, line_height = draw.textbbox((margin_x, current_y), line or " ", font=heading_font)
        current_y += line_height + 8

    current_y += 16

    # Body panel (subtle rounded rectangle)
    panel_top = current_y - 8
    panel_left = margin_x - 12
    panel_right = W - margin_x + 12
    panel_bottom = H - margin_y + 8
    try:
        draw.rounded_rectangle(
            [(panel_left, panel_top), (panel_right, panel_bottom)],
            radius=16,
            fill=(26, 33, 48),
            outline=(38, 46, 64),
            width=2,
        )
    except Exception:
        # Pillow without rounded corners fallback
        draw.rectangle([(panel_left, panel_top), (panel_right, panel_bottom)], fill=(26, 33, 48), outline=(38, 46, 64), width=2)

    # Bulleted body content: split into lines, constrain to concise bullets
    raw_lines = [ln.strip() for ln in (slide.content or "").splitlines() if ln.strip()]
    if not raw_lines:
        raw_lines = _wrap_text(draw, slide.content, content_font, content_width)

    # Enforce 3-8 bullets, each <= ~12 words when possible
    bullets: List[str] = []
    for ln in raw_lines:
        words = ln.split()
        if len(words) > 14:
            ln = " ".join(words[:14]) + "…"
        bullets.append(ln)
        if len(bullets) >= 8:
            break

    bullet_indent = 24
    bullet_gap = 10
    for ln in bullets:
        # bullet dot
        dot_x = margin_x
        dot_y = current_y + 12
        draw.ellipse([(dot_x, dot_y), (dot_x + 8, dot_y + 8)], fill=accent)
        # text
        text_x = margin_x + bullet_indent
        wrapped = _wrap_text(draw, ln, content_font, content_width - bullet_indent)
        for wline in wrapped:
            draw.text((text_x, current_y), wline, font=content_font, fill=content_color)
            _, _, _, line_height = draw.textbbox((text_x, current_y), wline or " ", font=content_font)
            current_y += line_height + 4
        current_y += bullet_gap

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
    """Generate a narrated slide video from the structured video script payload.

    Besides the .mp4, this function also emits WebVTT caption and chapter files
    next to the output video, using one caption cue per slide (aligned to the
    synthesized narration duration) and one chapter entry per slide heading.
    """
    presentation = video_payload.get("presentation", [])
    narration = video_payload.get("narration", [])
    pairs = _pair_presentation_and_narration(presentation, narration)

    output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Helper for time formatting in WebVTT
    def _fmt_ts(total_seconds: float) -> str:
        if total_seconds < 0:
            total_seconds = 0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        millis = int(round((total_seconds - int(total_seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

    with TemporaryDirectory() as tmp_dir:
        clips: List[ImageClip] = []
        slide_durations: List[float] = []
        slide_specs: List[SlideSpec] = []
        slide_scripts: List[str] = []

        for index, (slide, script) in enumerate(pairs, start=1):
            slide_path = Path(tmp_dir) / f"slide_{index:02d}.png"
            audio_path = Path(tmp_dir) / f"audio_{index:02d}.mp3"

            _create_slide_image(slide, slide_path)
            _synthesize_audio_clip(client, script, audio_path, voice=voice, tts_model=tts_model)

            audio_clip = AudioFileClip(str(audio_path))
            duration = float(audio_clip.duration)
            image_clip = ImageClip(str(slide_path)).set_duration(duration).set_audio(audio_clip)
            # Subtle audio fades for cleaner joins
            image_clip = image_clip.audio_fadein(0.15).audio_fadeout(0.2)

            clips.append(image_clip)
            slide_durations.append(duration)
            slide_specs.append(slide)
            slide_scripts.append(script)

        if not clips:
            raise VideoGenerationError("No clips were generated for the video.")

        # Apply crossfades between slides for smoother visual flow
        crossfade = 0.5 if len(clips) > 1 else 0
        if crossfade > 0:
            xfaded: List[ImageClip] = []
            for i, c in enumerate(clips):
                if i == 0:
                    xfaded.append(c)
                else:
                    xfaded.append(c.crossfadein(crossfade))
            final_clip = concatenate_videoclips(xfaded, method="compose", padding=-crossfade)
        else:
            final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

        # Build caption and chapter tracks aligned to narration timing
        try:
            captions_path = output_path.with_suffix(".vtt")
            chapters_path = output_path.with_name(output_path.stem + ".chapters.vtt")

            # WebVTT: one cue per slide with full script
            start = 0.0
            lines: List[str] = ["WEBVTT\n"]
            for idx, (spec, script_text, dur) in enumerate(zip(slide_specs, slide_scripts, slide_durations), start=1):
                end = start + dur
                lines.append(f"{_fmt_ts(start)} --> {_fmt_ts(end)}")
                # Prefix with slide heading for clarity
                safe_heading = (spec.heading or f"Slide {idx}").strip()
                lines.append(f"[{safe_heading}] {script_text.strip()}")
                lines.append("")
                start = end
            captions_path.write_text("\n".join(lines), encoding="utf-8")

            # WebVTT chapters: point to the start time of each slide with its heading
            ch_lines: List[str] = ["WEBVTT\n"]
            cumulative = 0.0
            for idx, (spec, dur) in enumerate(zip(slide_specs, slide_durations), start=1):
                start_ts = _fmt_ts(cumulative)
                end_ts = _fmt_ts(cumulative + max(dur, 0.1))
                title = (spec.heading or f"Slide {idx}").strip()
                ch_lines.append(f"{start_ts} --> {end_ts}")
                ch_lines.append(title)
                ch_lines.append("")
                cumulative += dur
            chapters_path.write_text("\n".join(ch_lines), encoding="utf-8")
        except Exception:
            # Generating VTTs is best-effort; don't fail video export on this.
            pass

        for clip in clips:
            clip.close()
        final_clip.close()

    return output_path
