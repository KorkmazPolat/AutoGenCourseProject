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
    visual_hints: Optional[List[str]] = None
    content_blocks: Optional[List[Dict[str, Any]]] = None


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
    background_color: Optional[Tuple[int, int, int]] = None,
    heading_color: Optional[Tuple[int, int, int]] = None,
    content_color: Optional[Tuple[int, int, int]] = None,
    *,
    page_total: Optional[int] = None,
    course_title: Optional[str] = None,
    theme: str = "dark",
    logo_path: Optional[Path] = None,
) -> Path:
    """Render a single slide image with improved visual design and bullets."""
    # Theme palette
    if theme == "light":
        background_color = background_color or (245, 248, 252)
        heading_color = heading_color or (24, 32, 48)
        content_color = content_color or (40, 50, 70)
        panel_fill = (255, 255, 255)
        panel_outline = (218, 225, 235)
        footer_fill = (235, 240, 246)
        footer_text = (80, 92, 110)
    else:
        background_color = background_color or (16, 22, 35)
        heading_color = heading_color or (255, 255, 255)
        content_color = content_color or (220, 230, 245)
        panel_fill = (26, 33, 48)
        panel_outline = (38, 46, 64)
        footer_fill = (20, 26, 40)
        footer_text = (200, 210, 225)
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
    footer_font = _resolve_font(24)

    W, H = size
    margin_x, margin_y = 72, 68
    content_width = W - (2 * margin_x)

    # Top accent bar
    draw.rectangle([(0, 0), (W, 8)], fill=accent)

    # Diagonal corner accent for subtle branding flair
    try:
        draw.polygon([(W, 0), (W, 80), (W - 140, 0)], fill=(accent[0], max(0, accent[1]-25), max(0, accent[2]-25)))
    except Exception:
        pass

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

    # Body panel with soft shadow
    panel_top = current_y - 8
    panel_left = margin_x - 12
    panel_right = W - margin_x + 12
    panel_bottom = H - margin_y + 8
    # shadow
    shadow_offset = 6
    draw.rectangle([(panel_left+shadow_offset, panel_top+shadow_offset), (panel_right+shadow_offset, panel_bottom+shadow_offset)], fill=(10, 12, 18))
    try:
        draw.rounded_rectangle(
            [(panel_left, panel_top), (panel_right, panel_bottom)],
            radius=16,
            fill=panel_fill,
            outline=panel_outline,
            width=2,
        )
    except Exception:
        draw.rectangle([(panel_left, panel_top), (panel_right, panel_bottom)], fill=panel_fill, outline=panel_outline, width=2)

    def draw_bullets(items: List[str], indent: int = 28) -> None:
        nonlocal current_y
        bullet_gap = 12
        for ln in items[:8]:
            words = ln.split()
            if len(words) > 14:
                ln = " ".join(words[:14]) + "…"
            dot_x = margin_x
            dot_y = current_y + 12
            draw.ellipse([(dot_x, dot_y), (dot_x + 10, dot_y + 10)], fill=accent)
            text_x = margin_x + indent
            wrapped = _wrap_text(draw, ln, content_font, content_width - indent)
            for wline in wrapped:
                draw.text((text_x, current_y), wline, font=content_font, fill=content_color)
                _, _, _, line_height = draw.textbbox((text_x, current_y), wline or " ", font=content_font)
                current_y += line_height + 4
            current_y += bullet_gap

    def draw_checklist(items: List[str]) -> None:
        nonlocal current_y
        box_size = 18
        gap = 12
        for ln in items[:8]:
            text_x = margin_x + 28 + box_size + 8
            # box
            bx0 = margin_x + 28
            by0 = current_y + 6
            bx1 = bx0 + box_size
            by1 = by0 + box_size
            draw.rectangle([(bx0, by0), (bx1, by1)], outline=accent, width=2, fill=None)
            wrapped = _wrap_text(draw, ln, content_font, content_width - (text_x - margin_x))
            for wline in wrapped:
                draw.text((text_x, current_y), wline, font=content_font, fill=content_color)
                _, _, _, line_height = draw.textbbox((text_x, current_y), wline or " ", font=content_font)
                current_y += line_height + 4
            current_y += gap

    def draw_callout(text: str) -> None:
        nonlocal current_y
        pad_x, pad_y = 18, 14
        text_lines = _wrap_text(draw, text, content_font, content_width - 2 * pad_x)
        # Measure height
        _, _, _, line_h = draw.textbbox((0, 0), "A", font=content_font)
        total_h = pad_y * 2 + len(text_lines) * (line_h + 4)
        y0 = current_y
        y1 = y0 + total_h
        # background slightly tinted by accent
        fill = (max(0, panel_fill[0] + (accent[0]-panel_fill[0])//6),
                max(0, panel_fill[1] + (accent[1]-panel_fill[1])//6),
                max(0, panel_fill[2] + (accent[2]-panel_fill[2])//6))
        try:
            draw.rounded_rectangle([(margin_x, y0), (W - margin_x, y1)], radius=12, fill=fill, outline=panel_outline)
        except Exception:
            draw.rectangle([(margin_x, y0), (W - margin_x, y1)], fill=fill, outline=panel_outline)
        # left accent bar
        draw.rectangle([(margin_x, y0), (margin_x + 6, y1)], fill=accent)
        tx = margin_x + pad_x
        ty = y0 + pad_y
        for line in text_lines:
            draw.text((tx, ty), line, font=content_font, fill=content_color)
            _, _, _, lh = draw.textbbox((tx, ty), line or " ", font=content_font)
            ty += lh + 4
        current_y = y1 + 12

    def draw_quote(text: str) -> None:
        nonlocal current_y
        pad_x, pad_y = 18, 10
        y0 = current_y
        # left bar
        draw.rectangle([(margin_x, y0 + 6), (margin_x + 6, y0 + 140)], fill=panel_outline)
        tx = margin_x + pad_x + 6
        text_lines = _wrap_text(draw, text, content_font, content_width - (tx - margin_x) - pad_x)
        for line in text_lines:
            draw.text((tx, current_y), line, font=content_font, fill=(190, 200, 215))
            _, _, _, lh = draw.textbbox((tx, current_y), line or " ", font=content_font)
            current_y += lh + 4
        current_y += 10

    def draw_example(title: Optional[str], text: str) -> None:
        nonlocal current_y
        pad_x, pad_y = 16, 12
        tx = margin_x + pad_x
        # header
        if title:
            draw.text((tx, current_y), title, font=_resolve_font(28, bold=True), fill=heading_color)
            _, _, _, lh = draw.textbbox((tx, current_y), title or " ", font=_resolve_font(28, bold=True))
            current_y += lh + 6
        # body box
        lines = _wrap_text(draw, text, content_font, content_width - 2 * pad_x)
        _, _, _, line_h = draw.textbbox((0, 0), "A", font=content_font)
        total_h = pad_y * 2 + len(lines) * (line_h + 4)
        y0 = current_y
        y1 = y0 + total_h
        try:
            draw.rounded_rectangle([(margin_x, y0), (W - margin_x, y1)], radius=12, fill=panel_fill, outline=panel_outline)
        except Exception:
            draw.rectangle([(margin_x, y0), (W - margin_x, y1)], fill=panel_fill, outline=panel_outline)
        ty = y0 + pad_y
        for line in lines:
            draw.text((tx, ty), line, font=content_font, fill=content_color)
            _, _, _, lh = draw.textbbox((tx, ty), line or " ", font=content_font)
            ty += lh + 4
        current_y = y1 + 12

    # Render content blocks if present; else fall back to bullets from content
    blocks = slide.content_blocks or []
    if blocks:
        for block in blocks:
            btype = (block.get("type") or "").lower()
            if btype == "bullets" and isinstance(block.get("items"), list):
                draw_bullets([str(x) for x in block.get("items")])
            elif btype == "checklist" and isinstance(block.get("items"), list):
                draw_checklist([str(x) for x in block.get("items")])
            elif btype == "callout" and isinstance(block.get("text"), str):
                draw_callout(block.get("text", ""))
            elif btype == "quote" and isinstance(block.get("text"), str):
                draw_quote(block.get("text", ""))
            elif btype == "example":
                txt = block.get("text") or ""
                title = block.get("title")
                if isinstance(txt, str):
                    draw_example(title if isinstance(title, str) else None, txt)
    else:
        # Bulleted body content from legacy "content" string, with a small heuristic
        # to synthesize richer blocks (checklist/callout) when possible.
        raw = (slide.content or "").strip()
        raw_lines = [ln.strip("-• \t").strip() for ln in raw.splitlines() if ln.strip()]
        if not raw_lines and raw:
            parts = [p.strip() for p in raw.replace("\u2022", " ").split(";") if p.strip()]
            if len(parts) > 1:
                raw_lines = parts
            else:
                sentences = [s.strip() for s in raw.replace("\n", " ").split(".") if s.strip()]
                raw_lines = sentences
        if not raw_lines:
            raw_lines = _wrap_text(draw, slide.content, content_font, content_width)

        # Heuristics
        tips = [ln for ln in raw_lines if ln.lower().startswith(("tip:", "note:", "caution:"))]
        remaining = [ln for ln in raw_lines if ln not in tips]
        is_checklisty = any(keyword in (slide.heading or "").lower() for keyword in ("step", "how to", "checklist")) or any(
            ln[:2].isdigit() for ln in raw_lines if ln
        )

        blocks_to_draw: List[Dict[str, Any]] = []
        if remaining:
            blocks_to_draw.append({"type": "bullets", "items": remaining[:5]})
        if is_checklisty and remaining:
            blocks_to_draw.append({"type": "checklist", "items": remaining[:4]})
        if tips:
            blocks_to_draw.append({"type": "callout", "text": tips[0]})

        if not blocks_to_draw:
            draw_bullets(raw_lines)
        else:
            for blk in blocks_to_draw:
                if blk["type"] == "bullets":
                    draw_bullets([str(x) for x in blk.get("items", [])])
                elif blk["type"] == "checklist":
                    draw_checklist([str(x) for x in blk.get("items", [])])
                elif blk["type"] == "callout":
                    draw_callout(str(blk.get("text", "")))

    # Visual hint badges (top-right of panel)
    hints = (slide.visual_hints or [])[:3]
    if hints:
        badge_y = panel_top + 16
        x = panel_right
        for hint in reversed(hints):
            text = hint[:24] + ("…" if len(hint) > 24 else "")
            badge_font = _resolve_font(20, bold=True)
            tw_l, _, tw_r, th = draw.textbbox((0, 0), text, font=badge_font)
            tw = tw_r - tw_l
            pad_x, pad_y = 12, 6
            bw = tw + pad_x * 2
            x -= (bw + 8)
            by0, by1 = badge_y, badge_y + th + pad_y * 2
            try:
                draw.rounded_rectangle([(x, by0), (x + bw, by1)], radius=12, fill=(panel_outline[0], panel_outline[1], panel_outline[2]))
            except Exception:
                draw.rectangle([(x, by0), (x + bw, by1)], fill=panel_outline)
            draw.text((x + pad_x, badge_y + pad_y), text, font=badge_font, fill=(30, 36, 50) if theme == "light" else (235, 242, 255))

    # Footer bar with course title and page number
    footer_h = 40
    footer_y0 = H - footer_h
    draw.rectangle([(0, footer_y0), (W, H)], fill=footer_fill)
    if course_title:
        label = course_title.strip()
        # Truncate visually if too long
        max_chars = 60
        if len(label) > max_chars:
            label = label[:max_chars-1] + "…"
        draw.text((margin_x, footer_y0 + 10), label, font=footer_font, fill=footer_text)
    if slide.page and page_total:
        page_text = f"{slide.page}/{page_total}"
        tw_left, _, tw_right, _ = draw.textbbox((0, 0), page_text, font=footer_font)
        tw = tw_right - tw_left
        draw.text((W - margin_x - tw, footer_y0 + 10), page_text, font=footer_font, fill=(160, 172, 192))

    # Optional logo watermark (bottom-right above footer)
    if logo_path and Path(logo_path).exists():
        try:
            logo = Image.open(logo_path).convert("RGBA")
            # scale by width
            target_w = 120
            w, h = logo.size
            scale = target_w / float(w)
            logo = logo.resize((int(w * scale), int(h * scale)))
            lw, lh = logo.size
            pos = (W - lw - margin_x, footer_y0 - lh - 12)
            image.paste(logo, pos, logo)
        except Exception:
            pass

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
        (
            SlideSpec(
                page=item["page"],
                heading=item.get("heading", ""),
                content=item.get("content", ""),
                visual_hints=item.get("visual_hints"),
                content_blocks=item.get("content_blocks"),
            )
            for item in presentation
        ),
        key=lambda slide: slide.page,
    )

    # If lengths differ, align to the shorter list to avoid hard failure
    if len(ordered_presentation) != len(narration):
        min_len = min(len(ordered_presentation), len(narration))
        if min_len == 0:
            raise VideoGenerationError(
                f"Presentation pages ({len(ordered_presentation)}) and narration sections ({len(narration)}) are not usable."
            )
        ordered_presentation = ordered_presentation[:min_len]
        narration = narration[:min_len]

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
    course_title: Optional[str] = None,
    theme: str = "dark",
    logo_path: Optional[str] = None,
    add_intro_outro: bool = True,
    pause_between_slides: float = 0.4,
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

        sequence: List[Tuple[SlideSpec, str]] = []

        # Intro card
        if add_intro_outro:
            intro_heading = (course_title or "Course Intro").strip()
            intro_content = (video_payload.get("hook") or "").strip()
            if intro_content:
                sequence.append((SlideSpec(page=0, heading=intro_heading, content=intro_content), intro_content))

        # Main slides
        sequence.extend(pairs)

        # Outro card
        if add_intro_outro:
            cta = (video_payload.get("call_to_action") or "").strip()
            if cta:
                sequence.append((SlideSpec(page=0, heading="Next Steps", content=cta), cta))

        total_pages = len(sequence)
        for index, (slide, script) in enumerate(sequence, start=1):
            slide_path = Path(tmp_dir) / f"slide_{index:02d}.png"
            audio_path = Path(tmp_dir) / f"audio_{index:02d}.mp3"

            _create_slide_image(
                slide,
                slide_path,
                page_total=total_pages,
                course_title=course_title,
                theme=theme,
                logo_path=Path(logo_path) if logo_path else None,
            )
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

            # Insert a silent pause (hold on last frame) between slides, if requested
            if pause_between_slides and index < total_pages:
                pause_dur = max(0.0, float(pause_between_slides))
                if pause_dur > 0:
                    pause_clip = ImageClip(str(slide_path)).set_duration(pause_dur)
                    clips.append(pause_clip)

        if not clips:
            raise VideoGenerationError("No clips were generated for the video.")

        # Apply crossfades only if no explicit pauses are requested
        crossfade = 0.0 if (pause_between_slides and pause_between_slides > 0) else (0.5 if len(clips) > 1 else 0)
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
                # Account for pause after each slide except the last
                if pause_between_slides and idx < len(slide_specs):
                    start = end + max(0.0, float(pause_between_slides))
                else:
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
                # Include pause in cumulative timeline between slides
                if pause_between_slides and idx < len(slide_specs):
                    cumulative += dur + max(0.0, float(pause_between_slides))
                else:
                    cumulative += dur
            chapters_path.write_text("\n".join(ch_lines), encoding="utf-8")
        except Exception:
            # Generating VTTs is best-effort; don't fail video export on this.
            pass

        for clip in clips:
            clip.close()
        final_clip.close()

    return output_path
