from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

from moviepy import (
    AudioFileClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    TextClip,
) # type: ignore[import]
try:
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
    from moviepy.video.fx.all import crossfadein
except ImportError:
    # Fallback or mock if imports fail (though they should exist in v2)
    # In v2.0.0+, structure might be different.
    # Try importing from moviepy.audio.fx and moviepy.video.fx
    try:
        from moviepy.audio.fx.AudioFadeIn import AudioFadeIn as audio_fadein
        from moviepy.audio.fx.AudioFadeOut import AudioFadeOut as audio_fadeout
        from moviepy.video.fx.CrossFadeIn import CrossFadeIn as crossfadein
    except ImportError:
        pass
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


def _resolve_font(size: int, bold: bool = False, family: str = None) -> ImageFont.FreeTypeFont:
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
    """Render a single slide image with a premium, modern design."""
    
    # --- Design System ---
    W, H = size
    
    # Color Palettes
    if theme == "light":
        bg_start = (255, 255, 255)
        bg_end = (241, 245, 249) # Slate 100
        text_primary = (15, 23, 42) # Slate 900
        text_secondary = (51, 65, 85) # Slate 700
        accent_colors = [
            ((79, 70, 229), (168, 85, 247)), # Indigo to Purple
            ((37, 99, 235), (56, 189, 248)), # Blue to Cyan
            ((219, 39, 119), (244, 114, 182)), # Pink to Rose
            ((5, 150, 105), (52, 211, 153)), # Emerald to Teal
            ((234, 88, 12), (251, 146, 60)), # Orange to Amber
        ]
        card_bg = (255, 255, 255, 240)
        card_border = (226, 232, 240, 255)
    else:
        bg_start = (15, 23, 42) # Slate 900
        bg_end = (2, 6, 23) # Slate 950
        text_primary = (248, 250, 252) # Slate 50
        text_secondary = (203, 213, 225) # Slate 300
        accent_colors = [
            ((99, 102, 241), (168, 85, 247)), # Indigo to Purple
            ((59, 130, 246), (34, 211, 238)), # Blue to Cyan
            ((236, 72, 153), (244, 63, 94)), # Pink to Rose
            ((16, 185, 129), (52, 211, 153)), # Emerald to Teal
            ((245, 158, 11), (251, 191, 36)), # Amber to Yellow
        ]
        card_bg = (30, 41, 59, 200) # Slate 800 transparent
        card_border = (51, 65, 85, 255) # Slate 700

    # Pick accent based on page number
    accent_idx = (slide.page - 1) % len(accent_colors) if slide.page else 0
    accent_start, accent_end = accent_colors[accent_idx]

    # Fonts
    # Try to get a better font if possible
    title_font = _resolve_font(64, bold=True)
    body_font = _resolve_font(38)
    small_font = _resolve_font(24)
    
    # --- Background Generation ---
    base_img = Image.new("RGB", size, color=bg_start)
    draw = ImageDraw.Draw(base_img)
    
    # Linear Gradient Background
    for y in range(H):
        r = int(bg_start[0] + (bg_end[0] - bg_start[0]) * y / H)
        g = int(bg_start[1] + (bg_end[1] - bg_start[1]) * y / H)
        b = int(bg_start[2] + (bg_end[2] - bg_start[2]) * y / H)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Add dynamic background shapes (Orbs)
    # Create a separate layer for shapes to handle transparency
    shape_layer = Image.new("RGBA", size, (0, 0, 0, 0))
    shape_draw = ImageDraw.Draw(shape_layer)
    
    # Top-right orb
    orb_size = 600
    orb_x = W - 200
    orb_y = -200
    shape_draw.ellipse(
        [(orb_x, orb_y), (orb_x + orb_size, orb_y + orb_size)],
        fill=(accent_start[0], accent_start[1], accent_start[2], 30)
    )
    
    # Bottom-left orb
    orb_size_2 = 500
    orb_x_2 = -100
    orb_y_2 = H - 300
    shape_draw.ellipse(
        [(orb_x_2, orb_y_2), (orb_x_2 + orb_size_2, orb_y_2 + orb_size_2)],
        fill=(accent_end[0], accent_end[1], accent_end[2], 20)
    )
    
    # Compose shapes
    base_img.paste(shape_layer, (0, 0), shape_layer)
    
    # --- Layout & Content ---
    image = base_img
    draw = ImageDraw.Draw(image)
    
    margin_x = 80
    margin_y = 60
    
    # Header
    # Draw a small accent line above title
    draw.line([(margin_x, margin_y), (margin_x + 100, margin_y)], fill=accent_start, width=6)
    
    title_y = margin_y + 20
    title_lines = _wrap_text(draw, slide.heading, title_font, W - 2 * margin_x)
    for line in title_lines:
        draw.text((margin_x, title_y), line, font=title_font, fill=text_primary)
        _, _, _, lh = draw.textbbox((0, 0), line, font=title_font)
        title_y += lh + 10
        
    content_start_y = title_y + 40
    
    # Content Area
    # We'll use a "card" look for the content to make it pop
    content_width = W - 2 * margin_x
    content_height = H - content_start_y - 80 # Leave room for footer
    
    # Helper to draw styled text
    current_y = content_start_y
    
    def draw_bullet(text: str):
        nonlocal current_y
        bullet_size = 12
        bullet_y = current_y + 18
        
        # Draw custom bullet (diamond shape)
        draw.polygon([
            (margin_x, bullet_y),
            (margin_x + bullet_size, bullet_y + bullet_size),
            (margin_x + 2 * bullet_size, bullet_y),
            (margin_x + bullet_size, bullet_y - bullet_size)
        ], fill=accent_end)
        
        text_x = margin_x + 40
        wrapped = _wrap_text(draw, text, body_font, content_width - 40)
        for line in wrapped:
            draw.text((text_x, current_y), line, font=body_font, fill=text_secondary)
            _, _, _, lh = draw.textbbox((0, 0), line, font=body_font)
            current_y += lh + 8
        current_y += 20 # Paragraph spacing

    def draw_callout(text: str, type: str = "info"):
        nonlocal current_y
        # Draw a colored pill background
        wrapped = _wrap_text(draw, text, body_font, content_width - 60)
        _, _, _, lh = draw.textbbox((0, 0), "A", font=body_font)
        box_h = len(wrapped) * (lh + 8) + 40
        
        # Background
        draw.rounded_rectangle(
            [(margin_x, current_y), (W - margin_x, current_y + box_h)],
            radius=15,
            fill=(accent_start[0], accent_start[1], accent_start[2], 40),
            outline=accent_start,
            width=2
        )
        
        text_y = current_y + 20
        for line in wrapped:
            draw.text((margin_x + 30, text_y), line, font=body_font, fill=text_primary)
            text_y += lh + 8
        current_y += box_h + 30

    def draw_quote(text: str):
        nonlocal current_y
        # Large quote mark
        quote_font = _resolve_font(120, bold=True)
        draw.text((margin_x - 20, current_y - 20), "“", font=quote_font, fill=(accent_start[0], accent_start[1], accent_start[2], 100))
        
        wrapped = _wrap_text(draw, text, _resolve_font(42, bold=True), content_width - 80)
        text_x = margin_x + 60
        for line in wrapped:
            draw.text((text_x, current_y), line, font=_resolve_font(42, bold=True), fill=text_primary)
            _, _, _, lh = draw.textbbox((0, 0), line, font=_resolve_font(42, bold=True))
            current_y += lh + 10
        current_y += 30

    def draw_subheading(text: str):
        nonlocal current_y
        current_y += 15
        draw.text((margin_x + 10, current_y), text.upper(), font=_resolve_font(32, bold=True), fill=accent_end)
        w = draw.textlength(text.upper(), font=_resolve_font(32, bold=True))
        draw.line([(margin_x + 10, current_y + 40), (margin_x + 10 + w + 20, current_y + 40)], fill=accent_start, width=3)
        current_y += 60

    def draw_key_takeaway(text: str):
        nonlocal current_y
        # Glowing box
        wrapped = _wrap_text(draw, text, body_font, content_width - 60)
        _, _, _, lh = draw.textbbox((0, 0), "A", font=body_font)
        box_h = len(wrapped) * (lh + 8) + 70
        
        # Border
        draw.rounded_rectangle(
            [(margin_x, current_y), (W - margin_x, current_y + box_h)],
            radius=20,
            fill=(20, 25, 40, 200),
            outline=accent_start,
            width=3
        )
        
        # Label
        label_font = _resolve_font(24, bold=True)
        draw.text((margin_x + 30, current_y + 20), "KEY TAKEAWAY", font=label_font, fill=accent_start)
        
        text_y = current_y + 60
        for line in wrapped:
            draw.text((margin_x + 30, text_y), line, font=body_font, fill=text_primary)
            text_y += lh + 8
        current_y += box_h + 30

    def draw_diagram(caption: str):
        nonlocal current_y
        # Placeholder for diagram
        box_h = 300
        draw.rectangle(
            [(margin_x + 40, current_y), (W - margin_x - 40, current_y + box_h)],
            fill=(40, 40, 60, 255),
            outline=text_secondary,
            width=2
        )
        
        # Icon
        cx = W // 2
        cy = current_y + box_h // 2
        r = 60
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=accent_end, width=4)
        draw.line([(cx - 40, cy), (cx + 40, cy)], fill=accent_end, width=4)
        draw.line([(cx, cy - 40), (cx, cy + 40)], fill=accent_end, width=4)
        
        # Caption
        if caption:
            draw.text((margin_x + 60, current_y + box_h - 40), f"Diagram: {caption}", font=_resolve_font(24), fill=text_secondary)
            
        current_y += box_h + 40

    def draw_code(code: str, language: str = "python"):
        nonlocal current_y
        code_font = _resolve_font(28, family="Courier New")
        lines = code.split('\n')
        # Limit lines
        if len(lines) > 12:
            lines = lines[:11] + ["..."]
            
        lh = 34
        box_h = len(lines) * lh + 40
        
        # BG
        draw.rectangle(
            [(margin_x, current_y), (W - margin_x, current_y + box_h)],
            fill=(10, 10, 15, 230),
            outline=(100, 100, 100),
            width=1
        )
        # Header
        draw.rectangle([(margin_x, current_y), (W - margin_x, current_y + 30)], fill=(40, 40, 50))
        draw.text((margin_x + 10, current_y+5), f" {language.upper()}", font=_resolve_font(18, bold=True), fill=text_secondary)
        
        ty = current_y + 40
        for ln in lines:
            draw.text((margin_x + 20, ty), ln, font=code_font, fill=(200, 200, 200))
            ty += lh
            
        current_y += box_h + 30

    def draw_table(headers: List[str], rows: List[List[str]]):
        nonlocal current_y
        # Simple table
        cols = max(len(headers), max(len(r) for r in rows) if rows else 0)
        if cols == 0: return
        
        col_w = (content_width - 20) // cols
        row_h = 50
        
        # Header
        for i, h in enumerate(headers):
            draw.rectangle(
                [(margin_x + i*col_w, current_y), (margin_x + (i+1)*col_w, current_y + row_h)],
                fill=accent_start, outline=bg_start
            )
            draw.text((margin_x + i*col_w + 10, current_y+10), str(h), font=_resolve_font(24, bold=True), fill=text_primary)
            
        current_y += row_h
        
        # Rows
        for r_idx, row in enumerate(rows):
            fill = (255, 255, 255, 20) if r_idx % 2 == 0 else (0, 0, 0, 20)
            for i, val in enumerate(row):
                 if i < cols:
                    draw.rectangle(
                        [(margin_x + i*col_w, current_y), (margin_x + (i+1)*col_w, current_y + row_h)],
                        fill=fill
                    )
                    draw.text((margin_x + i*col_w + 10, current_y+10), str(val), font=_resolve_font(24), fill=text_primary)
            current_y += row_h
            
        current_y += 30

    # Render Blocks
    blocks = slide.content_blocks or []
    if blocks:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = (block.get("type") or "").lower()
            if btype in ["bullets", "checklist"]:
                for item in block.get("items", []):
                    draw_bullet(str(item))
            elif btype == "callout":
                draw_callout(str(block.get("text", "")), block.get("style", "info"))
            elif btype == "quote":
                draw_quote(str(block.get("text", "")))
            elif btype == "example":
                draw_callout(f"Example: {block.get('text', '')}")
            elif btype == "subheading":
                draw_subheading(str(block.get("text", "")))
            elif btype == "key_takeaway":
                draw_key_takeaway(str(block.get("text", "")))
            elif btype == "diagram":
                draw_diagram(str(block.get("caption", "")))
            elif btype == "code":
                draw_code(str(block.get("code", "")), str(block.get("language", "text")))
            elif btype == "table":
                draw_table(block.get("headers", []), block.get("rows", []))
    else:
        # Fallback parsing
        raw = (slide.content or "").strip()
        lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
        if not lines:
            lines = _wrap_text(draw, raw, body_font, content_width)
        
        for line in lines:
            if line.lower().startswith("tip:") or line.lower().startswith("note:"):
                draw_callout(line)
            else:
                draw_bullet(line)

    # --- Footer ---
    footer_y = H - 50
    # Page Number
    if page_total:
        page_str = f"{slide.page} / {page_total}"
        draw.text((W - margin_x, footer_y), page_str, font=small_font, fill=text_secondary, anchor="rs")
    
    # Course Title
    if course_title:
        draw.text((margin_x, footer_y), course_title.upper(), font=small_font, fill=text_secondary, anchor="ls")

    # Save
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
    """Create an audio narration file for the provided script using OpenAI TTS or Fallback."""
    cleaned_script = script.strip()
    if not cleaned_script:
        cleaned_script = " " # silence placeholder

    try:
        if not client:
             raise ValueError("No OpenAI client provided.")
             
        with client.audio.speech.with_streaming_response.create(
            model=tts_model,
            voice=voice,
            input=cleaned_script,
        ) as response:
            response.stream_to_file(output_path)
            
    except Exception as exc:
        print(f"Warning: OpenAI TTS failed ({exc}). Falling back to gTTS.")
        try:
            from gtts import gTTS
            tts = gTTS(cleaned_script, lang='en')
            tts.save(str(output_path))
        except Exception as fallback_exc:
            print(f"Fallback TTS failed ({fallback_exc}). Generating 1s silence.")
            try:
                # Generate 1s silence using moviepy
                from moviepy.audio.AudioClip import AudioClip
                # Minimal silent clip
                silence = AudioClip(lambda t: [0], duration=1.0, fps=44100)
                silence.write_audiofile(str(output_path), logger=None)
            except Exception as final_exc: 
                raise VideoGenerationError(f"All TTS attempts failed: {exc}") from exc

    return output_path


def _pair_presentation_and_narration(
    presentation: Sequence[Dict[str, Any]] | Sequence[Any],
    narration: Sequence[Dict[str, Any]] | Sequence[Any],
) -> List[Tuple[SlideSpec, str]]:
    """Align presentation pages with narration scripts by index order.

    Be tolerant of slightly malformed payloads by accepting:
    - presentation as a list of dicts OR a list of strings (headings)
    - narration as a list of dicts with 'script' OR a list of strings (script text)
    """
    if not presentation or not narration:
        raise VideoGenerationError("Presentation and narration data must both be provided.")

    # Normalize presentation into SlideSpec list
    normalized_slides: List[SlideSpec] = []
    for idx, item in enumerate(presentation, start=1):
        if isinstance(item, dict):
            try:
                page_num = int(item.get("page", idx))
            except Exception:
                page_num = idx
            normalized_slides.append(
                SlideSpec(
                    page=page_num,
                    heading=str(item.get("heading", "")),
                    content=str(item.get("content", "")),
                    visual_hints=item.get("visual_hints"),
                    content_blocks=item.get("content_blocks"),
                )
            )
        elif isinstance(item, str):
            normalized_slides.append(SlideSpec(page=idx, heading=item.strip(), content=""))
        else:
            raise VideoGenerationError(
                f"Invalid presentation item at index {idx-1}: expected object or string, got {type(item).__name__}."
            )

    ordered_presentation = sorted(normalized_slides, key=lambda slide: slide.page)

    # Normalize narration into list of script strings
    normalized_scripts: List[str] = []
    for idx, item in enumerate(narration, start=1):
        if isinstance(item, dict):
            script = item.get("script") or item.get("text") or item.get("content") or ""
            normalized_scripts.append(str(script))
        elif isinstance(item, str):
            normalized_scripts.append(item)
        else:
            raise VideoGenerationError(
                f"Invalid narration item at index {idx-1}: expected object or string, got {type(item).__name__}."
            )

    # If lengths differ, align to the shorter list to avoid hard failure
    if len(ordered_presentation) != len(normalized_scripts):
        min_len = min(len(ordered_presentation), len(normalized_scripts))
        if min_len == 0:
            raise VideoGenerationError(f"Presentation pages ({len(ordered_presentation)}) and narration sections ({len(normalized_scripts)}) are not usable.")
        ordered_presentation = ordered_presentation[:min_len]
        normalized_scripts = normalized_scripts[:min_len]

    pairs: List[Tuple[SlideSpec, str]] = []
    for idx, slide in enumerate(ordered_presentation):
        script = normalized_scripts[idx]
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
    print(f"DEBUG: Starting video generation for {len(pairs)} slides.")

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
            # Create base image clip
            base_clip = ImageClip(str(slide_path)).with_duration(duration)
            
            # Apply "Ken Burns" effect (Slow Zoom)
            # We zoom from 1.0 to 1.05 over the duration of the clip
            try:
                # Define zoom function: t is time, returns scale factor
                def zoom_effect(t):
                    return 1.0 + 0.04 * (t / duration)  # 4% zoom over duration

                # Apply resize transformation
                zoomed_clip = base_clip.resize(zoom_effect)
                
                # Center crop to original size to maintain aspect ratio and frame
                w, h = base_clip.size
                final_slide_clip = zoomed_clip.with_position("center").crop(x_center=w/2, y_center=h/2, width=w, height=h)
            except Exception:
                # Fallback if complex effects fail
                final_slide_clip = base_clip

            image_clip = final_slide_clip.with_audio(audio_clip)

            clips.append(image_clip)
            slide_durations.append(duration)
            slide_specs.append(slide)
            slide_scripts.append(script)

            # Insert a silent pause (hold on last frame) between slides, if requested
            if pause_between_slides and index < total_pages:
                pause_dur = max(0.0, float(pause_between_slides))
                if pause_dur > 0:
                    pause_clip = ImageClip(str(slide_path)).with_duration(pause_dur)
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
                    # Apply crossfade
                    try:
                        c = crossfadein(c, crossfade)
                    except TypeError:
                        pass
                    xfaded.append(c)
            final_clip = concatenate_videoclips(xfaded, method="compose", padding=-crossfade)
        else:
            final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
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
            try:
                clip.close()
            except Exception:
                pass
        try:
            final_clip.close()
        except Exception:
            pass

    return output_path


def generate_slides_and_audio(
    *,
    video_payload: Dict[str, Any],
    output_dir: Path,
    client: Any,
    voice: str = "alloy",
    tts_model: str = "gpt-4o-mini-tts",
    course_title: Optional[str] = None,
    theme: str = "dark",
    logo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates slide images and audio files individually to output_dir.
    Returns a manifest of the generated content.
    """
    presentation = video_payload.get("presentation", [])
    narration = video_payload.get("narration", [])
    pairs = _pair_presentation_and_narration(presentation, narration)
    
    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    slides_manifest = []
    
    # 1. Intro
    intro_content = (video_payload.get("hook") or "").strip()
    if intro_content:
        # Create a virtual intro slide pair
        pairs.insert(0, (
            SlideSpec(page=0, heading=course_title or "Introduction", content=intro_content), 
            intro_content
        ))

    # 2. Outro
    cta = (video_payload.get("call_to_action") or "").strip()
    if cta:
         pairs.append((
             SlideSpec(page=999, heading="Next Steps", content=cta),
             cta
         ))
         
    total_pages = len(pairs)
    
    print(f"DEBUG: Generating {total_pages} slides + audio (NO VIDEO RENDER)...")
    
    for index, (slide, script) in enumerate(pairs, start=1):
        # filenames
        img_name = f"slide_{index:02d}.png"
        audio_name = f"audio_{index:02d}.mp3"
        
        slide_path = output_dir / img_name
        audio_path = output_dir / audio_name
        
        # Generate Image
        _create_slide_image(
            slide,
            slide_path,
            page_total=total_pages,
            course_title=course_title,
            theme=theme,
            logo_path=Path(logo_path) if logo_path else None
        )
        
        # Generate Audio
        _synthesize_audio_clip(client, script, audio_path, voice=voice, tts_model=tts_model)
        
        slides_manifest.append({
            "index": index,
            "image_file": img_name,
            "audio_file": audio_name,
            "script": script,
            "heading": slide.heading
        })

    return {
        "slides": slides_manifest,
        "base_dir": str(output_dir.name) # essentially the lesson_ID folder
    }
