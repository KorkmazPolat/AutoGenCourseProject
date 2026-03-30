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
    size: Tuple[int, int] = (1920, 1080),
    background_color: Optional[Tuple[int, int, int]] = None,
    heading_color: Optional[Tuple[int, int, int]] = None,
    content_color: Optional[Tuple[int, int, int]] = None,
    *,
    page_total: Optional[int] = None,
    course_title: Optional[str] = None,
    theme: str = "light", # Forced to match the HTML template style
    logo_path: Optional[Path] = None,
) -> Path:
    """Render a single slide image matching the 'view_slides.html' premium design."""
    
    W, H = size
    
    # --- Design Tokens (from view_slides.html) ---
    # Colors
    c_white = (255, 255, 255)
    c_gray_50 = (249, 250, 251)  # Sidebar / BGs
    c_gray_100 = (243, 244, 246) # Borders
    c_gray_900 = (17, 24, 39)    # Primary Text
    c_gray_600 = (75, 85, 99)    # Secondary Text
    c_pink_500 = (236, 72, 153)  # Accents
    c_pink_50 = (253, 242, 248)  # Light BG
    c_indigo_500 = (99, 102, 241) # Accents
    
    # Fonts
    # Adjusted sizes for 1080p density without overflow
    title_font = _resolve_font(72, bold=True) # Slightly smaller title
    body_font = _resolve_font(42) # Readable but allows more content
    small_font = _resolve_font(28)
    code_font = _resolve_font(32, family="Courier New")

    # --- Canvas Setup ---
    image = Image.new("RGB", size, color=c_white)
    draw = ImageDraw.Draw(image)
    
    # 1. Top Gradient Bar (h-2 in HTML ~ 8px, but scaled for 1080p -> 16px)
    bar_height = 20
    # Create horizontal gradient from Pink to Indigo
    for x in range(W):
        ratio = x / W
        r = int(c_pink_500[0] * (1 - ratio) + c_indigo_500[0] * ratio)
        g = int(c_pink_500[1] * (1 - ratio) + c_indigo_500[1] * ratio)
        b = int(c_pink_500[2] * (1 - ratio) + c_indigo_500[2] * ratio)
        draw.line([(x, 0), (x, bar_height)], fill=(r, g, b))

    # 2. Right Sidebar (w-16 in HTML ~ 64px, scaled -> 120px)
    sidebar_width = 120
    sidebar_x = W - sidebar_width
    draw.rectangle([(sidebar_x, bar_height), (W, H)], fill=c_gray_50)
    draw.line([(sidebar_x, bar_height), (sidebar_x, H)], fill=c_gray_100, width=2)
    
    # Decorative Dots in Sidebar
    dot_radius = 6
    dot_x = sidebar_x + sidebar_width // 2
    dot_start_y = H // 2 - 40
    
    # Dot 1 (Full)
    draw.ellipse([(dot_x - dot_radius, dot_start_y - dot_radius), 
                  (dot_x + dot_radius, dot_start_y + dot_radius)], fill=c_gray_900)
    # Dot 2 (50%)
    dot_y2 = dot_start_y + 40
    draw.ellipse([(dot_x - dot_radius, dot_y2 - dot_radius), 
                  (dot_x + dot_radius, dot_y2 + dot_radius)], fill=(128, 128, 128))
    # Dot 3 (25%)
    dot_y3 = dot_y2 + 40
    draw.ellipse([(dot_x - dot_radius, dot_y3 - dot_radius), 
                  (dot_x + dot_radius, dot_y3 + dot_radius)], fill=(192, 192, 192))

    # 3. Slide Number (if total)
    if page_total:
         # Large faded number on left (hidden xl:flex in HTML, but we'll put it bottom right or decorative)
         # Actually HTML puts it extreme left. Let's put it in the sidebar bottom.
         page_str = f"{slide.page:02d}"
         draw.text((dot_x, H - 100), page_str, font=title_font, fill=(200, 200, 200), anchor="ms")

    # --- Content Layout ---
    margin_x = 80 # Reduced side margin for more space
    margin_top = 100 # Reduced top margin
    content_width = W - sidebar_width - (margin_x * 2)
    max_y = H - 80 # Stop before footer
    
    current_y = margin_top
    
    # 4. Heading
    # Draw heading
    title_lines = _wrap_text(draw, slide.heading, title_font, content_width)
    for line in title_lines:
        draw.text((margin_x, current_y), line, font=title_font, fill=c_gray_900)
        _, _, _, lh = draw.textbbox((0, 0), line, font=title_font)
        current_y += lh + 10
        
    current_y += 40 # Gap after title
    
    # 5. Body Content
    # Helper to draw styled text

    def draw_table(headers: List[str], rows: List[List[str]]):
        nonlocal current_y
        if current_y > max_y: return
        
        # Simple table
        cols = max(len(headers), max(len(r) for r in rows) if rows else 0)
        if cols == 0: return
        
        col_w = (content_width - 20) // cols
        row_h = 50
        
        # Header
        for i, h in enumerate(headers):
            draw.rectangle(
                [(margin_x + i*col_w, current_y), (margin_x + (i+1)*col_w, current_y + row_h)],
                fill=c_gray_100, outline=c_white
            )
            # Use c_gray_900 for dark text on light header
            draw.text((margin_x + i*col_w + 10, current_y+10), str(h), font=_resolve_font(24, bold=True), fill=c_gray_900)
            
        current_y += row_h
        
        # Rows
        for r_idx, row in enumerate(rows):
            if current_y > max_y: break
            # Alternating white and light gray
            fill = c_white if r_idx % 2 == 0 else c_gray_50
            for i, val in enumerate(row):
                 if i < cols:
                    draw.rectangle(
                        [(margin_x + i*col_w, current_y), (margin_x + (i+1)*col_w, current_y + row_h)],
                        fill=fill
                    )
                    draw.text((margin_x + i*col_w + 10, current_y+10), str(val), font=_resolve_font(24), fill=c_gray_900)
            current_y += row_h
            
        current_y += 20

    def draw_styled_bullet(text: str, indent: int = 0):
        nonlocal current_y
        if current_y > max_y: return

        bullet_start_x = margin_x + indent
        
        # Pink bullet dot
        b_r = 6
        b_y = current_y + 22
        draw.ellipse([(bullet_start_x - b_r, b_y - b_r), (bullet_start_x + b_r, b_y + b_r)], fill=c_pink_500)
        
        text_x = bullet_start_x + 40
        wrapped = _wrap_text(draw, text, body_font, content_width - indent - 40)
        for line in wrapped:
            if current_y > max_y: break
            draw.text((text_x, current_y), line, font=body_font, fill=c_gray_600)
            _, _, _, lh = draw.textbbox((0, 0), line, font=body_font)
            current_y += lh + 10 # Tighter
        current_y += 20 # Tighter paragraph gap

    def draw_code_block(code: str, language: str = "text"):
        nonlocal current_y
        if current_y > max_y: return

        # Mac Window Style
        code_lines = code.split('\n')
        if len(code_lines) > 10: code_lines = code_lines[:10] + ["..."]
        
        lh = 40 # Smaller line height
        box_h = len(code_lines) * lh + 50
        
        # Check if entire box fits, else truncate logic (complex) or just draw
        if current_y + box_h > max_y:
             # simplistic: clip
             pass 

        # Draw Container
        draw.rounded_rectangle(
            [(margin_x, current_y), (margin_x + content_width, current_y + box_h)],
            radius=15, fill=(30, 41, 59) # Slate 800
        )
        
        # Header balls
        header_y = current_y + 15
        # Red
        draw.ellipse([(margin_x + 20, header_y), (margin_x + 35, header_y + 15)], fill=(255, 95, 86)) 
        # Yellow
        draw.ellipse([(margin_x + 50, header_y), (margin_x + 65, header_y + 15)], fill=(255, 189, 46)) 
        # Green
        draw.ellipse([(margin_x + 80, header_y), (margin_x + 95, header_y + 15)], fill=(39, 201, 63)) 
        
        # Text
        text_y = current_y + 50
        for ln in code_lines:
            draw.text((margin_x + 30, text_y), ln, font=code_font, fill=(248, 250, 252))
            text_y += lh
            
        current_y += box_h + 30

    def draw_callout(text: str, type: str = "info"):
        nonlocal current_y
        if current_y > max_y: return

        # Draw a colored pill background
        wrapped = _wrap_text(draw, text, body_font, content_width - 60)
        _, _, _, lh = draw.textbbox((0, 0), "A", font=body_font)
        box_h = len(wrapped) * (lh + 6) + 30
        
        # Background
        draw.rounded_rectangle(
            [(margin_x, current_y), (margin_x + content_width, current_y + box_h)],
            radius=15,
            fill=(c_indigo_500[0], c_indigo_500[1], c_indigo_500[2], 30), # Light indigo tint
            outline=c_indigo_500,
            width=2
        )
        
        text_y = current_y + 15
        for line in wrapped:
            draw.text((margin_x + 30, text_y), line, font=body_font, fill=c_gray_900)
            text_y += lh + 6
        current_y += box_h + 20

    def draw_quote(text: str):
        nonlocal current_y
        if current_y > max_y: return

        # Pink styling from HTML
        quote_font = _resolve_font(100, bold=True)
        draw.text((margin_x - 40, current_y - 20), "“", font=quote_font, fill=(c_pink_500[0], c_pink_500[1], c_pink_500[2], 100))
        
        wrapped = _wrap_text(draw, text, _resolve_font(42, bold=True), content_width - 80)
        
        # Pink left border
        start_y = current_y
        
        text_x = margin_x + 40
        for line in wrapped:
            draw.text((text_x, current_y), line, font=_resolve_font(42, bold=True), fill=c_gray_900)
            _, _, _, lh = draw.textbbox((0, 0), line, font=_resolve_font(42, bold=True))
            current_y += lh + 10
        
        current_y += 10
        # Draw left border line
        draw.rectangle([(margin_x, start_y), (margin_x + 6, current_y)], fill=c_pink_500)
        current_y += 20

    def draw_subheading(text: str):
        nonlocal current_y
        if current_y > max_y: return

        current_y += 10
        draw.text((margin_x, current_y), text.upper(), font=_resolve_font(32, bold=True), fill=c_indigo_500)
        current_y += 40
        
    def draw_key_takeaway(text: str):
        nonlocal current_y
        if current_y > max_y: return

        # Dark box with gradient border feel
        wrapped = _wrap_text(draw, text, body_font, content_width - 60)
        _, _, _, lh = draw.textbbox((0, 0), "A", font=body_font)
        box_h = len(wrapped) * (lh + 6) + 60
        
        draw.rounded_rectangle(
            [(margin_x, current_y), (margin_x + content_width, current_y + box_h)],
            radius=20,
            fill=c_gray_900,
            outline=c_pink_500,
            width=3
        )
        
        label_font = _resolve_font(24, bold=True)
        draw.text((margin_x + 30, current_y + 20), "KEY TAKEAWAY", font=label_font, fill=c_pink_500)
        
        text_y = current_y + 50
        for line in wrapped:
            draw.text((margin_x + 30, text_y), line, font=body_font, fill=c_white)
            text_y += lh + 6
        current_y += box_h + 20

    def draw_diagram(caption: str):
        nonlocal current_y
        if current_y > max_y: return

        # Placeholder for diagram
        box_h = 300 # Smaller height
        draw.rectangle(
            [(margin_x + 40, current_y), (margin_x + content_width - 40, current_y + box_h)],
            fill=c_gray_50,
            outline=c_gray_600,
            width=2
        )
        
        # Icon
        cx = margin_x + content_width // 2
        cy = current_y + box_h // 2
        r = 60
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=c_indigo_500, width=6)
        draw.line([(cx - 40, cy), (cx + 40, cy)], fill=c_indigo_500, width=6)
        draw.line([(cx, cy - 40), (cx, cy + 40)], fill=c_indigo_500, width=6)
        
        if caption:
            draw.text((margin_x + 60, current_y + box_h - 40), f"Diagram: {caption}", font=_resolve_font(28), fill=c_gray_600)
            
        current_y += box_h + 30

    # Draw Blocks
    if slide.content_blocks:
        for block in slide.content_blocks:
             if not isinstance(block, dict): continue
             btype = (block.get("type") or "").lower()
             
             if btype == "code":
                 draw_code_block(block.get("code", ""), block.get("language", ""))
             elif btype == "table":
                 draw_table(block.get("headers", []), block.get("rows", []))
             elif btype in ["bullets", "checklist"]:
                 for item in block.get("items", []):
                     draw_styled_bullet(str(item))
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
             # Add other types as needed
             else:
                  draw_styled_bullet(str(block.get("text", "")))
    else:
        # Fallback text parsing
        raw = (slide.content or "").strip()
        lines = [ln.strip("-• \t") for ln in raw.splitlines() if ln.strip()]
        if not lines:
             lines = _wrap_text(draw, raw, body_font, content_width)
             
        for line in lines:
            if line.lower().startswith("tip:") or line.lower().startswith("note:"):
                draw_callout(line)
            else:
                draw_styled_bullet(line)
            
    # 6. Speaker Notes / Footer (Botttom Bar style)
    if slide_spec_notes := getattr(slide, 'notes', None): # Assuming notes might be added to SlideSpec logic in future, currently absent in SlideSpec definition but present in calling code?
        # Actually SlideSpec definition in line 37 doesn't have notes.
        # But _pair_presentation_and_narration creates SlideSpec.
        # Check SlideSpec definition.
        pass

    # Draw Course Title at bottom left
    if course_title:
        draw.text((margin_x, H - 80), course_title.upper(), font=small_font, fill=(200, 200, 200))

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
