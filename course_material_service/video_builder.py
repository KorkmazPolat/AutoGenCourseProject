from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .renderer import SlideRenderer
from .assembler import VideoAssembler


class VideoGenerationError(RuntimeError):
    """Raised when the automated video generation pipeline fails."""


@dataclass(frozen=True)
class SlideSpec:
    page: int
    heading: str
    content: str
    visual_hints: Optional[List[str]] = None
    content_blocks: Optional[List[Dict[str, Any]]] = None


def _synthesize_audio_clip(
    client: Any,
    script: str,
    output_path: Path,
    *,
    voice: str = "nova",
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
            print(f"Fallback TTS failed ({fallback_exc}). Generating 1s silence using FFmpeg.")
            try:
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", 
                    "-t", "1", "-c:a", "libmp3lame", str(output_path)
                ], check=True, capture_output=True)
            except Exception as final_exc: 
                raise VideoGenerationError(f"All TTS attempts failed and FFmpeg silence generation failed: {final_exc}") from exc

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


async def generate_video_from_script(
    *,
    video_payload: Dict[str, Any],
    output_path: Path,
    client: Any,
    voice: str = "nova",
    tts_model: str = "gpt-4o-mini-tts",
    fps: int = 30,
    course_title: Optional[str] = None,
    theme: str = "dark",
    logo_path: Optional[str] = None,
    add_intro_outro: bool = True,
    pause_between_slides: float = 0.4,
) -> Path:
    """Generate a narrated slide video using HTML/CSS renderer and FFmpeg assembler."""
    presentation = video_payload.get("presentation", [])
    narration = video_payload.get("narration", [])
    pairs = _pair_presentation_and_narration(presentation, narration)

    output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Prepare components
        templates_dir = Path(__file__).resolve().parent / "templates"
        renderer = SlideRenderer(templates_dir)
        assembler = VideoAssembler(fps=fps)
        
        # Copy style.css to tmp_dir so Playwright can find it
        shutil.copy(templates_dir / "slides" / "style.css", tmp_path / "style.css")

        sequence: List[Tuple[SlideSpec, str]] = []
        if add_intro_outro:
            intro_heading = (course_title or "Course Intro").strip()
            intro_content = (video_payload.get("hook") or "").strip()
            if intro_content:
                sequence.append((SlideSpec(page=0, heading=intro_heading, content=intro_content), intro_content))

        sequence.extend(pairs)

        if add_intro_outro:
            cta = (video_payload.get("call_to_action") or "").strip()
            if cta:
                sequence.append((SlideSpec(page=99, heading="Next Steps", content=cta), cta))

        total_pages = len(sequence)
        slide_images: List[Path] = []
        slide_audios: List[Path] = []
        slide_scripts: List[str] = []
        slide_headings: List[str] = []
        slide_durations: List[float] = []

        semaphore = asyncio.Semaphore(3) # Limit parallel rendering to 3 browsers max
        
        async def _render_and_synthesize(index: int, spec: SlideSpec, script: str):
            img_path = tmp_path / f"slide_{index:02d}.png"
            aud_path = tmp_path / f"audio_{index:02d}.mp3"
            
            # Prepare Render Slide
            slide_data = {
                "heading": spec.heading,
                "content": spec.content,
                "content_blocks": spec.content_blocks
            }
            
            async with semaphore:
                # Render HTML Slide to PNG
                await renderer.render_slide(
                    slide_data, 
                    img_path, 
                    course_title=course_title or "Course", 
                    current_page=index, 
                    page_count=total_pages
                )
            
            # Synthesize Audio (Offload to thread as it's sync)
            await asyncio.to_thread(
                _synthesize_audio_clip, 
                client=client, 
                script=script, 
                output_path=aud_path, 
                voice=voice, 
                tts_model=tts_model
            )
            
            # Get audio duration
            import ffmpeg
            probe = await asyncio.to_thread(ffmpeg.probe, str(aud_path))
            duration = float(probe['format']['duration'])
            
            return {
                "index": index,
                "img_path": img_path,
                "aud_path": aud_path,
                "script": script,
                "heading": spec.heading,
                "duration": duration
            }

        # Start persistent browser
        await renderer.start()
        
        try:
            # Run parallel rendering & synthesis
            tasks = []
            for index, (spec, script) in enumerate(sequence, start=1):
                tasks.append(_render_and_synthesize(index, spec, script))
                
            results = await asyncio.gather(*tasks)
            # Sort results by index to maintain order
            results.sort(key=lambda x: x["index"])
        finally:
            # Always stop the browser
            await renderer.stop()
        
        for r in results:
            slide_images.append(r["img_path"])
            slide_audios.append(r["aud_path"])
            slide_scripts.append(r["script"])
            slide_headings.append(r["heading"])
            slide_durations.append(r["duration"])

        # Assemble Video using FFmpeg
        metadata = {
            "title": course_title or "Course Video",
            "comment": "Generated by AutoGenCourseProject"
        }
        assembler.assemble_video(
            slide_images, 
            slide_audios, 
            output_path, 
            metadata=metadata
        )

        # Generate sidecar files
        vtt_path = output_path.with_suffix(".vtt")
        assembler.generate_subtitles_vtt(slide_scripts, slide_durations, slide_headings, vtt_path)
        
        # Chapters (Optional, reusing VTT for now as a simple sidecar)
        chapters_path = output_path.with_name(output_path.stem + ".chapters.vtt")
        assembler.generate_subtitles_vtt(slide_scripts, slide_durations, slide_headings, chapters_path)

    return output_path


async def generate_slides_and_audio(
    *,
    video_payload: Dict[str, Any],
    output_dir: Path,
    client: Any,
    voice: str = "nova",
    tts_model: str = "gpt-4o-mini-tts",
    course_title: Optional[str] = None,
    theme: str = "dark",
    logo_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates slide images and audio files individually to output_dir using SlideRenderer.
    """
    presentation = video_payload.get("presentation", [])
    narration = video_payload.get("narration", [])
    pairs = _pair_presentation_and_narration(presentation, narration)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare components
    templates_dir = Path(__file__).resolve().parent / "templates"
    renderer = SlideRenderer(templates_dir)
    
    # Copy style.css to output_dir so Playwright can find it locally if needed
    shutil.copy(templates_dir / "slides" / "style.css", output_dir / "style.css")

    sequence: List[Tuple[SlideSpec, str]] = []
    intro_content = (video_payload.get("hook") or "").strip()
    if intro_content:
        sequence.append((SlideSpec(page=0, heading=course_title or "Introduction", content=intro_content), intro_content))

    sequence.extend(pairs)

    cta = (video_payload.get("call_to_action") or "").strip()
    if cta:
        sequence.append((SlideSpec(page=999, heading="Next Steps", content=cta), cta))

    total_pages = len(sequence)
    slides_manifest = []

    semaphore = asyncio.Semaphore(3)

    # Start persistent browser
    await renderer.start()
    
    try:
        async def _render_and_synthesize_individual(index: int, slide: SlideSpec, script: str):
            img_name = f"slide_{index:02d}.png"
            audio_name = f"audio_{index:02d}.mp3"
            slide_path = output_dir / img_name
            audio_path = output_dir / audio_name

            # Prepare Render Slide
            slide_data = {
                "heading": slide.heading,
                "content": slide.content,
                "content_blocks": slide.content_blocks
            }
            
            async with semaphore:
                # Render HTML Slide to PNG
                await renderer.render_slide(
                    slide_data, 
                    slide_path, 
                    course_title=course_title or "Course", 
                    current_page=index, 
                    page_count=total_pages
                )
            
            # Synthesize Audio (Offload to thread as it's sync)
            await asyncio.to_thread(
                _synthesize_audio_clip, 
                client=client, 
                script=script, 
                output_path=audio_path, 
                voice=voice, 
                tts_model=tts_model
            )
            
            return {
                "index": index,
                "image_file": img_name,
                "audio_file": audio_name,
                "script": script,
                "heading": slide.heading
            }

        tasks = []
        for index, (slide, script) in enumerate(sequence, start=1):
            tasks.append(_render_and_synthesize_individual(index, slide, script))
            
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x["index"])
    finally:
        # Always stop the browser
        await renderer.stop()

    slides_manifest.extend(results)

    return {
        "slides": slides_manifest,
        "base_dir": str(output_dir.name)
    }
