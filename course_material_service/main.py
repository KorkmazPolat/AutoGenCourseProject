from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import yaml
from dotenv import load_dotenv
from fastapi import BackgroundTasks, File, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Template
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from rag_ingest import IngestError, ingest_pdf_into_qdrant
from rag_retriever import RagRetrieverError, build_context as retrieve_rag_context
from video_builder import VideoGenerationError, generate_video_from_script
import shutil



class GenerationRequest(BaseModel):
    course_title: str = Field(..., description="Name of the course we are building.")
    learning_outcomes: List[str] = Field(
        ..., min_items=1, description="Desired learning outcomes phrased as learner capabilities."
    )
    audience: Optional[str] = Field(None, description="Target learner profile.")
    tone: Optional[str] = Field(None, description="Preferred tone of voice.")
    duration_minutes: Optional[int] = Field(None, gt=0, description="Target duration in minutes for the video lesson.")
    project_duration: Optional[str] = Field(None, description="Expected effort for the project assignment.")
    module_number: Optional[int] = Field(None, description="Module sequence number when generating per-module assets.")
    module_title: Optional[str] = Field(None, description="Module-specific title when generating per-module assets.")
    module_summary: Optional[str] = Field(None, description="Short description for the module focus.")
    module_topics: Optional[List[str]] = Field(
        None, description="Key topics or themes covered inside this module."
    )
    model: Optional[str] = Field(
        None,
        description="Optional OpenAI model override (defaults to gpt-4o-mini).",
    )

    # Accept numeric strings or empty strings for duration_minutes in JSON bodies
    @field_validator("duration_minutes", mode="before")
    @classmethod
    def _coerce_duration_minutes(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        if isinstance(v, int):
            return v
        try:
            return int(v)
        except Exception as exc:  # pragma: no cover
            raise ValueError("duration_minutes must be an integer") from exc


@dataclass
class PromptMessage:
    role: str
    template: Template


@dataclass
class PromptDefinition:
    description: str
    messages: List[PromptMessage]
    response_format: Optional[Dict[str, Any]] = None


class PromptPreviewResponse(BaseModel):
    prompt_id: str
    description: str
    messages: List[Dict[str, str]]
    response_format: Optional[Dict[str, Any]]


class MaterialResponse(PromptPreviewResponse):
    model: str
    content: Dict[str, Any]
    raw_text: str


@dataclass
class LLMOutput:
    model: str
    raw_text: str
    content: Dict[str, Any]


class MaterialGenerationRequest(GenerationRequest):
    prompt_id: str = Field(
        ..., description="Identifier of the prompt definition to execute."
    )
    create_video: Optional[bool] = Field(
        None,
        description="When true and the prompt is 'video_script', automatically render a narrated video asset.",
    )
    voice: Optional[str] = Field(
        None,
        description="Voice identifier supported by the configured TTS model (video generation only).",
    )
    tts_model: Optional[str] = Field(
        None,
        description="Text-to-speech model for narration synthesis (video generation only).",
    )
    theme: Optional[str] = Field(
        None,
        description="Visual theme for slides: 'dark' or 'light' (video generation only).",
    )
    logo_path: Optional[str] = Field(
        None,
        description="Optional path to a logo image to watermark slides (video generation only).",
    )


DEFAULT_MODEL = "gpt-4o-mini"
VIDEO_PROMPT_ID = "video_script"
VIDEO_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_videos"
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"


def _build_video_output_path(course_title: str) -> Path:
    safe_title = "".join(ch.lower() if ch.isalnum() else "_" for ch in course_title)
    safe_title = "_".join(filter(None, safe_title.split("_"))) or "course_video"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return (VIDEO_OUTPUT_DIR / f"{safe_title}_{timestamp}").with_suffix(".mp4")


def _slugify_title(title: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in title)
    safe = "-".join(filter(None, safe.split("-")))
    return safe or "course"


def _keep_uploaded_files() -> bool:
    toggle = os.getenv("RAG_KEEP_UPLOADS", "true").strip().lower()
    return toggle not in {"0", "false", "off", "no"}


def _load_prompt_definitions() -> Dict[str, PromptDefinition]:
    """Read prompt templates from YAML and compile them with Jinja2."""
    prompt_path = Path(__file__).resolve().parent / "prompts.yaml"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt definition file not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as handle:
        raw_prompts = yaml.safe_load(handle)

    definitions: Dict[str, PromptDefinition] = {}
    for key, value in raw_prompts.items():
        try:
            description = value["description"]
            messages = value["messages"]
        except KeyError as exc:
            raise ValueError(f"Prompt '{key}' is missing required field: {exc}") from exc

        compiled_messages = [
            PromptMessage(
                role=message["role"],
                template=Template(message["template"], trim_blocks=True, lstrip_blocks=True),
            )
            for message in messages
        ]

        definitions[key] = PromptDefinition(
            description=description,
            messages=compiled_messages,
            response_format=value.get("response_format"),
        )

    return definitions


PROMPT_DEFINITIONS = _load_prompt_definitions()

load_dotenv(dotenv_path=".env", override=True)


def _parse_learning_outcomes(raw_text: str) -> List[str]:
    """Split the textarea input into clean outcome lines."""
    outcomes = [
        line.lstrip("-• ").strip()
        for line in (raw_text or "").splitlines()
        if line.strip()
    ]
    if not outcomes:
        raise HTTPException(status_code=400, detail="At least one learning outcome is required.")
    return outcomes


def _rag_enabled() -> bool:
    toggle = os.getenv("RAG_ENABLED", "true").strip().lower()
    return toggle not in {"0", "false", "off", "no"}


def _compose_rag_query(prompt_id: str, payload: GenerationRequest) -> str:
    lines: List[str] = []
    if payload.course_title:
        lines.append(f"Course Title: {payload.course_title}")
    if payload.module_title:
        lines.append(f"Module: {payload.module_title}")
    if payload.module_summary:
        lines.append(f"Module Summary: {payload.module_summary}")
    if payload.audience:
        lines.append(f"Audience: {payload.audience}")
    if payload.tone:
        lines.append(f"Tone: {payload.tone}")
    if payload.duration_minutes:
        lines.append(f"Target Duration: {payload.duration_minutes} minutes")
    if payload.learning_outcomes:
        lines.append("Learning Outcomes:")
        lines.extend(f"- {outcome}" for outcome in payload.learning_outcomes)
    lines.append(f"Prompt Requested: {prompt_id}")
    return "\n".join(lines).strip()


def _build_rag_context(prompt_id: str, payload: GenerationRequest) -> str:
    if not _rag_enabled():
        return ""

    try:
        query = _compose_rag_query(prompt_id, payload)
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to compose RAG query for %s: %s", prompt_id, exc)
        return ""

    if not query:
        return ""

    try:
        top_k = int(os.getenv("QDRANT_TOP_K", "4"))
    except ValueError:
        top_k = 4
    try:
        max_chars = int(os.getenv("QDRANT_MAX_CHARS", "1800"))
    except ValueError:
        max_chars = 1800

    min_score_env = os.getenv("QDRANT_MIN_SCORE")
    min_score: Optional[float]
    try:
        min_score = float(min_score_env) if min_score_env else None
    except (TypeError, ValueError):
        min_score = None

    try:
        context = retrieve_rag_context(query, top_k=top_k, max_chars=max_chars, min_score=min_score)
    except RagRetrieverError as exc:
        logger.warning("RAG retrieval skipped for prompt '%s': %s", prompt_id, exc)
        return ""
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected RAG retrieval failure for prompt '%s'", prompt_id)
        return ""

    return context.strip()


def _safe_upload_filename(original_name: Optional[str]) -> str:
    stem = Path(original_name or "document").stem
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in stem)
    normalized = "-".join(filter(None, normalized.split("-"))) or "document"
    suffix = Path(original_name or "").suffix.lower() or ".pdf"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{uuid4().hex[:8]}_{normalized}{suffix}"


async def _persist_upload(upload: UploadFile) -> Path:
    filename = _safe_upload_filename(upload.filename)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOADS_DIR / filename

    total_bytes = 0
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            buffer.write(chunk)
            total_bytes += len(chunk)

    await upload.close()

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    return destination


def _cleanup_upload(path: Path) -> None:
    if _keep_uploaded_files():
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:  # pragma: no cover
        logger.debug("Failed to remove temporary upload %s", path)


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable must be set for generation.")
    return OpenAI(api_key=api_key)


app = FastAPI(
    title="Course Material Prompt Service",
    description="FastAPI service providing course content prompts plus a simple web UI for generating narrated videos.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory="static"), name="static")




def _coerce_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail="duration_minutes must be an integer") from exc


@app.get("/health", tags=["system"])
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/prompts", tags=["prompts"])
def list_prompts() -> List[Dict[str, str]]:
    return [
        {
            "id": prompt_id,
            "description": definition.description,
        }
        for prompt_id, definition in PROMPT_DEFINITIONS.items()
    ]


def _build_messages(
    prompt_id: str,
    payload: GenerationRequest,
    rag_context: Optional[str] = None,
) -> List[Dict[str, str]]:
    definition = PROMPT_DEFINITIONS.get(prompt_id)
    if not definition:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found.")

    template_context = payload.dict()
    duration = template_context.get("duration_minutes")
    if not duration:
        duration = 10
    target_word_count = max(400, int(duration * 135))
    target_segment_count = max(4, int(round(duration)))
    template_context["target_word_count"] = target_word_count
    template_context["target_segment_count"] = target_segment_count

    template_context = {
        key: value for key, value in template_context.items() if value not in (None, "", [])
    }
    rendered_messages = []
    for message in definition.messages:
        rendered_messages.append(
            {
                "role": message.role,
                "content": message.template.render(**template_context),
            }
        )

    if rag_context:
        context_message = {
            "role": "system",
            "content": (
                "Consult the following excerpts from the uploaded course materials. "
                "Ground all generated output in these sources when possible:\n\n"
                f"{rag_context}"
            ),
        }
        for index, message in enumerate(rendered_messages):
            if message["role"] == "user":
                rendered_messages.insert(index, context_message)
                break
        else:
            rendered_messages.append(context_message)

    return rendered_messages


def _call_openai(
    prompt_id: str,
    payload: GenerationRequest,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, Any]],
) -> LLMOutput:
    try:
        client = _get_openai_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    model_name = payload.model or DEFAULT_MODEL
    request_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if response_format:
        request_kwargs["response_format"] = {"type": "json_object"}

    try:
        completion = client.chat.completions.create(**request_kwargs)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI chat completion failed for '{prompt_id}': {exc}",
        ) from exc

    choice = completion.choices[0]
    message = choice.message
    content = message.content

    if isinstance(content, list):
        text_chunks: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_chunks.append(part.get("text", ""))
        raw_text = "".join(text_chunks).strip()
    else:
        raw_text = (content or "").strip()  # type: ignore[arg-type]

    if not raw_text:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI returned empty output for prompt '{prompt_id}'.",
        )

    try:
        parsed_content = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI returned non-JSON output for prompt '{prompt_id}': {exc}",
        ) from exc

    completion_model = getattr(completion, "model", model_name)

    return LLMOutput(
        model=completion_model,
        raw_text=raw_text,
        content=parsed_content,
    )


def _generate_material_payload(
    prompt_id: str,
    payload: GenerationRequest,
    preview: bool,
    *,
    create_video: Optional[bool] = None,
    voice: Optional[str] = None,
    tts_model: Optional[str] = None,
    theme: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> Dict[str, Any]:
    definition = PROMPT_DEFINITIONS.get(prompt_id)
    if not definition:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found.")

    rag_context = _build_rag_context(prompt_id, payload)

    messages = _build_messages(prompt_id, payload, rag_context=rag_context)

    if preview:
        preview_payload = PromptPreviewResponse(
            prompt_id=prompt_id,
            description=definition.description,
            messages=messages,
            response_format=definition.response_format,
        )
        return preview_payload.dict()

    llm_output = _call_openai(prompt_id, payload, messages, definition.response_format)
    response = MaterialResponse(
        prompt_id=prompt_id,
        description=definition.description,
        messages=messages,
        response_format=definition.response_format,
        model=llm_output.model,
        content=llm_output.content,
        raw_text=llm_output.raw_text,
    )
    result = response.dict()
    if rag_context:
        result["rag_context"] = rag_context

    auto_video = create_video if create_video is not None else (prompt_id == VIDEO_PROMPT_ID)
    if auto_video:
        if preview:
            raise HTTPException(status_code=400, detail="Preview mode cannot render video assets.")
        if prompt_id != VIDEO_PROMPT_ID:
            raise HTTPException(status_code=400, detail="Video creation is only supported for the 'video_script' prompt.")
        try:
            VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            client = _get_openai_client()
            output_path = _build_video_output_path(payload.course_title)
            video_path = generate_video_from_script(
                video_payload=response.content,
                output_path=output_path,
                client=client,
                voice=voice or "alloy",
                tts_model=tts_model or "gpt-4o-mini-tts",
                course_title=payload.course_title,
                theme=(theme or "dark"),
                logo_path=logo_path,
            )
        except VideoGenerationError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"Unexpected video generation failure: {exc}") from exc
        result["video_file"] = str(video_path)
        # Attach sidecar caption/chapter files when present
        captions_file = video_path.with_suffix(".vtt")
        chapters_file = video_path.with_name(video_path.stem + ".chapters.vtt")
        if captions_file.exists():
            result["captions_file"] = str(captions_file)
        if chapters_file.exists():
            result["chapters_file"] = str(chapters_file)

    return result


@app.post("/materials/{prompt_id}", tags=["materials"])
def generate_material(
    prompt_id: str,
    payload: GenerationRequest,
    preview: bool = False,
    create_video: Optional[bool] = None,
    voice: Optional[str] = None,
    tts_model: Optional[str] = None,
    theme: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> Dict[str, Any]:
    return _generate_material_payload(
        prompt_id,
        payload,
        preview,
        create_video=create_video,
        voice=voice,
        tts_model=tts_model,
        theme=theme,
        logo_path=logo_path,
    )


@app.post("/materials", tags=["materials"])
def generate_material_from_request(request: MaterialGenerationRequest, preview: bool = False) -> Dict[str, Any]:
    payload = GenerationRequest(**request.dict(exclude={"prompt_id", "create_video", "voice", "tts_model", "theme", "logo_path"}))
    return _generate_material_payload(
        request.prompt_id,
        payload,
        preview,
        create_video=request.create_video,
        voice=request.voice,
        tts_model=request.tts_model,
        theme=request.theme,
        logo_path=request.logo_path,
    )


@app.post("/materials/all", tags=["materials"])
def generate_all_materials(
    payload: GenerationRequest,
    preview: bool = False,
    create_video: Optional[bool] = None,
    voice: Optional[str] = None,
    tts_model: Optional[str] = None,
    theme: Optional[str] = None,
    logo_path: Optional[str] = None,
) -> Dict[str, Any]:
    prompts: Dict[str, Dict[str, Any]] = {}
    for prompt_id in PROMPT_DEFINITIONS.keys():
        prompts[prompt_id] = _generate_material_payload(
            prompt_id,
            payload,
            preview,
            create_video=create_video if prompt_id == VIDEO_PROMPT_ID else False if create_video is not None else None,
            voice=voice,
            tts_model=tts_model,
            theme=theme,
            logo_path=logo_path,
        )

    return {
        "course_title": payload.course_title,
        "materials": prompts,
    }


@app.get("/", response_class=HTMLResponse)
def render_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "values": {}})


@app.post("/documents/upload", tags=["rag"])
async def upload_course_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    content_type = (file.content_type or "").lower()
    filename = file.filename or "document.pdf"
    if "pdf" not in content_type and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for ingestion.")

    try:
        saved_path = await _persist_upload(file)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to persist uploaded document %s", filename)
        raise HTTPException(status_code=500, detail="Could not store uploaded file.") from exc

    try:
        stats = await run_in_threadpool(ingest_pdf_into_qdrant, saved_path)
    except IngestError as exc:
        _cleanup_upload(saved_path)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        _cleanup_upload(saved_path)
        logger.exception("Document ingestion failed for %s", saved_path)
        raise HTTPException(status_code=500, detail="Failed to index the uploaded PDF.") from exc

    background_tasks.add_task(_cleanup_upload, saved_path)

    return {
        "status": "indexed",
        "filename": stats.filename,
        "pages": stats.pages,
        "chunks": stats.chunks,
        "collection": stats.collection,
    }


# Removed dashboard route as dashboard UI no longer exists



@app.post("/create-course-video", response_class=HTMLResponse, tags=["web"])
def create_course_video(
    request: Request,
    course_title: str = Form(...),
    learning_outcomes: str = Form(...),
    audience: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    duration_minutes: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    tts_model: Optional[str] = Form(None),
    theme: Optional[str] = Form(None),
    logo_path: Optional[str] = Form(None),
) -> HTMLResponse:
    outcomes = _parse_learning_outcomes(learning_outcomes)

    payload = GenerationRequest(
        course_title=course_title,
        learning_outcomes=outcomes,
        audience=audience,
        tone=tone,
        duration_minutes=_coerce_optional_int(duration_minutes),
    )

    result = _generate_material_payload(
        VIDEO_PROMPT_ID,
        payload,
        preview=False,
        create_video=True,
        voice=voice,
        tts_model=tts_model,
        theme=theme,
        logo_path=logo_path,
    )

    video_path = result.get("video_file")
    video_url = ""
    captions_url = ""
    chapters_url = ""
    if video_path:
        path_obj = Path(video_path)
        video_url = f"/videos/{path_obj.name}"
        cap_file = result.get("captions_file")
        chap_file = result.get("chapters_file")
        if cap_file:
            captions_url = f"/videos/{Path(cap_file).name}"
    if chap_file:
        chapters_url = f"/videos/{Path(chap_file).name}"

    content = result.get("content", {})
    rag_context = result.get("rag_context")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "course_title": course_title,
            "hook": content.get("hook", ""),
            "recap": content.get("recap", ""),
            "outline": content.get("outline", []),
            "video_url": video_url,
            "captions_url": captions_url,
            "chapters_url": chapters_url,
            "voice": voice or "alloy",
            "model": result.get("model"),
            "video_file": Path(video_path).name if video_path else None,
            "rag_context": rag_context,
        },
    )


@app.post("/create-course-plan", response_class=HTMLResponse, tags=["web"])
def create_course_plan(
    request: Request,
    course_title: str = Form(...),
    learning_outcomes: str = Form(...),
    audience: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    duration_minutes: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),  # unused but accepted so the form can submit same fields
    tts_model: Optional[str] = Form(None),  # unused but accepted so the form can submit same fields
) -> HTMLResponse:
    outcomes = _parse_learning_outcomes(learning_outcomes)
    payload = GenerationRequest(
        course_title=course_title,
        learning_outcomes=outcomes,
        audience=audience,
        tone=tone,
        duration_minutes=_coerce_optional_int(duration_minutes),
    )

    blueprint_result = _generate_material_payload(
        "course_blueprint",
        payload,
        preview=False,
        create_video=False,
    )
    blueprint_content = blueprint_result.get("content", {})

    return templates.TemplateResponse(
        "course_plan.html",
        {
            "request": request,
            "course_title": course_title,
            "audience": audience,
            "tone": tone,
            "duration_minutes": duration_minutes,
            "learning_outcomes": outcomes,
            "learning_outcomes_text": learning_outcomes,
            "blueprint": blueprint_content,
        },
    )


@app.post("/create-course-pages", response_class=HTMLResponse, tags=["web"])
def create_course_pages(
    request: Request,
    course_title: str = Form(...),
    learning_outcomes: str = Form(...),
    audience: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    duration_minutes: Optional[str] = Form(None),
    blueprint_json: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    tts_model: Optional[str] = Form(None),
    theme: Optional[str] = Form(None),
    logo_path: Optional[str] = Form(None),
) -> HTMLResponse:
    outcomes = _parse_learning_outcomes(learning_outcomes)

    payload = GenerationRequest(
        course_title=course_title,
        learning_outcomes=outcomes,
        audience=audience,
        tone=tone,
        duration_minutes=_coerce_optional_int(duration_minutes),
    )

    blueprint_content: Dict[str, Any]
    if blueprint_json:
        try:
            blueprint_content = json.loads(blueprint_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid blueprint data: {exc}") from exc
    else:
        blueprint_result = _generate_material_payload(
            "course_blueprint",
            payload,
            preview=False,
            create_video=False,
        )
        blueprint_content = blueprint_result.get("content", {})

    modules = blueprint_content.get("modules") or []
    module_assets: List[Dict[str, Any]] = []

    for module in modules:
        module_number = module.get("number")
        module_title = module.get("title") or f"Module {module_number}"
        module_outcomes = module.get("outcomes") or outcomes
        module_payload = GenerationRequest(
            course_title=f"{course_title} - Module {module_number}: {module_title}",
            learning_outcomes=module_outcomes,
            audience=audience,
            tone=tone,
            duration_minutes=duration_minutes,
            module_number=module_number,
            module_title=module_title,
            module_summary=module.get("summary"),
            module_topics=module.get("key_topics"),
        )

        assets_for_module: Dict[str, Any] = {}
        for prompt_id in ("video_script", "reading_material", "quiz_questions"):
            assets_for_module[prompt_id] = _generate_material_payload(
                prompt_id,
                module_payload,
                preview=False,
                create_video=False,
                voice=voice,
                tts_model=tts_model,
                theme=theme,
                logo_path=logo_path,
            )
        module_assets.append(
            {
                "info": module,
                "assets": assets_for_module,
            }
        )

    return templates.TemplateResponse(
        "course_pages.html",
        {
            "request": request,
            "course_title": course_title,
            "audience": audience,
            "tone": tone,
            "duration_minutes": duration_minutes,
            "learning_outcomes": outcomes,
            "blueprint": blueprint_content,
            "modules_output": module_assets,
        },
    )


@app.get("/videos/{filename}", tags=["web"])
def serve_video(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = VIDEO_OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found.")
    suffix = file_path.suffix.lower()
    media = "application/octet-stream"
    if suffix == ".mp4":
        media = "video/mp4"
    elif suffix == ".vtt":
        media = "text/vtt"
    elif suffix == ".mp3":
        media = "audio/mpeg"
    return FileResponse(file_path, media_type=media)


@app.post("/create-full-course", response_class=HTMLResponse, tags=["web"])
def create_full_course(
    request: Request,
    course_title: str = Form(...),
    learning_outcomes: str = Form(...),
    audience: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    duration_minutes: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    tts_model: Optional[str] = Form(None),
    theme: Optional[str] = Form(None),
    logo_path: Optional[str] = Form(None),
) -> HTMLResponse:
    """Build a complete course: blueprint + per-module videos, readings, and quizzes."""
    outcomes = _parse_learning_outcomes(learning_outcomes)

    payload = GenerationRequest(
        course_title=course_title,
        learning_outcomes=outcomes,
        audience=audience,
        tone=tone,
        duration_minutes=_coerce_optional_int(duration_minutes),
    )

    # 1) Generate blueprint
    blueprint_result = _generate_material_payload(
        "course_blueprint",
        payload,
        preview=False,
        create_video=False,
    )
    blueprint_content = blueprint_result.get("content", {})

    # 2) For each module, generate assets and render a video
    modules = blueprint_content.get("modules") or []
    module_assets: List[Dict[str, Any]] = []

    for module in modules:
        module_number = module.get("number")
        module_title = module.get("title") or f"Module {module_number}"
        module_outcomes = module.get("outcomes") or outcomes
        module_payload = GenerationRequest(
            course_title=f"{course_title} - Module {module_number}: {module_title}",
            learning_outcomes=module_outcomes,
            audience=audience,
            tone=tone,
            duration_minutes=duration_minutes,
            module_number=module_number,
            module_title=module_title,
            module_summary=module.get("summary"),
            module_topics=module.get("key_topics"),
        )

        assets_for_module: Dict[str, Any] = {}

        # Generate and render video for this module
        video_result = _generate_material_payload(
            "video_script",
            module_payload,
            preview=False,
            create_video=True,
            voice=voice,
            tts_model=tts_model,
            theme=theme,
            logo_path=logo_path,
        )
        video_file = video_result.get("video_file")
        video_url = ""
        captions_url = ""
        chapters_url = ""
        if video_file:
            video_url = f"/videos/{Path(video_file).name}"
        cap_file = video_result.get("captions_file")
        chap_file = video_result.get("chapters_file")
        if cap_file:
            captions_url = f"/videos/{Path(cap_file).name}"
        if chap_file:
            chapters_url = f"/videos/{Path(chap_file).name}"
        video_result["video_url"] = video_url
        if captions_url:
            video_result["captions_url"] = captions_url
        if chapters_url:
            video_result["chapters_url"] = chapters_url
        assets_for_module["video_script"] = video_result

        # Reading material
        assets_for_module["reading_material"] = _generate_material_payload(
            "reading_material",
            module_payload,
            preview=False,
            create_video=False,
            voice=voice,
            tts_model=tts_model,
        )

        # Quiz questions
        assets_for_module["quiz_questions"] = _generate_material_payload(
            "quiz_questions",
            module_payload,
            preview=False,
            create_video=False,
            voice=voice,
            tts_model=tts_model,
        )

        module_assets.append(
            {
                "info": module,
                "assets": assets_for_module,
            }
        )

    # Side-effect: save a standalone HTML copy into exports folder (non-blocking)
    try:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        export_name = f"{_slugify_title(course_title)}-{timestamp}"
        export_dir = EXPORTS_DIR / export_name
        videos_dir = export_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        # Prepare assets with relative URLs and copy media
        exported_modules: List[Dict[str, Any]] = []
        for mod in module_assets:
            info = mod.get("info", {})
            assets = mod.get("assets", {})
            video_pack = dict(assets.get("video_script", {}) or {})
            for key in ("video_file", "captions_file", "chapters_file"):
                src = video_pack.get(key)
                if not src:
                    continue
                p = Path(src)
                if p.exists():
                    dest = videos_dir / p.name
                    try:
                        shutil.copy2(p, dest)
                    except Exception:
                        pass
                    if key == "video_file":
                        video_pack["video_url"] = f"videos/{dest.name}"
                    elif key == "captions_file":
                        video_pack["captions_url"] = f"videos/{dest.name}"
                    elif key == "chapters_file":
                        video_pack["chapters_url"] = f"videos/{dest.name}"
            new_assets = dict(assets)
            new_assets["video_script"] = video_pack
            exported_modules.append({"info": info, "assets": new_assets})

        # Render and write HTML using same template but without request
        env = getattr(templates, "env", None) or getattr(templates, "environment", None)
        if env is not None:
            template = env.get_template("full_course.html")
            html = template.render(
                course_title=course_title,
                audience=audience,
                tone=tone,
                duration_minutes=duration_minutes,
                learning_outcomes=outcomes,
                blueprint=blueprint_content,
                modules_output=exported_modules,
                voice=voice or "alloy",
            )
            (export_dir / "index.html").write_text(html, encoding="utf-8")
    except Exception:
        # Ignore export errors; interactive flow continues
        pass

    return templates.TemplateResponse(
        "full_course.html",
        {
            "request": request,
            "course_title": course_title,
            "audience": audience,
            "tone": tone,
            "duration_minutes": duration_minutes,
            "learning_outcomes": outcomes,
            "blueprint": blueprint_content,
            "modules_output": module_assets,
            "voice": voice or "alloy",
        },
    )


@app.get("/exported/{export_name}/{path:path}", tags=["web"])
def serve_exported_file(export_name: str, path: str = "index.html") -> FileResponse:
    base = (EXPORTS_DIR / Path(export_name)).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Exported file not found.")
    suffix = target.suffix.lower()
    media = "application/octet-stream"
    if suffix in (".html", ".htm"):
        media = "text/html"
    elif suffix == ".css":
        media = "text/css"
    elif suffix == ".js":
        media = "application/javascript"
    elif suffix == ".mp4":
        media = "video/mp4"
    elif suffix == ".vtt":
        media = "text/vtt"
    elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        media = f"image/{suffix.lstrip('.')}"
    return FileResponse(target, media_type=media)


@app.get("/exports", response_class=HTMLResponse, tags=["web"])
def list_exports(request: Request):
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, str]] = []
    for entry in sorted(EXPORTS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        if entry.is_dir():
            items.append({
                "name": entry.name,
                "path": str(entry),
                "url": f"/exported/{entry.name}/index.html",
            })
    # Render a simple list page (amateur look)
    html = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>Exports</title></head><body>",
        "<h1>Saved Courses</h1>",
        "<ul>",
    ]
    for it in items:
        html.append(f"<li><a href='{it['url']}'>{it['name']}</a></li>")
    html += ["</ul>", "<p><a href='/'>&larr; back</a></p>", "</body></html>"]
    return HTMLResponse("\n".join(html))
