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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from dotenv import load_dotenv
from fastapi import BackgroundTasks, File, FastAPI, Form, HTTPException, Request, UploadFile, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware 
from jinja2 import Template
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from .rag_ingest import IngestError, ingest_pdf_into_qdrant
from .rag_retriever import RagRetrieverError, build_context as retrieve_rag_context
from .video_builder import VideoGenerationError, generate_video_from_script
from routes.generate_course import router as agent_course_router
from manager.agent_manager import AgentManager
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


# Markdown support enabled
import markdown

DEFAULT_MODEL = "gpt-4o-mini"
VIDEO_PROMPT_ID = "video_script"
VIDEO_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_videos"
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters['markdown'] = lambda text: markdown.markdown(text, extensions=['extra', 'codehilite'])

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

def _export_html(template_name: str, context: Dict[str, Any], *, base_slug: str) -> Path:
    """Render a template to HTML and write it under EXPORTS_DIR/<slug>-<timestamp>/index.html.

    The export relies on app routes for assets (e.g., /static, /videos). Only HTML is persisted.
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{_slugify_title(base_slug)}-{timestamp}"
    out_dir = EXPORTS_DIR / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Render with the provided context; requires 'request' in context for url_for in templates
    template = templates.get_template(template_name)
    html = template.render(context)
    output_file = out_dir / "index.html"
    output_file.write_text(html, encoding="utf-8")
    return output_file


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

# Ensure we load the package-local .env (course_material_service/.env)
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)


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
    # Default disabled unless RAG_ENABLED is explicitly truthy
    toggle = os.getenv("RAG_ENABLED", "false").strip().lower()
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
    # Print masked API key once (function is cached)
    try:
        masked = (
            f"{api_key[:4]}...{api_key[-4:]}" if isinstance(api_key, str) and len(api_key) > 8 else "***"
        )
        print(f"Using OPENAI_API_KEY: {masked}")
    except Exception:
        pass
    return OpenAI(api_key=api_key)


app = FastAPI(
    title="Course Material Prompt Service",
    description="FastAPI service providing course content prompts plus a simple web UI for generating narrated videos.",
    version="0.2.0",
)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "default_fallback_secret_key_if_not_set")
)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
app.mount("/exports-files", StaticFiles(directory=str(EXPORTS_DIR)), name="exports_files")
app.include_router(agent_course_router)

from .database import init_db, get_db
from .database import init_db
from . import models # Register models
from .auth import verify_password, get_password_hash
from .services import save_course_to_db

@app.on_event("startup")
async def on_startup():
    # Ensure static directories exist
    os.makedirs("course_material_service/static/videos", exist_ok=True)
    os.makedirs("course_material_service/static/images", exist_ok=True)


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

@app.get("/register", response_class=HTMLResponse, tags=["web"])
def render_register_form(request: Request):
    """Serves the registration page."""
    if "user_id" in request.session:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        "register.html", 
        {
            "request": request,
            "values": {},
            "current_year": datetime.utcnow().year
        }
    )

@app.post("/register", response_class=HTMLResponse, tags=["web"])
async def handle_register(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Handles user registration."""
    # Check if user exists
    result = await db.execute(select(models.User).where(models.User.email == email))
    existing_user = result.scalars().first()
    
    if existing_user:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error": "Email already registered.",
                "values": {"full_name": full_name, "email": email},
                "current_year": datetime.utcnow().year
            },
            status_code=400
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    new_user = models.User(
        email=email,
        hashed_password=hashed_password,
        full_name=full_name
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    # Log them in automatically
    request.session["user_id"] = new_user.id
    request.session["user_email"] = new_user.email
    
    return RedirectResponse(url="/dashboard", status_code=303)

from .dependencies import get_session_user
@app.get("/agent-pipeline", response_class=HTMLResponse, tags=["web"])
def agent_pipeline_page(request: Request, user_id: int = Depends(get_session_user)):
    """Visualize the agentic AutoGenCourseProject pipeline as an interactive tree."""
    return templates.TemplateResponse(
        "agent_pipeline.html",
        {
            "request": request,
        },
    )

@app.get("/use-cases", response_class=HTMLResponse, tags=["web"])
def render_use_cases(request: Request, user_id: int = Depends(get_session_user)):
    """Render the use case diagrams page."""
    return templates.TemplateResponse(
        "use_cases.html",
        {
            "request": request,
            "current_year": datetime.utcnow().year,
        },
    )

@app.get("/exports", response_class=HTMLResponse, tags=["web"])
def list_saved_exports(request: Request, user_id: int = Depends(get_session_user)):
    """List saved course export folders with links to their index pages."""
    exports: List[Dict[str, str]] = []
    if EXPORTS_DIR.exists():
        for entry in sorted(EXPORTS_DIR.iterdir()):
            if entry.is_dir():
                name = entry.name
                index_url = f"/exports-files/{name}/index.html"
                exports.append({
                    "name": name,
                    "index_url": index_url,
                })
    return templates.TemplateResponse(
        "exports.html",
        {
            "request": request,
            "exports": exports,
            "current_year": datetime.utcnow().year,
        },
    )

@app.get("/login", response_class=HTMLResponse, tags=["web"])
def render_login_form(request: Request):
    """Serves the professional login page."""
    if "user_id" in request.session:
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        "login.html", 
        {
            "request": request,
            "values": {},
            "current_year": datetime.utcnow().year
        }
    )

@app.get("/", response_class=HTMLResponse, tags=["web"])
def render_root(request: Request):
    """Redirects root (/) to the login page."""
    return RedirectResponse(url="/login", status_code=303)

@app.post("/publish-course/{course_id}", tags=["web"])
async def publish_course(course_id: int, db: AsyncSession = Depends(get_db), user_id: int = Depends(get_session_user)):
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Ensure user owns the course
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this course")
    
    course.is_published = not course.is_published
    await db.commit()
    return {"status": "success", "is_published": course.is_published}

@app.get("/dashboard", response_class=HTMLResponse, tags=["web"])
async def render_dashboard(request: Request, user_id: int = Depends(get_session_user)):
    """Serves the main application page (course generator form)."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "values": {},
            "current_year": datetime.utcnow().year
        }
    )

@app.get("/library", response_class=HTMLResponse, tags=["web"])
async def render_library(request: Request, user_id: int = Depends(get_session_user), db: AsyncSession = Depends(get_db)):
    """Lists existing courses for the user."""
    result = await db.execute(select(models.Course).where(models.Course.user_id == user_id).order_by(models.Course.created_at.desc()))
    courses = result.scalars().all()

    return templates.TemplateResponse(
        "library.html",
        {
            "request": request,
            "courses": courses,
            "current_year": datetime.utcnow().year
        }
    )

@app.get("/course-builder/{course_id}", response_class=HTMLResponse, tags=["web"])
async def render_course_builder_with_data(
    request: Request, 
    course_id: int, 
    user_id: int = Depends(get_session_user), 
    db: AsyncSession = Depends(get_db)
):
    """Loads the course builder with specific course data."""
    # Fetch Course
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Fetch Modules
    result_mods = await db.execute(
        select(models.CourseModule).where(models.CourseModule.course_id == course_id).order_by(models.CourseModule.order_index)
    )
    modules = result_mods.scalars().all()

    course_data = {"modules": []}
    
    for module in modules:
        mod_data = {
            "id": str(module.id), # Use DB ID
            "title": module.title,
            "items": []
        }
        
        # Fetch Lessons
        result_lessons = await db.execute(
            select(models.Lesson).where(models.Lesson.module_id == module.id).order_by(models.Lesson.order_index)
        )
        lessons = result_lessons.scalars().all()
        
        for lesson in lessons:
            # Determine type based on assets
            result_assets = await db.execute(
                select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id)
            )
            assets = result_assets.scalars().all()
            
            item_type = "reading" # Default
            if any(a.asset_type == "video" for a in assets):
                item_type = "video"
            elif any(a.asset_type == "quiz" for a in assets):
                item_type = "quiz"
            
            mod_data["items"].append({
                "id": str(lesson.id),
                "type": item_type,
                "title": lesson.title,
                "desc": lesson.content[:100] + "..." if lesson.content else "No description"
            })
            
        course_data["modules"].append(mod_data)

    return templates.TemplateResponse(
        "course_builder.html",
        {
            "request": request,
            "course_data": json.dumps(course_data), # Pass as JSON string
            "course_id": course.id,
            "course_title": course.title
        }
    )

@app.post("/login", response_class=HTMLResponse, tags=["web"])
async def handle_login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Handles the login form submission."""
    result = await db.execute(select(models.User).where(models.User.email == email))
    user = result.scalars().first()
    
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Invalid email or password. Please try again.",
                "values": {"email": email},
                "current_year": datetime.utcnow().year
            },
            status_code=401
        )
    
    request.session["user_id"] = user.id
    request.session["user_email"] = user.email
    return RedirectResponse(url="/dashboard", status_code=303)
    
@app.get("/logout", response_class=HTMLResponse, tags=["web"])
async def handle_logout(request: Request):
    """Clears the user session and redirects to login."""
    request.session.clear() 
    return RedirectResponse(url="/login", status_code=303)


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



@app.post("/documents/upload", tags=["rag"])
async def upload_course_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    # Short-circuit when RAG is disabled
    if not _rag_enabled():
        return {
            "status": "disabled",
            "detail": "RAG ingestion is disabled. Set RAG_ENABLED=1 to enable.",
        }
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

    response = templates.TemplateResponse(
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
    # Save an export snapshot of the result page
    try:
        _export_html(
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
            base_slug=course_title,
        )
    except Exception:
        # Export is best-effort; do not block the response
        pass
    return response


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

    page_context = {
            "request": request,
            "course_title": course_title,
            "audience": audience,
            "tone": tone,
            "duration_minutes": duration_minutes,
            "learning_outcomes": outcomes,
            "blueprint": blueprint_content,
            "modules_output": module_assets,
        }
    # Persist export snapshot
    try:
        _export_html("course_pages.html", page_context, base_slug=course_title)
    except Exception:
        pass
    return templates.TemplateResponse(
        "course_pages.html",
        page_context,
    )


@app.get("/system-architecture", response_class=HTMLResponse, tags=["web"])
def system_architecture(request: Request):
    """Visualize the agentic system architecture."""
    return templates.TemplateResponse("system_architecture.html", {"request": request})


@app.get("/course-builder", response_class=HTMLResponse, tags=["web"])
def course_builder(request: Request):
    """Visual drag-and-drop course builder interface."""
    return templates.TemplateResponse("course_builder.html", {"request": request})


@app.get("/course-builder/{course_id}", response_class=HTMLResponse, tags=["web"])
async def course_builder_edit(
    request: Request,
    course_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Fetch Course
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Fetch Modules and Lessons
    result_mods = await db.execute(
        select(models.CourseModule).where(models.CourseModule.course_id == course_id).order_by(models.CourseModule.order_index)
    )
    modules = result_mods.scalars().all()
    
    modules_data = []
    for mod in modules:
        result_lessons = await db.execute(
            select(models.Lesson).where(models.Lesson.module_id == mod.id).order_by(models.Lesson.order_index)
        )
        lessons = result_lessons.scalars().all()
        
        items = []
        for lesson in lessons:
            # Determine type based on assets
            result_assets = await db.execute(select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id))
            assets = result_assets.scalars().all()
            
            # Default type
            item_type = 'reading'
            if any(a.asset_type == 'video' for a in assets):
                item_type = 'video'
            elif any(a.asset_type == 'quiz' for a in assets):
                item_type = 'quiz'
            
            items.append({
                "id": lesson.id, # Use integer ID
                "type": item_type,
                "title": lesson.title,
                "desc": lesson.content[:100] if lesson.content else ""
            })
            
        modules_data.append({
            "id": mod.id,
            "title": mod.title,
            "items": items
        })

    course_data = {
        "id": course.id,
        "title": course.title,
        "modules": modules_data
    }

    return templates.TemplateResponse("course_builder.html", {
        "request": request,
        "course_data": json.dumps(course_data)
    })


@app.get("/content-editor/{lesson_id}", response_class=HTMLResponse, tags=["web"])
async def content_editor(
    request: Request,
    lesson_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Fetch Lesson to verify existence and get course_id for back button
    result = await db.execute(select(models.Lesson).where(models.Lesson.id == lesson_id))
    lesson = result.scalars().first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Verify ownership via Course
    course_result = await db.execute(select(models.Course).join(models.CourseModule).where(models.CourseModule.id == lesson.module_id))
    course = course_result.scalars().first()
    if not course or course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Fetch Modules and Lessons for Sidebar
    result_mods = await db.execute(
        select(models.CourseModule).where(models.CourseModule.course_id == course.id).order_by(models.CourseModule.order_index)
    )
    modules = result_mods.scalars().all()
    
    modules_data = []
    for mod in modules:
        result_lessons = await db.execute(
            select(models.Lesson).where(models.Lesson.module_id == mod.id).order_by(models.Lesson.order_index)
        )
        lessons = result_lessons.scalars().all()
        
        items = []
        for l in lessons:
            items.append({
                "id": l.id,
                "title": l.title
            })
            
        modules_data.append({
            "id": mod.id,
            "title": mod.title,
            "lessons": items
        })

    return templates.TemplateResponse("content_editor.html", {
        "request": request,
        "lesson_id": lesson_id,
        "course_id": course.id,
        "course_title": course.title,
        "modules": modules_data
    })


@app.get("/edit-course-content/{course_id}", tags=["web"])
async def edit_course_content(
    course_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Fetch Course
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
        
    # Find first lesson
    # We need to join modules to find the first lesson of the first module
    result_lesson = await db.execute(
        select(models.Lesson)
        .join(models.CourseModule)
        .where(models.CourseModule.course_id == course_id)
        .order_by(models.CourseModule.order_index, models.Lesson.order_index)
        .limit(1)
    )
    first_lesson = result_lesson.scalars().first()
    
    if first_lesson:
        return RedirectResponse(url=f"/content-editor/{first_lesson.id}")
    else:
        # If no lessons, go to builder
        return RedirectResponse(url=f"/course-builder/{course_id}")


@app.get("/videos/{filename}", tags=["web"])
def get_video(filename: str) -> FileResponse:
    """Serve generated video files."""
    # Securely serve the file from static/videos
    video_dir = Path(__file__).resolve().parent / "static" / "videos"
    file_path = video_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
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
        assets_for_module["video_script"] = video_result
        
        # Format video URLs for the template
        video_path = video_result.get("video_file")
        if video_path:
            path_obj = Path(video_path)
            assets_for_module["video_script"]["video_url"] = f"/videos/{path_obj.name}"
            cap_file = video_result.get("captions_file")
            chap_file = video_result.get("chapters_file")
            if cap_file:
                assets_for_module["video_script"]["captions_url"] = f"/videos/{Path(cap_file).name}"
            if chap_file:
                assets_for_module["video_script"]["chapters_url"] = f"/videos/{Path(chap_file).name}"

        # Generate reading and quiz (no video)
        for prompt_id in ("reading_material", "quiz_questions"):
            assets_for_module[prompt_id] = _generate_material_payload(
                prompt_id,
                module_payload,
                preview=False,
                create_video=False,
            )
        
        module_assets.append(
            {
                "info": module,
                "assets": assets_for_module,
            }
        )

    page_context = {
            "request": request,
            "course_title": course_title,
            "audience": audience,
            "tone": tone,
            "duration_minutes": duration_minutes,
            "learning_outcomes": outcomes,
            "blueprint": blueprint_content,
            "modules_output": module_assets,
            "voice": voice or "alloy", # Pass voice to template
        }
    try:
        _export_html("full_course.html", page_context, base_slug=course_title)
    except Exception:
        pass
    return templates.TemplateResponse(
        "full_course.html",
        page_context,
    )


@app.post("/create-full-course-agent", response_class=HTMLResponse, tags=["web"])
def create_full_course_agentic(
    request: Request,
    course_title: str = Form(...),
    learning_outcomes: str = Form(...),
    audience: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    duration_minutes: Optional[str] = Form(None),
    skip_video: bool = Form(False),
    num_modules: Optional[int] = Form(None),
    num_lessons: Optional[int] = Form(None),
    user_id: int = Depends(get_session_user),
) -> HTMLResponse:
    """Build a complete course using the agentic AgentManager pipeline."""
    outcomes = _parse_learning_outcomes(learning_outcomes)

    manager = AgentManager()
    result = manager.run(outcomes, skip_video=skip_video, num_modules=num_modules, num_lessons=num_lessons)

    course_plan = result.get("course_plan", {})
    modules = course_plan.get("modules") or []

    page_context = {
        "request": request,
        "course_title": course_plan.get("title") or course_title,
        "audience": audience,
        "tone": tone,
        "duration_minutes": duration_minutes,
        "learning_outcomes": outcomes,
        "course_plan": course_plan,
        "modules": modules,
        "lessons": result.get("lessons") or [],
        "videos": result.get("videos") or [],
        "quizzes": result.get("quizzes") or [],
        "final_validation": result.get("final_validation") or {},
    }

    return templates.TemplateResponse(
        "agentic_full_course.html",
        page_context,
    )


@app.post("/agentic-finalize-course", response_class=HTMLResponse, tags=["web"])
async def agentic_finalize_course(
    request: Request,
    course_title: str = Form(...),
    learning_outcomes: Optional[str] = Form(None),
    learning_outcomes_json: Optional[str] = Form(None),
    audience: Optional[str] = Form(None),
    tone: Optional[str] = Form(None),
    duration_minutes: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    tts_model: Optional[str] = Form(None),
    theme: Optional[str] = Form(None),
    logo_path: Optional[str] = Form(None),
    skip_video: bool = Form(False),
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user),
) -> HTMLResponse:
    """Finalize an agentic course by rendering videos from generated scripts.

    Accepts either newline-separated ``learning_outcomes`` or a JSON array
    provided in ``learning_outcomes_json``. Falls back gracefully so that
    older forms that only submit raw text continue to work.
    """

    outcomes: List[str]

    # Prefer JSON payload when present and valid
    if learning_outcomes_json:
        raw_json = (learning_outcomes_json or "").strip()
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            # If JSON fails but we still have the plain-text version, fall back
            if learning_outcomes:
                outcomes = _parse_learning_outcomes(learning_outcomes)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid learning_outcomes_json: {exc}",
                ) from exc
        else:
            if not isinstance(parsed, list):
                raise HTTPException(
                    status_code=400,
                    detail="learning_outcomes_json must be a JSON array of strings.",
                )
            outcomes = [str(item).strip() for item in parsed if str(item).strip()]
            if not outcomes:
                raise HTTPException(status_code=400, detail="At least one learning outcome is required.")
    elif learning_outcomes is not None:
        outcomes = _parse_learning_outcomes(learning_outcomes)
    else:
        raise HTTPException(status_code=400, detail="At least one learning outcome is required.")

    # Use agentic plan as the source of truth for the structure (modules and
    # lessons), and adapt the generated lessons/quizzes into the data
    # structures expected by full_course.html so the full course layout is
    # driven by the agentic plan.

    manager = AgentManager()
    # Run in threadpool to avoid blocking
    result = await run_in_threadpool(manager.run, outcomes, skip_video=skip_video)
    
    # Save to DB
    await save_course_to_db(db, result, outcomes, user_id)

    course_plan = result.get("course_plan", {}) or {}
    modules = course_plan.get("modules") or []
    lessons = result.get("lessons") or []
    scripts = result.get("scripts") or []
    videos = result.get("videos") or []
    quizzes = result.get("quizzes") or []

    # Build modules_output from agentic plan. We aggregate all lessons for a
    # module into a single video/reading/quiz block so the UI matches the
    # legacy full_course.html template but the content comes from AgentManager.
    modules_output: List[Dict[str, Any]] = []
    lesson_index = 0

    for module_idx, module in enumerate(modules, start=1):
        module_title = module.get("title") if isinstance(module, dict) else getattr(module, "title", None)
        module_lessons_names = module.get("lessons") if isinstance(module, dict) else getattr(module, "lessons", [])
        module_lessons_names = module_lessons_names or []

        # Slice lessons/quizzes corresponding to this module based on the
        # order AgentManager generates them (module-by-module, lesson-by-lesson).
        count = len(module_lessons_names)
        module_lessons = lessons[lesson_index : lesson_index + count] if count else []
        module_scripts = scripts[lesson_index : lesson_index + count] if count else []
        module_videos = videos[lesson_index : lesson_index + count] if count else []
        module_quizzes = quizzes[lesson_index : lesson_index + count] if count else []
        lesson_index += count

        # Build a summary from the first lesson, if available.
        primary_lesson: Dict[str, Any] = module_lessons[0] if module_lessons else {}
        title = primary_lesson.get("title") or module_title or f"Module {module_idx}"
        summary = primary_lesson.get("summary") or ""

        # VIDEO SCRIPT: Use the generated scripts and videos
        narration_blocks: List[Dict[str, Any]] = []
        
        # Use the first script/video for the module-level video display (limitation of current template)
        # But we aggregate narration from all scripts to show full content.
        for script in module_scripts:
            # script is a dict matching VideoScript schema: {"lesson": str, "scenes": [...]}
            lesson_title = script.get("lesson") or "Lesson"
            scenes = script.get("scenes") or []
            for i, scene in enumerate(scenes, 1):
                narration_blocks.append(
                    {
                        "section": f"{lesson_title} - Scene {i}",
                        "summary": f"Duration: {scene.get('duration')}s",
                        "script": scene.get("text", ""),
                    }
                )

        video_content: Dict[str, Any] = {
            "hook": summary or f"Overview of {title}",
            "narration": narration_blocks,
        }
        
        # Determine video URL (use the first video generated for this module)
        video_url = None
        captions_url = None
        chapters_url = None
        
        if module_videos:
            # Just take the first one for now
            first_video = module_videos[0]
            v_path = first_video.get("video_file")
            if v_path:
                video_url = f"/videos/{Path(v_path).name}"
                
            c_path = first_video.get("captions_file")
            if c_path:
                captions_url = f"/videos/{Path(c_path).name}"
                
            ch_path = first_video.get("chapters_file")
            if ch_path:
                chapters_url = f"/videos/{Path(ch_path).name}"

        # READING: treat each lesson as a section in the reading.
        reading_sections: List[Dict[str, Any]] = []
        for lesson in module_lessons:
            reading_sections.append(
                {
                    "heading": lesson.get("title") or title,
                    "content": lesson.get("text", ""),
                    "key_points": [],
                    "outcome_alignment": [],
                }
            )

        reading_content: Dict[str, Any] = {
            "summary": summary,
            "sections": reading_sections,
        }

        # QUIZ: merge all agentic quizzes for this module into the template
        # format expected by full_course.html.
        quiz_questions: List[Dict[str, Any]] = []
        labels = ["A", "B", "C", "D", "E", "F"]
        for quiz in module_quizzes:
            q_list = quiz.get("questions") if isinstance(quiz, dict) else quiz.get("questions")
            if not q_list:
                continue
            for q in q_list:
                options = q.get("options") or []
                choices = []
                for idx_opt, opt_text in enumerate(options):
                    label = labels[idx_opt] if idx_opt < len(labels) else str(idx_opt + 1)
                    choices.append({"label": label, "text": opt_text})
                correct_index = q.get("correct_index", 0)
                try:
                    correct_label = choices[correct_index]["label"]
                except Exception:
                    correct_label = choices[0]["label"] if choices else "A"
                quiz_questions.append(
                    {
                        "stem": q.get("question", ""),
                        "choices": choices,
                        "answer": correct_label,
                        "learning_outcome": None,
                        "rationale": None,
                    }
                )

        quiz_content: Optional[Dict[str, Any]]
        if quiz_questions:
            quiz_content = {
                "questions": quiz_questions,
                "remediation": None,
            }
        else:
            quiz_content = None

        assets_for_module: Dict[str, Any] = {
            "video_script": {
                "content": video_content,
                "video_url": video_url,
                "captions_url": captions_url,
                "chapters_url": chapters_url
            },
            "reading_material": {"content": reading_content},
            "quiz_questions": {"content": quiz_content},
        }

        modules_output.append(
            {
                "info": {
                    "number": module_idx,
                    "title": module_title or title,
                    "summary": summary,
                },
                "assets": assets_for_module,
            }
        )

    page_context = {
        "request": request,
        "course_title": course_plan.get("title") or course_title,
        "audience": audience,
        "tone": tone,
        "duration_minutes": duration_minutes,
        "modules_output": modules_output,
        "voice": voice or "alloy",
    }

    return templates.TemplateResponse(
        "full_course.html",
        page_context,
    )
