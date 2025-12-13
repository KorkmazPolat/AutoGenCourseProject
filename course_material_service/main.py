from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
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
import sys
# Add parent directory to path to allow importing from routes and manager
sys.path.append(str(Path(__file__).resolve().parent.parent))

from course_material_service.rag_ingest import IngestError, ingest_pdf_into_qdrant
from course_material_service.rag_retriever import RagRetrieverError, build_context as retrieve_rag_context
from course_material_service.video_builder import VideoGenerationError, generate_video_from_script
from course_material_service.job_manager import job_manager, Job
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


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    step: str
    message: str
    eta_seconds: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


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

from course_material_service.admin import router as admin_router
app.include_router(admin_router)

from course_material_service.database import init_db, get_db
from course_material_service import models # Register models
from course_material_service.auth import verify_password, get_password_hash
from course_material_service.services import save_course_to_db

@app.on_event("startup")
async def on_startup():
    # Initialize database
    await init_db()
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

from course_material_service.dependencies import get_session_user
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
    """Serves the public landing page."""
    # Always show landing page at root, even if logged in.
    return templates.TemplateResponse(
        "landing_page.html",
        {"request": request}
    )

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

@app.get("/enrolled-courses", response_class=HTMLResponse, tags=["web"])
async def render_enrolled_courses(request: Request, user_id: int = Depends(get_session_user), db: AsyncSession = Depends(get_db)):
    """Lists courses the user is enrolled in."""
    # Join Enrollment and Course to get course details
    result = await db.execute(
        select(models.Course)
        .join(models.Enrollment, models.Enrollment.course_id == models.Course.id)
        .where(models.Enrollment.user_id == user_id)
        .order_by(models.Enrollment.enrolled_at.desc())
    )
    courses = result.scalars().all()

    return templates.TemplateResponse(
        "enrolled_courses.html",
        {
            "request": request,
            "courses": courses,
            "current_year": datetime.utcnow().year
        }
    )

@app.get("/course-builder", response_class=HTMLResponse, tags=["web"])
async def create_draft_course_and_redirect(
    request: Request,
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db)
):
    """Creates a new draft course and redirects to the builder."""
    # Create a new empty course
    new_course = models.Course(
        title="New Video Course",
        description="Draft course created via builder.",
        learning_outcomes=[],
        user_id=user_id,
        is_published=False
    )
    db.add(new_course)
    await db.commit()
    await db.refresh(new_course)
    
    return RedirectResponse(url=f"/course-builder/{new_course.id}", status_code=303)

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

@app.get("/course/{course_id}", response_class=HTMLResponse, tags=["web"])
async def render_full_course_db(
    request: Request,
    course_id: int,
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db)
):
    """Renders the full course view (video/reading/quiz) from the database."""
    # Fetch Course
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Fetch Modules
    result_mods = await db.execute(
        select(models.CourseModule).where(models.CourseModule.course_id == course_id).order_by(models.CourseModule.order_index)
    )
    modules = result_mods.scalars().all()

    modules_output = []
    for module in modules:
        # Fetch Lessons
        result_lessons = await db.execute(
            select(models.Lesson).where(models.Lesson.module_id == module.id).order_by(models.Lesson.order_index)
        )
        lessons = result_lessons.scalars().all()
        
        lesson_list = []
        for lesson in lessons:
            # Fetch Assets
            result_assets = await db.execute(
                select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id)
            )
            assets = result_assets.scalars().all()
            
            asset_map = {}
            for asset in assets:
                if asset.asset_type == "video":
                    # We might store script in asset.content or have a separate script asset
                    # For now, let's just pass the video URL if it exists
                    asset_map["video_script"] = {
                        "video_url": f"/static/videos/{Path(asset.file_path).name}" if asset.file_path else None,
                        "content": asset.content # Script content
                    }
                elif asset.asset_type == "reading_material":
                    asset_map["reading_material"] = {"content": asset.content}
                elif asset.asset_type == "quiz":
                    asset_map["quiz_questions"] = {"content": asset.content}
            
            lesson_list.append({
                "info": {
                    "number": lesson.order_index, # Use order index as number
                    "title": lesson.title,
                    "summary": lesson.content[:200] + "..." if lesson.content else "",
                    "id": lesson.id
                },
                "assets": asset_map
            })
            
        modules_output.append({
            "title": module.title,
            "lessons": lesson_list
        })

    return templates.TemplateResponse(
        "full_course.html",
        {
            "request": request,
            "course_title": course.title,
            "course_id": course.id,
            "modules_output": modules_output,
            "voice": "alloy" 
        }
    )

@app.get("/api/courses/{course_id}/modules/{module_id}/lessons/{lesson_id}/pdf", tags=["content"])
async def get_lesson_pdf(
    course_id: int,
    module_id: int,
    lesson_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Generates and returns a PDF for the specified lesson."""
    try:
        from course_material_service.pdf_utils import markdown_to_pdf
    except ImportError:
        from pdf_utils import markdown_to_pdf
    
    # Fetch Lesson
    result = await db.execute(select(models.Lesson).where(models.Lesson.id == lesson_id))
    lesson = result.scalars().first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")

    # Fetch Assets (to check for reading material overrides)
    result_assets = await db.execute(select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id))
    assets = result_assets.scalars().all()
    
    # Determine content
    content = lesson.content
    reading_asset = next((a for a in assets if a.asset_type == "reading_material"), None)
    if reading_asset and reading_asset.content:
        # If structured reading asset exists, use its text
        if isinstance(reading_asset.content, dict):
            content = reading_asset.content.get("text", content)
        elif isinstance(reading_asset.content, str):
            content = reading_asset.content

    pdf_bytes = markdown_to_pdf(content, title=lesson.title)
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={lesson.title}.pdf"}
    )

@app.get("/api/courses/{course_id}/modules/index/{module_idx}/lessons/index/{lesson_idx}/pdf", tags=["content"])
async def get_lesson_pdf_by_index(
    course_id: int,
    module_idx: int,
    lesson_idx: int,
    db: AsyncSession = Depends(get_db)
):
    """Generates PDF by looking up lesson via index (more robust than ID)."""
    try:
        from course_material_service.pdf_utils import markdown_to_pdf
    except ImportError:
        from pdf_utils import markdown_to_pdf

    # Fetch Course
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Fetch Modules
    result_mods = await db.execute(
        select(models.CourseModule)
        .where(models.CourseModule.course_id == course_id)
        .order_by(models.CourseModule.order_index)
    )
    modules = result_mods.scalars().all()
    
    if module_idx < 0 or module_idx >= len(modules):
        raise HTTPException(status_code=404, detail="Module index out of range")
        
    module = modules[module_idx]

    # Fetch Lessons
    result_lessons = await db.execute(
        select(models.Lesson)
        .where(models.Lesson.module_id == module.id)
        .order_by(models.Lesson.order_index)
    )
    lessons = result_lessons.scalars().all()

    if lesson_idx < 0 or lesson_idx >= len(lessons):
        # Fallback: if there's only one lesson and index is 0, it might be okay
        raise HTTPException(status_code=404, detail="Lesson index out of range")

    lesson = lessons[lesson_idx]

    # Fetch Assets
    result_assets = await db.execute(select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id))
    assets = result_assets.scalars().all()
    
    # Determine content
    content = lesson.content
    reading_asset = next((a for a in assets if a.asset_type == "reading_material"), None)
    if reading_asset and reading_asset.content:
        if isinstance(reading_asset.content, dict):
            content = reading_asset.content.get("text", content)
        elif isinstance(reading_asset.content, str):
            content = reading_asset.content

    pdf_bytes = markdown_to_pdf(content, title=lesson.title)
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={lesson.title}.pdf"}
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
        "telemetry": result.get("telemetry") or {},
    }
    return templates.TemplateResponse(
        "agentic_full_course.html",
        page_context,
    )


async def _run_agentic_job(job_id: str, payload: Dict[str, Any]) -> None:
    """Run the agentic pipeline in the background and update job status."""
    job_manager.start_job(job_id, step="initializing", message="Starting agentic pipeline")
    outcomes = _parse_learning_outcomes(payload.get("learning_outcomes", ""))

    def progress_cb(pct: int, step: str, message: str) -> None:
        job_manager.set_progress(job_id, pct, step=step, message=message)

    manager = AgentManager()
    try:
        result = await run_in_threadpool(
            manager.run,
            outcomes,
            payload.get("skip_video", False),
            payload.get("num_modules"),
            payload.get("num_lessons"),
            progress_cb,
        )
        course_plan = result.get("course_plan", {})
        modules = course_plan.get("modules") or []
        page_context = {
            "request": None,  # will be injected at render time
            "course_title": course_plan.get("title") or payload.get("course_title"),
            "audience": payload.get("audience"),
            "tone": payload.get("tone"),
            "duration_minutes": payload.get("duration_minutes"),
            "learning_outcomes": outcomes,
            "course_plan": course_plan,
            "modules": modules,
            "lessons": result.get("lessons") or [],
            "videos": result.get("videos") or [],
            "quizzes": result.get("quizzes") or [],
            "final_validation": result.get("final_validation") or {},
            "telemetry": result.get("telemetry") or {},
        }
        job_manager.complete_job(job_id, {"page_context": page_context})
    except Exception as exc:
        job_manager.fail_job(job_id, f"{exc}")
        logging.exception("Agentic job %s failed", job_id)


@app.post("/agentic-jobs/start", response_class=HTMLResponse, tags=["web"])
async def start_agentic_job(
    request: Request,
    background_tasks: BackgroundTasks,
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
    """Start the agentic pipeline as a background job. Returns JSON for XHR; redirects otherwise."""
    job = job_manager.create_job(step="queued", message="Queued for processing")
    payload = {
        "course_title": course_title,
        "learning_outcomes": learning_outcomes,
        "audience": audience,
        "tone": tone,
        "duration_minutes": duration_minutes,
        "skip_video": skip_video,
        "num_modules": num_modules,
        "num_lessons": num_lessons,
    }
    background_tasks.add_task(_run_agentic_job, job.job_id, payload)

    accept_header = request.headers.get("accept", "")
    x_requested_with = request.headers.get("x-requested-with", "")
    wants_json = "application/json" in accept_header or x_requested_with.lower() == "xmlhttprequest"
    if wants_json:
        return JSONResponse({"job_id": job.job_id})

    return RedirectResponse(
        url=f"/agentic-jobs/{job.job_id}/status",
        status_code=303,
    )


@app.get("/agentic-jobs/{job_id}", response_model=JobStatusResponse, tags=["web"])
async def get_agentic_job(job_id: str) -> JobStatusResponse:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        step=job.step,
        message=job.message,
        eta_seconds=job.eta_seconds,
        result=job.result,
        error=job.error,
    )


@app.get("/agentic-jobs/{job_id}/status", response_class=HTMLResponse, tags=["web"])
async def agentic_job_status_page(request: Request, job_id: str) -> HTMLResponse:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(
        "agentic_job_status.html",
        {
            "request": request,
            "job_id": job_id,
            "initial_status": job.status,
            "initial_progress": job.progress,
            "initial_step": job.step,
            "initial_message": job.message,
        },
    )


@app.get("/agentic-jobs/{job_id}/view", response_class=HTMLResponse, tags=["web"])
async def agentic_job_view(request: Request, job_id: str) -> HTMLResponse:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result:
        return RedirectResponse(url=f"/agentic-jobs/{job_id}/status", status_code=303)
    context = job.result.get("page_context") or {}
    context["request"] = request
    return templates.TemplateResponse("agentic_full_course.html", context)


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
    saved_course = await save_course_to_db(db, result, outcomes, user_id)
    print(f"DEBUG: Saved course ID: {saved_course.id}") # Debug log

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
    
    # We need to fetch the actual DB lessons to get their IDs for the PDF link
    # Since save_course_to_db commits, we can query them back or modify save_course_to_db to return them.
    # For now, let's query them back to be safe and simple.
    result_db_lessons = await db.execute(
        select(models.Lesson)
        .join(models.CourseModule)
        .where(models.CourseModule.course_id == saved_course.id)
        .order_by(models.CourseModule.order_index, models.Lesson.order_index)
    )
    db_lessons = result_db_lessons.scalars().all()
    # Create a map of lesson title -> lesson id for easy lookup
    # Note: Titles might not be unique globally, but should be unique per module usually. 
    # A safer way is to rely on the order since we save them in order.
    
    db_lesson_index = 0

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
        
        # Get corresponding DB lessons
        module_db_lessons = db_lessons[db_lesson_index : db_lesson_index + count] if count else []
        
        lesson_index += count
        db_lesson_index += count

        # Build a summary from the first lesson, if available.
        primary_lesson: Dict[str, Any] = module_lessons[0] if module_lessons else {}
        primary_db_lesson = module_db_lessons[0] if module_db_lessons else None
        
        title = primary_lesson.get("title") or module_title or f"Module {module_idx}"
        summary = primary_lesson.get("summary") or ""
        
        # Use the DB ID if available
        lesson_id = primary_db_lesson.id if primary_db_lesson else 0

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
                video_url = f"/static/videos/{Path(v_path).name}"
                
            c_path = first_video.get("captions_file")
            if c_path:
                captions_url = f"/static/videos/{Path(c_path).name}"
                
            ch_path = first_video.get("chapters_file")
            if ch_path:
                chapters_url = f"/static/videos/{Path(ch_path).name}"

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

        # QUIZ: Aggregate quizzes
        # We'll just take the first quiz for now or merge them
        quiz_content = {}
        if module_quizzes:
            # Merge questions from all quizzes in this module
            all_questions = []
            for q in module_quizzes:
                all_questions.extend(q.get("questions", []))
            quiz_content = {"questions": all_questions}

        modules_output.append(
            {
                "title": title,
                "summary": summary,
                "lessons": [
                    {
                        "info": {
                            "number": module_idx,
                            "title": title,
                            "summary": summary,
                            "id": lesson_id, # DB ID of the first lesson in module
                            "course_id": saved_course.id
                        },
                        "assets": {
                            "video_script": {
                                "content": video_content,
                                "video_url": video_url,
                                "captions_url": captions_url,
                                "chapters_url": chapters_url,
                            },
                            "reading_material": {
                                "content": {
                                    "summary": "Module Reading",
                                    "sections": reading_sections
                                }
                            },
                            "quiz_questions": {
                                "content": quiz_content
                            }
                        }
                    }
                ]
            }
        )

    return templates.TemplateResponse(
        "full_course.html",
        {
            "request": request,
            "course_title": course_title,
            "course_id": saved_course.id,
            "modules_output": modules_output,
            "voice": voice or "alloy",
            "telemetry": result.get("telemetry") or {},
        },
    )


# --- New Endpoints for Quiz, Invite, and Stats ---

class QuizSubmission(BaseModel):
    lesson_id: int
    score: float
    max_score: float

@app.post("/api/quiz/submit", tags=["student"])
async def submit_quiz_result(
    submission: QuizSubmission,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Find the quiz asset for this lesson
    # We assume one quiz per lesson for simplicity, or we attach it to the lesson
    # In our DB model, QuizAttempt links to an Asset.
    # We need to find the asset_id for the 'quiz' type of this lesson.
    result = await db.execute(
        select(models.LessonAsset)
        .where(models.LessonAsset.lesson_id == submission.lesson_id)
        .where(models.LessonAsset.asset_type == "quiz")
    )
    asset = result.scalars().first()
    
    if not asset:
        # If no specific asset found (maybe generated on fly?), we can't link it easily.
        # But wait, we saved assets in save_course_to_db.
        # If it's missing, we might need to create a placeholder or error.
        # For now, let's log warning and try to find ANY asset or just fail.
        print(f"Warning: No quiz asset found for lesson {submission.lesson_id}")
        # Fallback: Create a dummy asset if needed or just return success (but data won't be tracked perfectly)
        return {"status": "error", "message": "Quiz asset not found"}

    attempt = models.QuizAttempt(
        user_id=user_id,
        asset_id=asset.id,
        score=submission.score,
        max_score=submission.max_score
    )
    db.add(attempt)
    await db.commit()
    return {"status": "success", "score": submission.score}


class InviteRequest(BaseModel):
    email: str

@app.post("/api/courses/{course_id}/invite", tags=["instructor"])
async def invite_user_to_course(
    course_id: int,
    invite: InviteRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Verify ownership
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Only the course owner can invite users")

    # Find user to invite
    result_user = await db.execute(select(models.User).where(models.User.email == invite.email))
    target_user = result_user.scalars().first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User with this email not found")

    # Check if already enrolled
    result_enroll = await db.execute(
        select(models.Enrollment)
        .where(models.Enrollment.course_id == course_id)
        .where(models.Enrollment.user_id == target_user.id)
    )
    if result_enroll.scalars().first():
        return {"status": "already_enrolled", "message": "User is already enrolled"}

    # Enroll
    enrollment = models.Enrollment(user_id=target_user.id, course_id=course_id)
    db.add(enrollment)
    await db.commit()
    return {"status": "success", "message": f"User {target_user.full_name or target_user.email} enrolled"}


@app.get("/api/courses/{course_id}/stats", tags=["instructor"])
async def get_course_stats(
    course_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Verify ownership
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get enrollments
    result_enrollments = await db.execute(
        select(models.Enrollment).where(models.Enrollment.course_id == course_id)
    )
    enrollments = result_enrollments.scalars().all()
    
    stats = []
    for enroll in enrollments:
        # Get user details
        u_res = await db.execute(select(models.User).where(models.User.id == enroll.user_id))
        student = u_res.scalars().first()
        
        # Get quiz scores
        # We need to find all quiz assets for this course, then find attempts for this user
        # 1. Get all module IDs
        mod_res = await db.execute(select(models.CourseModule.id).where(models.CourseModule.course_id == course_id))
        mod_ids = mod_res.scalars().all()
        
        # 2. Get all lesson IDs
        les_res = await db.execute(select(models.Lesson.id).where(models.Lesson.module_id.in_(mod_ids)))
        lesson_ids = les_res.scalars().all()
        
        # 3. Get all quiz asset IDs
        asset_res = await db.execute(
            select(models.LessonAsset.id)
            .where(models.LessonAsset.lesson_id.in_(lesson_ids))
            .where(models.LessonAsset.asset_type == "quiz")
        )
        quiz_asset_ids = asset_res.scalars().all()
        
        # 4. Get attempts
        attempts_res = await db.execute(
            select(models.QuizAttempt)
            .where(models.QuizAttempt.user_id == student.id)
            .where(models.QuizAttempt.asset_id.in_(quiz_asset_ids))
        )
        attempts = attempts_res.scalars().all()
        
        # Calculate average score
        avg_score = 0
        if attempts:
            avg_score = sum(a.score / a.max_score for a in attempts) / len(attempts) * 100
            
        stats.append({
            "student_name": student.full_name or "Unknown",
            "student_email": student.email,
            "enrolled_at": enroll.enrolled_at.strftime("%Y-%m-%d"),
            "progress": f"{enroll.progress_percent}%",
            "quiz_average": f"{avg_score:.1f}%",
            "quizzes_taken": len(attempts)
        })
        
    return stats

@app.get("/course/{course_id}/dashboard", response_class=HTMLResponse, tags=["web"])
async def course_dashboard(
    request: Request,
    course_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course or course.user_id != user_id:
        raise HTTPException(status_code=404, detail="Course not found or access denied")
        
    return templates.TemplateResponse(
        "course_dashboard.html",
        {"request": request, "course": course}
    )


class GlobalAgentRequest(BaseModel):
    message: str
    context: Dict[str, Any]


class GlobalAgentToolRequest(BaseModel):
    action: str
    course_id: Optional[int] = None
    outcomes: Optional[List[str]] = None

@app.post("/api/global-agent")
async def global_agent_chat(
    payload: GlobalAgentRequest,
    request: Request,
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Handle chat messages for the global AI assistant.
    """
    try:
        # 1. System Prompt
        system_prompt = """
You are the in-product AI Assistant for the "Course Generator" web application.
Your primary goal is to help users create, improve, and understand their courses, and to help them use this platform effectively.

LANGUAGE
- Always answer in the same language as the user's last message (e.g., Turkish vs English).
- If the language is unclear, default to English.

SCOPE – WHAT YOU MAY ANSWER
You SHOULD answer questions that are clearly about one or more of these:
- Course creation and design (learning outcomes, modules, lessons, quizzes, videos, assignments).
- Using this platform (navigation, which page does what, common workflows, errors the user may see here).
- Educational theory or content structuring (e.g., Bloom's taxonomy, scaffolding, pacing) as it relates to designing courses.
- The specific course content or structure the user is working on, when it is provided in the `context.courseData` field.

OUT OF SCOPE – WHAT YOU MUST DECLINE
- General chat or entertainment (jokes, chit-chat not related to teaching or this app).
- Unrelated domains such as politics, news, sports, stock tips, medical or legal advice.
- Arbitrary programming help that is not clearly about using this Course Generator platform.
If the user asks something out of scope, POLITELY decline and gently redirect back to course creation. For example:
- "I’m here to help with your course creation and this platform. I can’t answer general questions about [topic], but I can help you turn that into a learning outcome or a course idea."

PAGE & CONTEXT AWARENESS
You receive a `context` object with:
- `url`: current page URL (e.g., "/dashboard", "/library").
- `title`: current document title.
- `courseData`: optional JSON representation of the course being edited (modules, lessons, outcomes), or null.
You also receive an optional `workflow_context` object (server-side state) that may include previous steps and a proposed outline.
Use these to adapt your answer:
- If `url` contains "/library": focus on explaining existing courses, where to find things, or which course to edit.
- If `url` contains "/course-builder": focus on module/lesson structure, what to add/remove, and how to improve structure.
- If `url` contains "/enrolled-courses": focus on learner experience and how to take or improve an existing course.
- If `courseData` is present: reference high-level elements (e.g., module titles, lesson counts, outcomes) instead of inventing structure. Never claim to see elements that are not in `courseData`.
- If `workflow_context` has previous steps or a draft outline, continue the flow instead of restarting.

INTERACTION STYLE
- Be concise and practical. Prefer short paragraphs and numbered or bulleted lists for suggestions.
- When the user asks a vague question, briefly ask 1–2 clarifying questions instead of guessing.
- When suggesting changes (e.g., improve outcomes), give 2–5 concrete examples, not huge walls of text.
- Do NOT expose raw JSON you receive from the backend unless the user explicitly asks for a technical view.

WORKFLOW SUPPORT (MULTI-STEP DESIGN)
- You can maintain a lightweight workflow across turns. If the user says “start a course on X”, you may propose outcomes first, then on confirmation propose modules/lessons, etc.
- Use the optional `workflow_state` field in your JSON to return a summary of steps taken and any draft outline so the server can persist it.
- If the user says “reset” or “start over”, set `workflow_state` to null or an empty object so the server clears stored state.

COURSE OUTLINE PAYLOAD
- When you have enough info, you MAY return a normalized outline in the optional `course_outline` field (alongside your `message`). Use this shape:
  {
    "course_title": "string",
    "outcomes": ["..."],
    "modules": [
      {
        "title": "string",
        "summary": "string",
        "lessons": [
          {
            "title": "string",
            "summary": "string",
            "type": "lesson" | "video" | "quiz" | "assignment",
            "duration_minutes": number | null
          }
        ]
      }
    ]
  }
- Keep it concise; do not invent excessive detail.
- Only include this when the user explicitly asks for a plan/outline or when continuing a design workflow.

NAVIGATION ACTIONS
You can optionally request a navigation action when it clearly helps the user.
- Use `"action": "navigate"` and set `"url"` to one of the app routes, for example:
  - "/dashboard" – main course creation / configuration page.
  - "/library" – list of created courses.
  - "/enrolled-courses" – list of courses the user is taking.
  - "/course-builder" – interactive drag-and-drop course builder (when appropriate).
- Only request navigation if the user explicitly asks for it or strongly implies it (e.g., "take me to my courses", "open the builder").
Otherwise leave `"action": null` and `"url": null` or omit them.

ACTION SET (ENHANCED)
- "navigate": redirect to a page (requires "url").
- "open_builder_with_outline": open the builder, optionally applying a provided "course_outline".
- "attach_outcomes_to_existing_course": attach/overwrite outcomes on an existing course (provide "target_course_id" and "outcomes").
- "fetch_course_summary": request a brief summary/stats for a course (provide "target_course_id").
- "quiz_quality_report": request a quick quality/stats check for quizzes in a course (provide "target_course_id").
- "content_preview": request lesson text snippets from a course (provide "target_course_id") so you can comment on coverage/gaps.
Only use these when they clearly match the user request.

OUTPUT FORMAT (VERY IMPORTANT)
 You MUST return a single JSON object with these fields:
 - "message": string – the text response to show the user. You may use simple Markdown-style formatting like **bold**, numbered lists, and bullet lists.
 - "action": string or null – one of ["navigate", "open_builder_with_outline", "attach_outcomes_to_existing_course", "fetch_course_summary", "quiz_quality_report", "content_preview"] or null.
 - "url": string or null – required only when "action" is "navigate". It must be a valid path in this application (e.g., "/library").
 - "workflow_state": object or null (optional) – include if you want the server to persist your multi-step design state; omit to leave unchanged.
 - "course_outline": object or null (optional) – include only if you are proposing a concrete outline as described above.
 - "target_course_id": number or null (optional) – required when you need to operate on a specific course (e.g., attach outcomes, fetch summary, quiz reports, content previews).
 - "outcomes": array of strings or null (optional) – use when action is "attach_outcomes_to_existing_course".

EXAMPLES
1) Navigation help
User: "Go to my library."
Assistant JSON:
{
  "message": "I’ll take you to your course library so you can see all the courses you’ve created.",
  "action": "navigate",
  "url": "/library"
}

2) Course design help
User: "My learning outcome is: Understand AI. Can you improve it?"
Assistant JSON:
{
  "message": "This outcome is a bit broad. Here are 3 clearer alternatives:\\n\\n1. **Explain the difference between rule-based systems and modern AI models.**\\n2. **Apply basic machine learning concepts (features, labels, training, evaluation) to a small dataset.**\\n3. **Compare strengths and limitations of at least two AI approaches for a real-world problem.**",
  "action": null,
  "url": null,
  "workflow_state": null
}

3) Out-of-scope question
User: "Who will win the football match tonight?"
Assistant JSON:
{
  "message": "I’m focused on helping you design and manage courses in this platform. I can’t predict sports results, but if you’d like, I can help you design a short course or lesson about football analytics.",
  "action": null,
  "url": null,
  "workflow_state": null
}

4) Outline proposal (when user asked for a plan)
User: "Design a beginner Python course"
Assistant JSON:
{
  "message": "Here’s a lean outline with outcomes and modules you can apply directly.",
  "action": null,
  "url": null,
  "workflow_state": {
    "previous_steps": ["proposed_outcomes", "draft_outline"]
  },
  "course_outline": {
    "course_title": "Beginner Python",
    "outcomes": [
      "Write simple Python scripts with variables, conditionals, and loops",
      "Work with lists and dictionaries to manage data",
      "Read from files and handle basic errors"
    ],
    "modules": [
      {
        "title": "Python Basics",
        "summary": "Setup, syntax, variables, and control flow",
        "lessons": [
          { "title": "Getting Started", "summary": "Install, run, Hello World", "type": "lesson", "duration_minutes": 15 },
          { "title": "Control Flow", "summary": "if/else, loops", "type": "lesson", "duration_minutes": 20 }
        ]
      }
    ]
  }
}

5) Quiz quality report
User: "Check quiz quality for course 42"
Assistant JSON:
{
  "message": "I’ll fetch a quick quiz quality snapshot for course 42.",
  "action": "quiz_quality_report",
  "target_course_id": 42
}

6) Content preview for a course
User: "Show me snippets from course 7"
Assistant JSON:
{
  "message": "Here are short snippets from your course so we can spot gaps.",
  "action": "content_preview",
  "target_course_id": 7
}
"""

        # 2. User Context
        workflow_context = request.session.get("global_agent_workflow", {})

        user_context_str = f"Current Page: {payload.context.get('url')}\nPage Title: {payload.context.get('title')}\n"
        if payload.context.get('courseData'):
            user_context_str += f"Current Course Data: {json.dumps(payload.context.get('courseData'))[:1000]}..." # Truncate if too long
        if workflow_context:
            user_context_str += f"\nWorkflow Context: {json.dumps(workflow_context)[:1000]}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{user_context_str}\n\nUser Message: {request.message}"}
        ]

        # 3. Call OpenAI
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        result = json.loads(response_text)

        # Persist or clear workflow state if provided
        if isinstance(result, dict) and "workflow_state" in result:
            if result["workflow_state"]:
                request.session["global_agent_workflow"] = result["workflow_state"]
            else:
                request.session.pop("global_agent_workflow", None)

        return result

    except Exception as e:
        logger.error(f"Global agent error: {e}")
        return {"message": "I'm having trouble connecting right now. Please try again."}


async def _get_course_for_user(db: AsyncSession, course_id: int, user_id: int):
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return course


@app.post("/api/global-agent/tools")
async def global_agent_tools(
    payload: GlobalAgentToolRequest,
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Lightweight tool endpoints the chatbot can call for DB-backed info.
    Supported actions:
    - course_summary: returns course title, outcomes, module/lesson counts
    - attach_outcomes: overwrites learning_outcomes for a course
    - quiz_quality: quick stats about quizzes in a course
    - content_preview: short snippets of lesson text for a course
    """
    try:
        if payload.action == "course_summary":
            if not payload.course_id:
                raise HTTPException(status_code=400, detail="course_id is required for course_summary")
            course = await _get_course_for_user(db, payload.course_id, user_id)

            result_mods = await db.execute(
                select(models.CourseModule).where(models.CourseModule.course_id == course.id)
            )
            modules = result_mods.scalars().all()
            module_count = len(modules)

            lesson_count = 0
            for module in modules:
                result_lessons = await db.execute(
                    select(models.Lesson).where(models.Lesson.module_id == module.id)
                )
                lesson_count += len(result_lessons.scalars().all())

            summary = {
                "course_id": course.id,
                "title": course.title,
                "outcomes": course.learning_outcomes or [],
                "modules": module_count,
                "lessons": lesson_count,
            }
            return {"status": "ok", "summary": summary}

        if payload.action == "attach_outcomes":
            if not payload.course_id:
                raise HTTPException(status_code=400, detail="course_id is required for attach_outcomes")
            if not payload.outcomes or not isinstance(payload.outcomes, list):
                raise HTTPException(status_code=400, detail="outcomes list is required")
            cleaned = [str(o).strip() for o in payload.outcomes if str(o).strip()]
            if not cleaned:
                raise HTTPException(status_code=400, detail="outcomes list cannot be empty")

            course = await _get_course_for_user(db, payload.course_id, user_id)
            course.learning_outcomes = cleaned
            await db.commit()
            return {"status": "ok", "updated_outcomes": cleaned}

        if payload.action == "quiz_quality":
            if not payload.course_id:
                raise HTTPException(status_code=400, detail="course_id is required for quiz_quality")
            course = await _get_course_for_user(db, payload.course_id, user_id)
            result_mods = await db.execute(select(models.CourseModule).where(models.CourseModule.course_id == course.id))
            modules = result_mods.scalars().all()
            quiz_count = 0
            total_questions = 0
            lessons_missing_quiz = 0

            for module in modules:
                result_lessons = await db.execute(select(models.Lesson).where(models.Lesson.module_id == module.id))
                lessons = result_lessons.scalars().all()
                for lesson in lessons:
                    result_assets = await db.execute(select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id))
                    assets = result_assets.scalars().all()
                    quiz_assets = [a for a in assets if a.asset_type == "quiz"]
                    if not quiz_assets:
                        lessons_missing_quiz += 1
                    for qa in quiz_assets:
                        quiz_count += 1
                        if isinstance(qa.content, dict):
                            total_questions += len(qa.content.get("questions", []))
            return {
                "status": "ok",
                "quiz_count": quiz_count,
                "total_questions": total_questions,
                "lessons_missing_quiz": lessons_missing_quiz,
            }

        if payload.action == "content_preview":
            if not payload.course_id:
                raise HTTPException(status_code=400, detail="course_id is required for content_preview")
            course = await _get_course_for_user(db, payload.course_id, user_id)
            result_mods = await db.execute(select(models.CourseModule).where(models.CourseModule.course_id == course.id))
            modules = result_mods.scalars().all()
            previews = []
            for module in modules:
                result_lessons = await db.execute(select(models.Lesson).where(models.Lesson.module_id == module.id))
                lessons = result_lessons.scalars().all()
                for lesson in lessons:
                    snippet = (lesson.content or "").strip()
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "..."
                    previews.append({
                        "lesson_id": lesson.id,
                        "lesson_title": lesson.title,
                        "module_title": module.title,
                        "snippet": snippet,
                    })
            return {"status": "ok", "previews": previews}

        raise HTTPException(status_code=400, detail="Unsupported tool action")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"global_agent_tools error: {e}")
        raise HTTPException(status_code=500, detail="Tool handler failed")
