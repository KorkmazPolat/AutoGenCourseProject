from __future__ import annotations

from typing import Any, Dict, List
import json
import os

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from manager.agent_manager import AgentManager
from course_material_service.database import get_db
from course_material_service import models

router = APIRouter()
templates = Jinja2Templates(directory="course_material_service/templates")


from course_material_service.dependencies import get_session_user

class GenerateCourseRequest(BaseModel):
    learning_outcomes: List[str]
    skip_video: bool = False
    num_modules: int | None = None
    num_lessons: int | None = None


from course_material_service.services import save_course_to_db


@router.post("/generate-course")
async def generate_course(
    payload: GenerateCourseRequest, 
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
) -> Dict[str, Any]:
    manager = AgentManager()
    # Run the heavy blocking task in a threadpool
    result = await run_in_threadpool(
        manager.run, 
        payload.learning_outcomes, 
        skip_video=payload.skip_video,
        num_modules=payload.num_modules,
        num_lessons=payload.num_lessons
    )
    
    # Save to DB
    await save_course_to_db(db, result, payload.learning_outcomes, user_id)
    
    return result


# --- New Design Saving Logic ---

class DesignItem(BaseModel):
    id: str
    type: str
    title: str
    desc: str | None = None
    duration: str | int | None = None
    questions: str | int | None = None

class DesignModule(BaseModel):
    id: str
    title: str
    items: List[DesignItem]

class CourseDesignRequest(BaseModel):
    modules: List[DesignModule]
    course_title: str = "My Designed Course"

@router.post("/save-course-design")
async def save_course_design(
    payload: CourseDesignRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Create Course
    course = models.Course(
        title=payload.course_title,
        description="Manually designed course structure.",
        learning_outcomes=["Manual Design"],
        user_id=user_id
    )
    db.add(course)
    await db.flush()

    # Create Modules and Lessons
    for mod_idx, mod_data in enumerate(payload.modules):
        module = models.CourseModule(
            course_id=course.id,
            title=mod_data.title,
            summary=f"Module {mod_idx + 1}",
            order_index=mod_idx
        )
        db.add(module)
        await db.flush()

        for item_idx, item_data in enumerate(mod_data.items):
            # Create Lesson
            lesson = models.Lesson(
                module_id=module.id,
                title=item_data.title,
                content=item_data.desc or "Placeholder content",
                order_index=item_idx,
                duration_minutes=int(item_data.duration) if item_data.duration else 10
            )
            db.add(lesson)
            await db.flush()

            # Create Placeholder Asset based on type
            if item_data.type == "video":
                db.add(models.LessonAsset(
                    lesson_id=lesson.id,
                    asset_type="video",
                    content={"status": "pending_generation", "type": "video"}
                ))
            elif item_data.type == "quiz":
                db.add(models.LessonAsset(
                    lesson_id=lesson.id,
                    asset_type="quiz",
                    content={"status": "pending_generation", "questions_count": item_data.questions}
                ))
            elif item_data.type == "assignment":
                db.add(models.LessonAsset(
                    lesson_id=lesson.id,
                    asset_type="assignment",
                    content={"status": "pending_generation"}
                ))

    await db.commit()
    return {"status": "success", "course_id": course.id, "message": "Course structure saved successfully"}


@router.post("/generate-course-content/{course_id}")
async def generate_course_content(
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    async def generate_stream():
        # Fetch Course
        result = await db.execute(
            select(models.Course).where(models.Course.id == course_id)
        )
        course = result.scalars().first()
        if not course:
            yield f"data: {json.dumps({'error': 'Course not found'})}\\n\\n"
            return

        yield f"data: {json.dumps({'message': f'Starting generation for course: {course.title}', 'progress': 0})}\\n\\n"

        # Fetch Modules
        result_mods = await db.execute(
            select(models.CourseModule).where(models.CourseModule.course_id == course_id).order_by(models.CourseModule.order_index)
        )
        modules = result_mods.scalars().all()
        
        manager = AgentManager()
        
        # Count total lessons for progress calculation
        total_lessons = 0
        for m in modules:
            # We need to query lessons to count them
            r = await db.execute(select(func.count(models.Lesson.id)).where(models.Lesson.module_id == m.id))
            total_lessons += r.scalar()

        processed_count = 0

        for module in modules:
            result_lessons = await db.execute(
                select(models.Lesson).where(models.Lesson.module_id == module.id).order_by(models.Lesson.order_index)
            )
            lessons = result_lessons.scalars().all()
            
            for lesson in lessons:
                yield f"data: {json.dumps({'message': f'Generating content for lesson: {lesson.title}...'})}\\n\\n"
                
                # Check assets
                result_assets = await db.execute(
                    select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id)
                )
                assets = result_assets.scalars().all()
                should_gen_video = any(a.asset_type == "video" for a in assets)
                
                # Run Agent Bundle
                bundle = await run_in_threadpool(
                    manager.generate_lesson_bundle,
                    module_title=module.title,
                    lesson_title=lesson.title,
                    lesson_desc=lesson.content if lesson.content != "Placeholder content" else f"Lesson about {lesson.title}",
                    skip_video=not should_gen_video
                )
                
                # Update Lesson
                lesson.content = bundle["lesson"].get("markdown") or bundle["lesson"].get("content")
                db.add(lesson)
                
                # Update Assets
                script_asset = next((a for a in assets if a.asset_type == "script"), None)
                if not script_asset:
                    script_asset = models.LessonAsset(lesson_id=lesson.id, asset_type="script")
                script_asset.content = bundle["script"]
                db.add(script_asset)
                
                quiz_asset = next((a for a in assets if a.asset_type == "quiz"), None)
                if not quiz_asset:
                    quiz_asset = models.LessonAsset(lesson_id=lesson.id, asset_type="quiz")
                quiz_asset.content = bundle["quiz"]
                db.add(quiz_asset)
                
                if should_gen_video and bundle["video"]:
                    video_asset = next((a for a in assets if a.asset_type == "video"), None)
                    if video_asset:
                        video_asset.content = bundle["video"]
                        video_asset.file_path = bundle["video"].get("video_file")
                        db.add(video_asset)
                
                await db.flush()
                processed_count += 1
                progress = int((processed_count / total_lessons) * 100) if total_lessons > 0 else 100
                yield f"data: {json.dumps({'message': f'Completed {lesson.title}', 'progress': progress})}\\n\\n"

        await db.commit()
        yield f"data: {json.dumps({'message': 'All content generated successfully!', 'progress': 100, 'done': True})}\\n\\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@router.get("/course/{course_id}", response_class=HTMLResponse)
async def get_course_view(
    request: Request,
    course_id: int,
    db: AsyncSession = Depends(get_db)
):
    # Fetch Course
    result = await db.execute(
        select(models.Course).where(models.Course.id == course_id)
    )
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    # Fetch Modules and Lessons with Assets
    # We'll fetch all and structure them in python
    result_mods = await db.execute(
        select(models.CourseModule).where(models.CourseModule.course_id == course_id).order_by(models.CourseModule.order_index)
    )
    modules = result_mods.scalars().all()

    modules_output = []
    global_lesson_index = 1

    for module in modules:
        result_lessons = await db.execute(
            select(models.Lesson).where(models.Lesson.module_id == module.id).order_by(models.Lesson.order_index)
        )
        lessons = result_lessons.scalars().all()

        for lesson in lessons:
            # Treat each lesson as a "unit" in the UI for now to fit the template
            result_assets = await db.execute(
                select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id)
            )
            assets = result_assets.scalars().all()

            # Map Assets
            video_script = {}
            reading_material = {}
            quiz_questions = {}

            # 1. Video
            video_asset = next((a for a in assets if a.asset_type == "video"), None)
            if video_asset and video_asset.content:
                vid_data = video_asset.content
                if isinstance(vid_data, str):
                    try:
                        vid_data = json.loads(vid_data)
                    except:
                        vid_data = {}
                
                # Script asset for narration text
                script_asset = next((a for a in assets if a.asset_type == "script"), None)
                narration = []
                if script_asset and script_asset.content:
                    s_data = script_asset.content
                    if isinstance(s_data, str):
                        try:
                            s_data = json.loads(s_data)
                        except:
                            s_data = {}
                    # Extract scenes
                    scenes = s_data.get("scenes", [])
                    for i, scene in enumerate(scenes, 1):
                        narration.append({
                            "section": f"Scene {i}",
                            "summary": f"Duration: {scene.get('duration')}s",
                            "script": scene.get("text", "")
                        })

                video_url = None
                captions_url = None
                chapters_url = None

                if video_asset.file_path:
                    filename = os.path.basename(video_asset.file_path)
                    video_url = f"/static/videos/{filename}"
                    
                    # Check for sidecars (captions/chapters)
                    # We assume they are in the same directory as the video file
                    base_path = os.path.splitext(video_asset.file_path)[0]
                    if os.path.exists(base_path + ".vtt"):
                        captions_url = f"/static/videos/{os.path.splitext(filename)[0]}.vtt"
                    if os.path.exists(base_path + ".chapters.vtt"):
                        chapters_url = f"/static/videos/{os.path.splitext(filename)[0]}.chapters.vtt"

                video_script = {
                    "content": {
                        "hook": f"Video for {lesson.title}",
                        "narration": narration,
                        "recap": ""
                    },
                    "video_url": video_url,
                    "captions_url": captions_url,
                    "chapters_url": chapters_url
                }

            # 2. Reading (Lesson Content)
            # We use the lesson.content (markdown) as the reading material
            reading_material = {
                "content": {
                    "summary": lesson.title,
                    "sections": [
                        {
                            "heading": "Lesson Content",
                            "content": lesson.content,
                            "key_points": [],
                            "outcome_alignment": []
                        }
                    ]
                }
            }

            # 3. Quiz
            quiz_asset = next((a for a in assets if a.asset_type == "quiz"), None)
            if quiz_asset and quiz_asset.content:
                q_data = quiz_asset.content
                if isinstance(q_data, str):
                    try:
                        q_data = json.loads(q_data)
                    except:
                        q_data = {}
                
                questions = []
                raw_qs = q_data.get("questions", [])
                labels = ["A", "B", "C", "D"]
                for q in raw_qs:
                    choices = []
                    for idx, opt in enumerate(q.get("options", [])):
                        choices.append({
                            "label": labels[idx] if idx < 4 else str(idx),
                            "text": opt
                        })
                    
                    # Determine correct label
                    correct_idx = q.get("correct_index", 0)
                    correct_label = labels[correct_idx] if correct_idx < len(labels) else "A"

                    questions.append({
                        "stem": q.get("question"),
                        "choices": choices,
                        "answer": correct_label,
                        "learning_outcome": None,
                        "rationale": None
                    })

                quiz_questions = {
                    "content": {
                        "questions": questions,
                        "remediation": None
                    }
                }

            modules_output.append({
                "info": {
                    "number": global_lesson_index,
                    "title": f"{module.title}: {lesson.title}",
                    "summary": f"Module: {module.title}"
                },
                "assets": {
                    "video_script": video_script,
                    "reading_material": reading_material,
                    "quiz_questions": quiz_questions
                }
            })
            global_lesson_index += 1

    return templates.TemplateResponse(
        "full_course.html",
        {
            "request": request,
            "course_title": course.title,
            "modules_output": modules_output,
            "voice": "alloy"
        }
    )
