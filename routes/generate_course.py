from __future__ import annotations

from typing import Any, Dict, List, Optional
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
from pathlib import Path

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "course_material_service" / "templates"))


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


@router.post("/update-course-design/{course_id}")
async def update_course_design(
    course_id: int,
    payload: CourseDesignRequest,
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

    # Update Title
    if payload.course_title:
        course.title = payload.course_title
    
    # We will replace all modules and lessons for simplicity in this visual builder context
    # Delete existing modules (cascade deletes lessons/assets)
    # Note: This is destructive! In a real app, we might want to diff/merge to preserve asset content.
    # For now, we assume the builder sends the FULL state.
    # However, deleting everything loses generated content (assets).
    # We should try to preserve assets if IDs match.
    
    # Strategy: 
    # 1. Fetch existing structure.
    # 2. Map existing assets by (module_index, lesson_index) or by ID if we persisted it.
    # Since we are using DB IDs in the builder, we can try to match.
    
    # Actually, simpler approach for now:
    # Just update the structure. If we delete and recreate, we lose the 'content' field of lessons and assets.
    # We need to be careful.
    
    # Let's try to update in place where possible.
    
    existing_mods_result = await db.execute(select(models.CourseModule).where(models.CourseModule.course_id == course.id))
    existing_mods = existing_mods_result.scalars().all()
    
    # Map existing modules by ID
    existing_mods_map = {str(m.id): m for m in existing_mods}
    
    # Track which IDs are kept
    kept_mod_ids = set()
    
    for mod_idx, mod_data in enumerate(payload.modules):
        mod_id = mod_data.id
        
        if mod_id in existing_mods_map:
            # Update existing module
            module = existing_mods_map[mod_id]
            module.title = mod_data.title
            module.order_index = mod_idx
            kept_mod_ids.add(mod_id)
        else:
            # Create new module
            module = models.CourseModule(
                course_id=course.id,
                title=mod_data.title,
                summary=f"Module {mod_idx + 1}",
                order_index=mod_idx
            )
            db.add(module)
            await db.flush() # Get ID
        
        # Handle Lessons for this module
        existing_lessons_result = await db.execute(select(models.Lesson).where(models.Lesson.module_id == module.id))
        existing_lessons = existing_lessons_result.scalars().all()
        existing_lessons_map = {str(l.id): l for l in existing_lessons}
        kept_lesson_ids = set()
        
        for item_idx, item_data in enumerate(mod_data.items):
            item_id = item_data.id
            
            if item_id in existing_lessons_map:
                # Update existing lesson
                lesson = existing_lessons_map[item_id]
                lesson.title = item_data.title
                lesson.order_index = item_idx
                if item_data.desc:
                    lesson.content = item_data.desc
                kept_lesson_ids.add(item_id)
            else:
                # Create new lesson
                lesson = models.Lesson(
                    module_id=module.id,
                    title=item_data.title,
                    content=item_data.desc or "Placeholder content",
                    order_index=item_idx,
                    duration_minutes=int(item_data.duration) if item_data.duration else 10
                )
                db.add(lesson)
                await db.flush()
                
                # Create Assets
                if item_data.type == "video":
                    db.add(models.LessonAsset(lesson_id=lesson.id, asset_type="video", content={"status": "pending"}))
                elif item_data.type == "quiz":
                    db.add(models.LessonAsset(lesson_id=lesson.id, asset_type="quiz", content={"questions_count": item_data.questions}))
        
        # Delete removed lessons
        for l in existing_lessons:
            if str(l.id) not in kept_lesson_ids:
                await db.delete(l)
                
    # Delete removed modules
    for m in existing_mods:
        if str(m.id) not in kept_mod_ids:
            await db.delete(m)

    await db.commit()
    return {"status": "success", "course_id": course.id, "message": "Course updated successfully"}


@router.post("/publish-course/{course_id}")
async def publish_course(
    course_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    course.is_published = not course.is_published
    await db.commit()
    
    return {"status": "success", "is_published": course.is_published}


@router.get("/lesson/{lesson_id}/details")
async def get_lesson_details(
    lesson_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Fetch Lesson
    result = await db.execute(select(models.Lesson).where(models.Lesson.id == lesson_id))
    lesson = result.scalars().first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Verify ownership via Course
    course_result = await db.execute(select(models.Course).join(models.CourseModule).where(models.CourseModule.id == lesson.module_id))
    course = course_result.scalars().first()
    if not course or course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Fetch Assets
    assets_result = await db.execute(select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id))
    assets = assets_result.scalars().all()
    
    return {
        "lesson": {
            "id": lesson.id,
            "title": lesson.title,
            "content": lesson.content,
            "duration_minutes": lesson.duration_minutes
        },
        "assets": [
            {
                "id": a.id,
                "type": a.asset_type,
                "content": a.content
            } for a in assets
        ]
    }


class UpdateLessonContentRequest(BaseModel):
    lesson_content: Optional[str] = None
    assets: Optional[List[Dict[str, Any]]] = None # List of {id: int, content: dict}


@router.post("/lesson/{lesson_id}/update-content")
async def update_lesson_content(
    lesson_id: int,
    payload: UpdateLessonContentRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Fetch Lesson
    result = await db.execute(select(models.Lesson).where(models.Lesson.id == lesson_id))
    lesson = result.scalars().first()
    if not lesson:
        raise HTTPException(status_code=404, detail="Lesson not found")
    
    # Verify ownership
    course_result = await db.execute(select(models.Course).join(models.CourseModule).where(models.CourseModule.id == lesson.module_id))
    course = course_result.scalars().first()
    if not course or course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Update Lesson Content
    if payload.lesson_content is not None:
        lesson.content = payload.lesson_content
    
    # Update Assets
    if payload.assets:
        for asset_update in payload.assets:
            asset_id = asset_update.get("id")
            new_content = asset_update.get("content")
            
            # Verify asset belongs to lesson
            asset_result = await db.execute(select(models.LessonAsset).where(models.LessonAsset.id == asset_id, models.LessonAsset.lesson_id == lesson.id))
            asset = asset_result.scalars().first()
            
            if asset:
                asset.content = new_content
    
    await db.commit()
    return {"status": "success", "message": "Content updated successfully"}


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
                lesson.content = bundle["lesson"].get("text") or bundle["lesson"].get("markdown") or bundle["lesson"].get("content")
                db.add(lesson)
                
                # Update Assets
                script_asset = next((a for a in assets if a.asset_type == "script"), None)
                if not script_asset and should_gen_video:
                     # Only save script if we are making a video, or if it already exists
                     # Actually, script is useful for the editor, but maybe we only want it if video is involved?
                     # Let's keep script for now as it doesn't trigger a tab in the course view directly.
                     # But to be safe and clean, let's only create it if we have a video asset or if it exists.
                     pass 
                
                if not script_asset:
                     script_asset = models.LessonAsset(lesson_id=lesson.id, asset_type="script")
                script_asset.content = bundle["script"]
                db.add(script_asset)
                
                quiz_asset = next((a for a in assets if a.asset_type == "quiz"), None)
                if quiz_asset:
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
    # Ensure markdown filter is available
    import markdown
    templates.env.filters['markdown'] = lambda text: markdown.markdown(text, extensions=['extra', 'codehilite'])

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
        
        module_data = {
            "id": module.id,
            "title": module.title,
            "summary": module.summary,
            "lessons": []
        }

        for lesson in lessons:
            result_assets = await db.execute(
                select(models.LessonAsset).where(models.LessonAsset.lesson_id == lesson.id)
            )
            assets = result_assets.scalars().all()

            # Map Assets
            video_script = {}
            reading_material = {}
            quiz_questions = {}

            # 1. Video
            video_script = None
            video_asset = next((a for a in assets if a.asset_type == "video"), None)
            if video_asset:
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
            else:
                video_script = None

            # 2. Reading (Lesson Content)
            # Only show reading if it's not the default placeholder AND it is a pure reading lesson (no video/quiz)
            reading_material = {}
            has_special_assets = any(a.asset_type in ["video", "quiz"] for a in assets)
            
            if not has_special_assets and lesson.content and lesson.content != "Placeholder content":
                reading_material = {
                    "content": {
                        "text": lesson.content,
                        "summary": lesson.title,
                        "sections": []
                    }
                }

            # 3. Quiz
            quiz_questions = {}
            quiz_asset = next((a for a in assets if a.asset_type == "quiz"), None)
            if quiz_asset:
                print(f"DEBUG: Found quiz asset for lesson {lesson.id}")
                q_data = quiz_asset.content
                if isinstance(q_data, str):
                    try:
                        q_data = json.loads(q_data)
                    except:
                        print(f"DEBUG: Failed to parse quiz JSON for lesson {lesson.id}")
                        q_data = {}
                
                print(f"DEBUG: Quiz data for lesson {lesson.id}: {q_data}")

                questions = []
                raw_qs = q_data.get("questions", [])
                if raw_qs:
                    labels = ["A", "B", "C", "D"]
                    for q in raw_qs:
                        choices = []
                        for idx, opt in enumerate(q.get("options", [])):
                            choices.append({
                                "label": labels[idx] if idx < 4 else str(idx),
                                "text": opt
                            })
                        
                        correct_idx = q.get("correct_index", 0)
                        correct_label = labels[correct_idx] if correct_idx < len(labels) else "A"

                        questions.append({
                            "stem": q.get("question"),
                            "choices": choices,
                            "answer": correct_label,
                            "learning_outcome": None,
                            "rationale": None
                        })

                # Always populate quiz_questions if asset exists, so tab shows up
                quiz_questions = {
                    "content": {
                        "questions": questions,
                        "remediation": q_data.get("remediation") or "No questions available."
                    }
                }

            module_data["lessons"].append({
                "info": {
                    "number": global_lesson_index,
                    "title": lesson.title,
                    "summary": f"Lesson {global_lesson_index}"
                },
                "assets": {
                    "video_script": video_script,
                    "reading_material": reading_material,
                    "quiz_questions": quiz_questions
                }
            })
            global_lesson_index += 1
        
        modules_output.append(module_data)

    return templates.TemplateResponse(
        "full_course.html",
        {
            "request": request,
            "course_title": course.title,
            "modules_output": modules_output,
            "voice": "alloy"
        }
    )
class CourseAssistRequest(BaseModel):
    message: str
    current_plan: Dict[str, Any]

@router.post("/assist-course-design")
async def assist_course_design(request: CourseAssistRequest):
    """
    AI Assistant to modify the course plan based on user natural language request.
    """
    from openai import OpenAI
    
    api_key = os.getenv("AUTOGEN_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
    client = OpenAI(api_key=api_key)
    
    system_prompt = """You are an expert instructional designer assistant. 
    You help the user build a course structure. You can either answer their questions or modify the course plan.

    The Course Plan JSON structure is:
    {
      "modules": [
        {
          "id": "mod-123",
          "title": "Module Title",
          "items": [
            { 
              "id": "item-456", 
              "type": "video" | "quiz" | "reading", 
              "title": "Lesson Title", 
              "desc": "Description" 
            }
          ]
        }
      ]
    }
    
    You must return a JSON object with the following structure:
    {
        "message": "Your reply to the user (explanation, confirmation, or answer)",
        "updated_plan": { ... the full course plan object ... } OR null
    }

    Rules:
    1. If the user asks a question (e.g., "What is a good topic?"), provide a helpful "message" and set "updated_plan" to null.
    2. If the user asks to change the course (e.g., "Add a module"), perform the change on the provided Current Course Plan and return the FULL updated object in "updated_plan". Also provide a brief confirmation in "message".
    3. Maintain existing IDs for existing items.
    4. If adding new modules or items, generate a unique string ID (e.g., "mod-new-1", "item-new-1").
    5. Default new item type to "video" unless specified otherwise.
    6. Default new item description to a brief summary of the title.
    """
    
    user_prompt = f"""
    Current Course Plan:
    {json.dumps(request.current_plan, indent=2)}
    
    User Request: "{request.message}"
    
    Return the JSON response:
    """
    
    try:
        completion = await run_in_threadpool(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        content = completion.choices[0].message.content
        response_data = json.loads(content)
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.delete("/course/{course_id}")
async def delete_course(
    course_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_session_user)
):
    # Fetch Course
    result = await db.execute(
        select(models.Course).where(models.Course.id == course_id)
    )
    course = result.scalars().first()
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
        
    if course.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Delete the course (cascade should handle related modules/lessons/assets if configured, 
    # otherwise we might need to delete manually. Assuming cascade is set up or we rely on SQLAlchemy)
    # If cascade is not set up in models, we should delete children first.
    # Let's assume standard cascade or manual deletion.
    # For safety, let's delete explicitly if we are unsure, but usually models have cascade.
    # Let's try simple delete first.
    
    await db.delete(course)
    await db.commit()
    
    return {"status": "success", "message": "Course deleted"}
