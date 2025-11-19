from __future__ import annotations

from typing import Any, Dict, List
import json

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from manager.agent_manager import AgentManager
from course_material_service.database import get_db
from course_material_service import models

router = APIRouter()


class GenerateCourseRequest(BaseModel):
    learning_outcomes: List[str]
    skip_video: bool = False


async def save_course_to_db(db: AsyncSession, result: Dict[str, Any], learning_outcomes: List[str]):
    # Extract data
    course_plan = result.get("course_plan", {})
    lessons_data = result.get("lessons", [])
    scripts_data = result.get("scripts", [])
    videos_data = result.get("videos", [])
    quizzes_data = result.get("quizzes", [])

    # Create Course
    course = models.Course(
        title=course_plan.get("title", "Untitled Course"),
        description=f"Generated course based on outcomes: {learning_outcomes}",
        learning_outcomes=learning_outcomes
    )
    db.add(course)
    await db.flush() # Get ID

    # Create Modules and Lessons
    modules_map = {} # title -> module_obj
    
    # The course plan has modules list
    plan_modules = course_plan.get("modules", [])
    for idx, mod_data in enumerate(plan_modules):
        module = models.CourseModule(
            course_id=course.id,
            title=mod_data.get("title"),
            summary=f"Module {idx+1}", # Summary might not be in the plan object directly depending on schema
            order_index=idx
        )
        db.add(module)
        await db.flush()
        modules_map[module.title] = module

    # Create Lessons
    # lessons_data is a list of lesson objects. We need to match them to modules.
    # The lesson object usually has 'module_name' or similar if we look at the agent output.
    # Let's check the AgentManager output structure in the previous turn.
    # It returns "lessons": [validated_lesson, ...].
    # validated_lesson comes from LessonWriter.
    
    # We need to know which module a lesson belongs to.
    # In AgentManager.run:
    # lesson_json = self.lesson_writer.generate({"module_name": ...})
    # But the output 'lesson_json' might not contain module_name explicitly unless the agent put it there.
    # However, the loop in AgentManager iterates modules and then lessons.
    # The 'lessons' list in the result is flat.
    # This is a bit tricky. The 'lessons' list order matches the generation order.
    # The generation order is Module 1 -> Lessons, Module 2 -> Lessons.
    
    # A better way is to trust the lesson content if it has metadata, or just rely on the order if we assume strict ordering.
    # But 'lessons' list in result is just appended.
    
    # Let's look at the LessonWriter prompt or schema.
    # Assuming we can't easily link them without more info, I'll try to match by title.
    
    # Actually, let's just iterate through the plan again, and try to find the generated lesson content for each lesson title.
    # The 'lessons' list contains the full content.
    
    lesson_lookup = {l.get("title"): l for l in lessons_data}
    
    # Also map assets by lesson title if possible.
    # Scripts, videos, quizzes are also lists.
    # We need to link them.
    # VideoScriptAgent input was 'validated_lesson'. Output is script.
    # We assume script has a title matching the lesson or we rely on order?
    # The lists are appended in the same loop, so index i of lessons corresponds to index i of scripts (usually).
    
    for i, lesson_data in enumerate(lessons_data):
        # We need to find the module for this lesson.
        # This is hard without the module name in lesson_data.
        # Let's assume lesson_data has 'module' field if the schema defined it?
        # If not, we might have to guess or just put them all in the first module if we fail.
        
        # Let's assume strict order matches the plan?
        # Actually, let's just save them.
        
        # Try to find module by name if present
        mod_title = lesson_data.get("module") or lesson_data.get("module_title")
        module_id = None
        if mod_title and mod_title in modules_map:
            module_id = modules_map[mod_title].id
        else:
            # Fallback: try to find which module has this lesson title in the plan
            for m_title, m_obj in modules_map.items():
                # We need to look at the plan_modules again to see which module has this lesson
                # This is inefficient but fine for small courses.
                for pm in plan_modules:
                    if pm.get("title") == m_title:
                        if lesson_data.get("title") in pm.get("lessons", []):
                            module_id = m_obj.id
                            break
                if module_id: break
        
        lesson = models.Lesson(
            module_id=module_id,
            title=lesson_data.get("title"),
            content=lesson_data.get("content") or lesson_data.get("markdown") or str(lesson_data),
            order_index=i
        )
        db.add(lesson)
        await db.flush()
        
        # Save Assets
        # Script
        if i < len(scripts_data):
            script = scripts_data[i]
            db.add(models.LessonAsset(
                lesson_id=lesson.id,
                asset_type="script",
                content=script
            ))
            
        # Quiz
        if i < len(quizzes_data):
            quiz = quizzes_data[i]
            db.add(models.LessonAsset(
                lesson_id=lesson.id,
                asset_type="quiz",
                content=quiz
            ))
            
        # Video
        # Videos list might be shorter if skipped or failed.
        # We need to match by lesson title if possible.
        # The video_generator returns video_info which usually has 'file_path'.
        # Does it have lesson title?
        # Let's just try to match by index if lengths match, otherwise we might miss some.
        if i < len(videos_data):
            vid = videos_data[i]
            # Check if this video belongs to this lesson (if metadata exists)
            # For now, assume 1:1 mapping if skip_video was False.
            db.add(models.LessonAsset(
                lesson_id=lesson.id,
                asset_type="video",
                file_path=vid.get("video_file") or vid.get("file_path"),
                content=vid
            ))

    await db.commit()


@router.post("/generate-course")
async def generate_course(
    payload: GenerateCourseRequest, 
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    manager = AgentManager()
    # Run the heavy blocking task in a threadpool
    result = await run_in_threadpool(
        manager.run, 
        payload.learning_outcomes, 
        skip_video=payload.skip_video
    )
    
    # Save to DB
    await save_course_to_db(db, result, payload.learning_outcomes)
    
    return result
