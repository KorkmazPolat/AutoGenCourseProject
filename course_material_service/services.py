from typing import Any, Dict, List
from sqlalchemy.ext.asyncio import AsyncSession
from . import models

async def save_course_to_db(db: AsyncSession, result: Dict[str, Any], learning_outcomes: List[str], user_id: int):
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
        learning_outcomes=learning_outcomes,
        user_id=user_id
    )
    db.add(course)
    await db.flush() # Get ID

    # Create Modules and Lessons
    modules_map = {} # title -> module_obj
    
    # The course plan has modules list
    plan_modules = course_plan.get("modules", [])
    for idx, mod_data in enumerate(plan_modules):
        # Handle both dict and object (if pydantic)
        title = mod_data.get("title") if isinstance(mod_data, dict) else getattr(mod_data, "title", f"Module {idx+1}")
        
        module = models.CourseModule(
            course_id=course.id,
            title=title,
            summary=f"Module {idx+1}", 
            order_index=idx
        )
        db.add(module)
        await db.flush()
        modules_map[module.title] = module

    # Create Lessons
    for i, lesson_data in enumerate(lessons_data):
        # Try to find module by name if present
        mod_title = lesson_data.get("module") or lesson_data.get("module_title")
        module_id = None
        if mod_title and mod_title in modules_map:
            module_id = modules_map[mod_title].id
        else:
            # Fallback: try to find which module has this lesson title in the plan
            for m_title, m_obj in modules_map.items():
                for pm in plan_modules:
                    pm_title = pm.get("title") if isinstance(pm, dict) else getattr(pm, "title", "")
                    if pm_title == m_title:
                        pm_lessons = pm.get("lessons", []) if isinstance(pm, dict) else getattr(pm, "lessons", [])
                        if lesson_data.get("title") in pm_lessons:
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
        if i < len(videos_data):
            vid = videos_data[i]
            db.add(models.LessonAsset(
                lesson_id=lesson.id,
                asset_type="video",
                file_path=vid.get("video_file") or vid.get("file_path"),
                content=vid
            ))

    await db.commit()
    return course
