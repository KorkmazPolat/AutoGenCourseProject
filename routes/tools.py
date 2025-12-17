from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import json
from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from course_material_service.dependencies import get_session_user
from course_material_service.database import get_db
from course_material_service import models
from course_material_service.slide_engine.service import SlideGeneratorService

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "course_material_service" / "templates"))

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API Key not set")
    return OpenAI(api_key=api_key)

@router.get("/tools/quiz", response_class=HTMLResponse)
async def tool_quiz_page(request: Request, user_id: int = Depends(get_session_user)):
    return templates.TemplateResponse("tools/quiz_generator.html", {"request": request})

@router.post("/tools/generate/quiz")
async def generate_quiz(
    topic: str = Form(...),
    difficulty: str = Form("intermediate"),
    question_type: str = Form("multiple_choice"),
    count: int = Form(10),
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db)
):
    client = get_openai_client()
    
    # Customize prompt based on question type
    type_instructions = ""
    if question_type == "true_false":
        type_instructions = "All questions MUST be True/False questions. The 'options' array must contain strictly ['True', 'False']."
    elif question_type == "multiple_choice":
        type_instructions = "All questions MUST be Multiple Choice with 4 options."
    elif question_type == "short_answer":
        type_instructions = "All questions MUST be Short Answer questions. The 'options' array must be empty []."
    elif question_type == "mixed":
        type_instructions = "Generate a mix of True/False, Multiple Choice, and Short Answer questions."

    prompt = f"""
    Create a {count}-question {difficulty} level quiz about: {topic}.
    
    INSTRUCTIONS:
    {type_instructions}
    
    Return JSON format:
    {{
        "title": "Creative Quiz Title",
        "description": "Short description of what this quiz covers.",
        "questions": [
            {{
                "question": "Question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Correct Option Text or Answer",
                "explanation": "Why this is correct"
            }}
        ]
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a professional quiz generator. Output valid JSON."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = json.loads(completion.choices[0].message.content)
        
        # Save to DB
        # 1. Create Course (Type=Quiz)
        course = models.Course(
            title=content['title'],
            description=content['description'],
            user_id=user_id,
            learning_outcomes=[],
            course_type="quiz",
            is_published=True
        )
        db.add(course)
        await db.flush()
        
        # 2. Module & Lesson
        module = models.CourseModule(course_id=course.id, title="Quiz Module", order_index=1)
        db.add(module)
        await db.flush()
        
        lesson = models.Lesson(module_id=module.id, title="Quiz Assessment", content="Take this quiz to test your knowledge.", order_index=1)
        db.add(lesson)
        await db.flush()
        
        # 3. Asset
        asset = models.LessonAsset(
            lesson_id=lesson.id,
            asset_type="quiz",
            content=content['questions']
        )
        db.add(asset)
        await db.commit()
        
        return JSONResponse({"status": "success", "course_id": course.id, "redirect_url": "/library", "data": content})
        
    except Exception as e:
        print(f"Quiz Gen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/readings", response_class=HTMLResponse)
async def tool_reading_page(request: Request, user_id: int = Depends(get_session_user)):
    return templates.TemplateResponse("tools/reading_generator.html", {"request": request})

@router.post("/tools/generate/reading")
async def generate_reading(
    topic: str = Form(...),
    tone: str = Form("conversational"),
    length: str = Form("medium"),
    audience: str = Form("general audience"),
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db)
):
    client = get_openai_client()
    length_prompt = "around 400 words" if length == "short" else "around 1000 words" if length == "medium" else "over 2000 words"
    
    prompt = f"""
    Write a highly engaging, visually structured, and comprehensive reading article about: {topic}.
    Target Audience: {audience}.
    Tone: {tone}.
    Length: {length_prompt}.
    
    You MUST include the following elements to make the reading "awesome" and high-quality:
    1. **Clear Hierarchy**: Use H2 (##) for main sections and H3 (###) for subsections.
    2. **Rich Formatting**: Use **bold** for key concepts and *italics* for emphasis.
    3. **Code Snippets**: If the topic allows (even remotely), include at least one relevant code block (python, sql, javascript, etc) or a structured data example.
    4. **Data Table**: Include at least one Markdown table comparing concepts, showing stats, or listing pros/cons.
    5. **Key Takeaways/Callouts**: Use blockquotes (>) to highlight "Key Takeaways" or "Fun Facts".
    6. **Lists**: Use bullet points and numbered lists frequently to break up text.

    Return JSON format:
    {{
        "title": "Catchy & Professional Title",
        "description": "Compelling summary (2-3 sentences).",
        "content_markdown": "# Title\\n\\n## Introduction\\n\\nEngaging intro...\\n\\n> **Key Takeaway:** Summary...\\n\\n## Main Concept\\n\\nText...\\n\\n### Comparison\\n\\n| Feature | Value |\\n|---|---|\\n...\\n\\n## Implementation\\n\\n```python\\nprint('Hello')\\n```\\n"
    }}
    IMPORTANT: The 'content_markdown' MUST be rich and structured. Do not output walls of text.
    """
    
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an expert article writer. Output valid JSON."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = json.loads(completion.choices[0].message.content)
        
        # Save to DB
        course = models.Course(
            title=content['title'],
            description=content['description'],
            user_id=user_id,
            learning_outcomes=[],
            course_type="reading",
            is_published=True
        )
        db.add(course)
        await db.flush()
        
        module = models.CourseModule(course_id=course.id, title="Reading Material", order_index=1)
        db.add(module)
        await db.flush()
        
        lesson = models.Lesson(module_id=module.id, title=content['title'], content=content['content_markdown'], order_index=1)
        db.add(lesson)
        await db.commit()
        
        return JSONResponse({"status": "success", "course_id": course.id, "redirect_url": "/library", "data": content})
    except Exception as e:
        print(f"Reading Gen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools/slides", response_class=HTMLResponse)
async def tool_slides_page(request: Request, user_id: int = Depends(get_session_user)):
    return templates.TemplateResponse("tools/slide_generator.html", {"request": request})

@router.post("/tools/generate/slides")
async def generate_slides(
    topic: str = Form(...),
    style: str = Form("modern"),
    slide_count: int = Form(10),
    audience: str = Form("general audience"),
    user_id: int = Depends(get_session_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        service = SlideGeneratorService()
        content = service.generate_slides(topic, audience, slide_count, style)
        
        # Save to DB
        course = models.Course(
            title=content['title'],
            description=content['description'],
            user_id=user_id,
            learning_outcomes=[],
            course_type="slides",
            is_published=True
        )
        db.add(course)
        await db.flush()
        
        module = models.CourseModule(course_id=course.id, title="Presentation", order_index=1)
        db.add(module)
        await db.flush()
        
        # Save slides as an asset, but also put main content in lesson for fallback
        lesson_content = f"# {content['title']}\n\n"
        for slide in content.get('slides', []):
            lesson_content += f"## {slide.get('title')}\n{slide.get('content')}\n\n*Note: {slide.get('notes')}*\n\n---\n\n"

        lesson = models.Lesson(module_id=module.id, title="Slide Deck", content=lesson_content, order_index=1)
        db.add(lesson)
        await db.flush()
        
        # Save structured data as asset
        asset = models.LessonAsset(
            lesson_id=lesson.id,
            asset_type="script", # reusing script type for slides
            content=content
        )
        db.add(asset)
        await db.commit()
        
        return JSONResponse({"status": "success", "course_id": course.id, "redirect_url": "/library", "data": content})
    except Exception as e:
        print(f"Slide Gen Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


