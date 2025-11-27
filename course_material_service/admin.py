from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from course_material_service.database import get_db
from course_material_service import models
from course_material_service.dependencies import get_admin_user

from pathlib import Path

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_admin_user)]
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@router.get("/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: models.User = Depends(get_admin_user)
):
    # Fetch all users
    result_users = await db.execute(select(models.User).order_by(models.User.created_at.desc()))
    users = result_users.scalars().all()

    # Fetch all courses with creator info
    result_courses = await db.execute(
        select(models.Course)
        .options(selectinload(models.Course.creator))
        .order_by(models.Course.created_at.desc())
    )
    courses = result_courses.scalars().all()

    return templates.TemplateResponse(
        "admin_dashboard.html",
        {
            "request": request,
            "users": users,
            "courses": courses,
            "current_user": admin
        }
    )

@router.post("/users/{user_id}/delete")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: models.User = Depends(get_admin_user)
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    await db.commit()
    
    return RedirectResponse(url="/admin/dashboard", status_code=303)

@router.post("/users/{user_id}/promote")
async def promote_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: models.User = Depends(get_admin_user)
):
    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_admin = True
    await db.commit()
    
    return RedirectResponse(url="/admin/dashboard", status_code=303)

@router.post("/courses/{course_id}/delete")
async def delete_course(
    course_id: int,
    db: AsyncSession = Depends(get_db),
    admin: models.User = Depends(get_admin_user)
):
    result = await db.execute(select(models.Course).where(models.Course.id == course_id))
    course = result.scalars().first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    await db.delete(course)
    await db.commit()
    
    return RedirectResponse(url="/admin/dashboard", status_code=303)

@router.post("/users/create")
async def create_user(
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    is_admin: bool = Form(False),
    db: AsyncSession = Depends(get_db),
    admin: models.User = Depends(get_admin_user)
):
    # Check if user exists
    result = await db.execute(select(models.User).where(models.User.email == email))
    existing_user = result.scalars().first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    from course_material_service.auth import get_password_hash
    
    new_user = models.User(
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
        is_admin=is_admin
    )
    db.add(new_user)
    await db.commit()
    
    return RedirectResponse(url="/admin/dashboard", status_code=303)
