from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from course_material_service.database import get_db
from course_material_service import models

async def get_session_user(request: Request):
    """Dependency: Checks if a user is in the session. Redirects to login if not."""
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=307, detail="Not authorized", headers={"Location": "/login"})
    return user_id

async def get_current_user(
    user_id: int = Depends(get_session_user), 
    db: AsyncSession = Depends(get_db)
) -> models.User:
    result = await db.execute(select(models.User).where(models.User.id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=307, detail="User not found", headers={"Location": "/login"})
    return user

async def get_admin_user(user: models.User = Depends(get_current_user)) -> models.User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user
