from fastapi import Request, HTTPException

async def get_session_user(request: Request):
    """Dependency: Checks if a user is in the session. Redirects to login if not."""
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=307, detail="Not authorized", headers={"Location": "/login"})
    return user_id
