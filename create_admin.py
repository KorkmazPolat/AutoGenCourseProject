import asyncio
import sys
from sqlalchemy.future import select
from course_material_service.database import get_db, init_db
from course_material_service.models import User

async def promote_user(email: str):
    # We need to manually handle the session context since we are not in a request
    from course_material_service.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        
        if not user:
            print(f"Error: User with email '{email}' not found.")
            return
        
        if user.is_admin:
            print(f"User '{email}' is already an admin.")
            return
            
        user.is_admin = True
        await db.commit()
        print(f"Success: User '{email}' has been promoted to Admin.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_admin.py <email>")
        sys.exit(1)
        
    email = sys.argv[1]
    asyncio.run(promote_user(email))
