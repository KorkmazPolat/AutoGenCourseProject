import asyncio
import sys
import os
from sqlalchemy.future import select
from course_material_service.database import AsyncSessionLocal
from course_material_service.models import User
from course_material_service.auth import get_password_hash

async def create_or_promote_admin(email: str, password: str = "Admin123!"):
    async with AsyncSessionLocal() as db:
        # Check if user exists
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()
        
        if not user:
            print(f"User '{email}' not found. Creating a new user...")
            hashed_pw = get_password_hash(password)
            user = User(
                email=email,
                hashed_password=hashed_pw,
                full_name="Admin User",
                is_admin=True
            )
            db.add(user)
            await db.commit()
            print(f"Success: Created new admin user '{email}' with password '{password}'.")
        else:
            if user.is_admin:
                print(f"User '{email}' is already an admin.")
            else:
                user.is_admin = True
                await db.commit()
                print(f"Success: Existing user '{email}' has been promoted to Admin.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_admin.py <email> [password]")
        sys.exit(1)
        
    email = sys.argv[1]
    password = sys.argv[2] if len(sys.argv) > 2 else "Admin123!"
    
    asyncio.run(create_or_promote_admin(email, password))
