import asyncio
from course_material_service.database import engine
from sqlalchemy import text

async def add_course_type_column():
    async with engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE courses ADD COLUMN course_type VARCHAR DEFAULT 'course'"))
            print("Added course_type column successfully.")
        except Exception as e:
            print(f"Column might already exist or error: {e}")

if __name__ == "__main__":
    asyncio.run(add_course_type_column())
