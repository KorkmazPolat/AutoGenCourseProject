# How to Connect to the Database

The project uses **SQLite**, which stores the entire database in a single file on your disk.

## 1. Database File Location
The database file is located in the root directory of your project:
```
/Users/polatkorkmaz/Documents/GitHub/AutoGenCourseProject/course_materials.db
```
*(Note: This file is created automatically when you run the application for the first time.)*

## 2. Connecting with a GUI Tool (Recommended)
You can use any SQLite-compatible database viewer to inspect the data.

### Option A: DB Browser for SQLite (Free, Open Source)
1.  Download and install [DB Browser for SQLite](https://sqlitebrowser.org/).
2.  Open the application.
3.  Click **"Open Database"**.
4.  Navigate to your project folder and select `course_materials.db`.
5.  Go to the **"Browse Data"** tab to view tables like `courses`, `modules`, `lessons`, etc.

### Option B: DBeaver (Universal Database Tool)
1.  Download and install [DBeaver](https://dbeaver.io/).
2.  Click **"New Database Connection"**.
3.  Select **SQLite**.
4.  In the "Path" field, browse and select `course_materials.db`.
5.  Click **"Finish"**.

/Users/polatkorkmaz/Documents/GitHub/AutoGenCourseProject/course_material_service/database.py

### Option C: VS Code Extensions
If you are using VS Code, you can install an extension like **"SQLite Viewer"**.
1.  Install the extension.
2.  Click on the `course_materials.db` file in your file explorer.
3.  It will open a view where you can query tables directly in the editor.

## 3. Connecting via Command Line
You can use the built-in `sqlite3` command on macOS.

1.  Open your terminal.
2.  Navigate to the project root:
    ```bash
    cd /Users/polatkorkmaz/Documents/GitHub/AutoGenCourseProject
    ```
3.  Open the database:
    ```bash
    sqlite3 course_materials.db
    ```
4.  Run SQL commands:
    ```sql
    .tables                  -- List all tables
    SELECT * FROM courses;   -- View all courses
    .quit                    -- Exit
    ```

## 4. Connecting in Python (Programmatically)
The application already connects using **SQLAlchemy** in `course_material_service/database.py`.

If you want to write a separate script to access the data:

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

DATABASE_URL = "sqlite+aiosqlite:///./course_materials.db"

async def main():
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession)

    async with async_session() as session:
        result = await session.execute(text("SELECT * FROM courses"))
        for row in result:
            print(row)

if __name__ == "__main__":
    asyncio.run(main())
```
