import os
from dotenv import load_dotenv
from pathlib import Path

# Load from the same place main.py does
load_dotenv(dotenv_path=Path("course_material_service/.env"), override=True)

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("DATABASE_URL not set in environment or .env file.")
else:
    # Mask password for safety
    if "@" in db_url:
        prefix, rest = db_url.split("@", 1)
        if "://" in prefix:
            proto, auth = prefix.split("://", 1)
            if ":" in auth:
                user, pw = auth.split(":", 1)
                masked_url = f"{proto}://{user}:****@{rest}"
                print(f"DATABASE_URL: {masked_url}")
            else:
                print(f"DATABASE_URL: {proto}://{auth}:****@{rest}")
        else:
            print(f"DATABASE_URL: ****@{rest}")
    else:
        print(f"DATABASE_URL: {db_url}")
