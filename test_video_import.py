import sys
from pathlib import Path
# Add current dir to path
sys.path.append(str(Path.cwd()))

try:
    from course_material_service.video_builder import generate_video_from_script
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
