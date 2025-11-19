try:
    import moviepy.audio.fx.all as afx
    print("Imported afx")
    print(dir(afx))
except ImportError as e:
    print(f"Failed to import afx: {e}")

try:
    import moviepy.video.fx.all as vfx
    print("Imported vfx")
    print(dir(vfx))
except ImportError as e:
    print(f"Failed to import vfx: {e}")
