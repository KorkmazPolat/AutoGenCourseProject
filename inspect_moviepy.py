from moviepy import AudioFileClip, ImageClip, ColorClip
import inspect

print("MoviePy Version Check")
try:
    import moviepy
    print(f"Version: {moviepy.__version__}")
except:
    print("Could not get version")

print("\nChecking AudioFileClip attributes for 'fade':")
for attr in dir(AudioFileClip):
    if "fade" in attr.lower():
        print(f"  {attr}")

print("\nChecking ImageClip attributes for 'fade':")
for attr in dir(ImageClip):
    if "fade" in attr.lower():
        print(f"  {attr}")

print("\nChecking ColorClip (VideoClip) attributes for 'fade':")
for attr in dir(ColorClip):
    if "fade" in attr.lower():
        print(f"  {attr}")
