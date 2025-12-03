import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
import asyncio

from course_material_service.video_builder import generate_video_from_script

# Mock client
class MockClient:
    class Audio:
        class Speech:
            class WithStreamingResponse:
                def create(self, **kwargs):
                    class Response:
                        def __enter__(self): return self
                        def __exit__(self, *args): pass
                        def stream_to_file(self, path):
                            import wave
                            with wave.open(str(path), 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(44100)
                                wf.writeframes(b'\x00' * 44100) # 1 second of silence
                    return Response()
            with_streaming_response = WithStreamingResponse()
        speech = Speech()
    audio = Audio()

client = MockClient()

payload = {
    "presentation": [
        {"page": 1, "heading": "Test Slide", "content": "Bullet 1\nBullet 2"}
    ],
    "narration": [
        {"script": "This is a test narration."}
    ]
}

output_path = Path("test_output.mp4")

try:
    generate_video_from_script(
        video_payload=payload,
        output_path=output_path,
        client=client,
        voice="alloy",
        tts_model="gpt-4o-mini-tts"
    )
    print("Video generated successfully")
except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()
