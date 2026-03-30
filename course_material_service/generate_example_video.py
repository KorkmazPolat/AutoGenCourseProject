
import os
import sys
from pathlib import Path

# Add the current directory to sys.path to ensure we can import video_builder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_builder import generate_video_from_script

def main():
    print("Generating example video for AutoGen Course Project...")
    
    # Define the content for the video (approx 1 minute)
    # Based on SPEECH.md Slides 1 and 2
    
    video_payload = {
        "course_title": "AutoGen Course Generator",
        "hook": "What if you could create a high-quality video course just by typing a single sentence?",
        "presentation": [
            {
                "page": 1,
                "heading": "AutoGen Course Generator",
                "content": "• Multi-Agent Systems\n• Automated Production Studio\n• Text to Video Support",
                "visual_hints": ["Clean", "Modern", "Tech"]
            },
            {
                "page": 2,
                "heading": "Why Did We Build This?",
                "content": "• Democratize Education\n• Acceleration (Weeks -> Minutes)\n• Showcase Agentic AI",
                "visual_hints": ["Comparison", "Speed", "AI Network"]
            },
            {
                "page": 3,
                "heading": "The Solution Architecture",
                "content": "",
                "content_blocks": [
                    {
                        "type": "subheading",
                        "text": "Core Principles"
                    },
                    {
                         "type": "table",
                         "headers": ["Component", "Tech", "Role"],
                         "rows": [
                             ["Orchestrator", "FastAPI", "Async Event Loop"],
                             ["Memory", "Qdrant", "Vector Retrieval"],
                             ["Agents", "AutoGen", "Reasoning"]
                         ]
                    },
                    {
                        "type": "code",
                        "language": "python",
                        "code": "async def search_memory(query: str):\n    vector = await embed(query)\n    results = qdrant.search(\n        collection=\"textbooks\",\n        query_vector=vector,\n        limit=5\n    )\n    return results"
                    },
                    {
                        "type": "key_takeaway",
                        "text": "By combining structural logic (SQL) with semantic understanding (Vectors), we achieve 99% accuracy."
                    }
                ],
                "visual_hints": ["Cloud", "Database", "Agents"]
            }
        ],
        "narration": [
            "Good afternoon. We are presenting the AutoGen Course Generator. Our project addresses a simple question: What if you could create a high-quality video course just by typing a single sentence? We have built an automated platform that uses Multi-Agent Systems to act as a production studio in the cloud.",
            "So, why did we build this? First, to Democratize Education by removing technical barriers. Second, for Acceleration, reducing production time from weeks to minutes. And finally, to showcase Agentic AI, proving that autonomous agents can coordinate and execute complex workflows without human intervention.",
            "Our solution utilizes a robust Cloud-Native architecture. As you can see, we use FastAPI for orchestration and Qdrant for semantic memory. The Python code snippet shows how we perform vector searches to find relevant textbook content. This hybrid approach ensures high factual accuracy."
        ],
        "call_to_action": "Visit our live demo to generate your own course today."
    }

    output_path = Path("generated_videos/example_presentation.mp4")
    
    # We pass client=None to force fallback to gTTS (Google Text-to-Speech)
    # assuming the user might not have OpenAI API configured or we want to use the free fallback.
    try:
        generate_video_from_script(
            video_payload=video_payload,
            output_path=output_path,
            client=None, # This triggers the gTTS fallback in video_builder.py
            voice="alloy", # Ignored by gTTS but required argument
            course_title="AutoGen Course Generator",
            theme="dark"
        )
        print(f"Success! Video generated at: {output_path.absolute()}")
    except Exception as e:
        print(f"Error generating video: {e}")

if __name__ == "__main__":
    main()
