
import os
import sys
import json
import markdown
from pathlib import Path

# Add project root to sys.path to allow imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from course_material_service.slide_engine.service import SlideGeneratorService

def prepare_slides_for_view(slides_raw):
    """
    Replicates the logic from routes.generate_course._prepare_slides_for_view
    to generate HTML content for the slides.
    """
    prepared = []
    if not slides_raw:
        return prepared

    for idx, slide in enumerate(slides_raw):
        title = slide.get("title") or f"Slide {idx + 1}"

        raw_content = slide.get("content")
        if raw_content is None or not str(raw_content).strip():
            content = "## No Content\n\nNo content available for this slide."
        else:
            content = str(raw_content)

        raw_notes = slide.get("notes")
        notes = str(raw_notes) if raw_notes not in (None, "") else "No notes."

        # Convert Markdown to HTML
        content_html = markdown.markdown(content, extensions=['extra', 'codehilite'])

        prepared.append({
            "title": title,
            "content": content,
            "content_html": content_html,
            "notes": notes,
            "layout": slide.get("layout", "full_content")
        })

    return prepared

def generate_html_preview(slides, output_path):
    """
    Generates a simple HTML file to preview the slides.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Generated Slides Preview</title>
        <style>
            body { font-family: sans-serif; padding: 20px; background: #f0f0f0; }
            .slide { 
                background: white; 
                padding: 40px; 
                margin-bottom: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                max-width: 800px; 
                margin-left: auto; 
                margin-right: auto;
            }
            .slide h1 { border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .notes { background: #fffde7; padding: 10px; border-left: 4px solid #fbc02d; margin-top: 20px; font-size: 0.9em; }
            .meta { color: #666; font-size: 0.8em; margin-bottom: 10px; }
            /* Markdown Styles */
            table { border-collapse: collapse; width: 100%; margin: 15px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            blockquote { border-left: 4px solid #ccc; margin: 0; padding-left: 16px; color: #555; }
            pre { background: #f4f4f4; padding: 10px; overflow-x: auto; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Generated Slides Preview</h1>
    """

    for i, slide in enumerate(slides):
        html_content += f"""
        <div class="slide">
            <div class="meta">Slide {i+1} - Layout: {slide['layout']}</div>
            <div class="content">
                {slide['content_html']}
            </div>
            <div class="notes">
                <strong>Speaker Notes:</strong> {slide['notes']}
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML preview saved to {output_path}")

def main():
    print("Initializing SlideGeneratorService...")
    service = SlideGeneratorService()

    topic = "The Future of AI in Education"
    audience = "University Students"
    slide_count = 3
    style = "Modern and Minimalist"

    print(f"Generating slides for topic: '{topic}'...")
    result = service.generate_slides(topic, audience, slide_count, style)
    
    if not result or "slides" not in result:
        print("Failed to generate slides or invalid format.")
        return

    print("Slides generated successfully.")
    
    # Save raw JSON
    output_dir = Path("example_slides")
    output_dir.mkdir(exist_ok=True)
    
    json_path = output_dir / "example_deck.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Raw JSON saved to {json_path}")

    # Process and Save HTML Preview
    prepared_slides = prepare_slides_for_view(result["slides"])
    html_path = output_dir / "example_deck.html"
    generate_html_preview(prepared_slides, html_path)

if __name__ == "__main__":
    main()
