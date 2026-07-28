import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from playwright.async_api import async_playwright
import jinja2
from PIL import Image, ImageDraw, ImageFont

class SlideRenderer:
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.slides_dir = templates_dir / "slides"
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.slides_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        # Add a zfill filter for the template
        self.env.filters['zfill'] = lambda x, n: str(x).zfill(n)

        self._playwright = None
        self._browser = None
        self._browser_unavailable = False

    async def start(self):
        """Start the persistent browser instance."""
        if self._browser or self._browser_unavailable:
            return
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
        except (NotImplementedError, OSError) as exc:
            self._browser_unavailable = True
            self._browser = None
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            print(f"Warning: Playwright renderer unavailable ({exc}). Using Pillow slide fallback.")

    async def stop(self):
        """Stop the persistent browser instance."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    async def render_slide(
        self, 
        slide_data: Dict[str, Any], 
        output_path: Path,
        course_title: str = "Course Title",
        current_page: int = 1,
        page_count: Optional[int] = None
    ):
        """Render a single slide to a PNG image using Playwright."""
        template = self.env.get_template("base_slide.html")
        
        # Merge slide data with context
        context = {
            **slide_data,
            "course_title": course_title,
            "current_page": current_page,
            "page_count": page_count
        }
        
        html_content = template.render(context)
        
        # Use a temporary file for the HTML content to ensure easy loading by Playwright
        tmp_html = output_path.with_suffix(".html")
        tmp_html.write_text(html_content, encoding="utf-8")

        if self._browser_unavailable:
            self._render_slide_with_pillow(slide_data, output_path, course_title, current_page, page_count)
            tmp_html.unlink(missing_ok=True)
            return output_path
        
        # Determine if we should use existing browser or start a temporary one
        should_close_browser = False
        browser = self._browser
        
        if not browser:
            # Fallback for one-off calls
            try:
                p = await async_playwright().start()
                browser = await p.chromium.launch(headless=True)
                should_close_browser = True
            except (NotImplementedError, OSError) as exc:
                self._browser_unavailable = True
                print(f"Warning: Playwright renderer unavailable ({exc}). Using Pillow slide fallback.")
                self._render_slide_with_pillow(slide_data, output_path, course_title, current_page, page_count)
                tmp_html.unlink(missing_ok=True)
                return output_path
            
        try:
            page = await browser.new_page(viewport={"width": 1920, "height": 1080})
            
            # Load the local HTML file
            await page.goto(f"file://{tmp_html.absolute()}")
            
            # Wait for any fonts or external resources if necessary
            await page.wait_for_load_state("networkidle")
            
            # Take a screenshot
            await page.screenshot(path=str(output_path), full_page=False)
            
            await page.close()
        finally:
            if should_close_browser:
                await browser.close()
                await p.stop()
            
        # Cleanup temporary HTML
        tmp_html.unlink(missing_ok=True)
        
        return output_path

    def _render_slide_with_pillow(
        self,
        slide_data: Dict[str, Any],
        output_path: Path,
        course_title: str,
        current_page: int,
        page_count: Optional[int],
    ) -> None:
        width, height = 1920, 1080
        image = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(image)

        try:
            title_font = ImageFont.truetype("arial.ttf", 64)
            heading_font = ImageFont.truetype("arial.ttf", 54)
            body_font = ImageFont.truetype("arial.ttf", 36)
            footer_font = ImageFont.truetype("arial.ttf", 26)
        except OSError:
            title_font = ImageFont.load_default()
            heading_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            footer_font = ImageFont.load_default()

        draw.rectangle((0, 0, width, 130), fill="#111827")
        draw.text((80, 42), course_title or "Course", fill="#ffffff", font=title_font)

        heading = str(slide_data.get("heading") or "Lesson")
        draw.text((90, 190), heading, fill="#1f2937", font=heading_font)

        y = 300
        content_blocks = slide_data.get("content_blocks") or []
        if content_blocks:
            for block in content_blocks[:5]:
                block_type = block.get("type") if isinstance(block, dict) else None
                if block_type == "bullets":
                    for item in block.get("items", [])[:6]:
                        for line in self._wrap_text(str(item), 78):
                            draw.text((130, y), f"- {line}", fill="#374151", font=body_font)
                            y += 52
                        y += 8
                else:
                    text = block.get("text") or block.get("content") or block.get("title") or str(block)
                    for line in self._wrap_text(str(text), 84):
                        draw.text((110, y), line, fill="#374151", font=body_font)
                        y += 52
                y += 18
                if y > 900:
                    break
        else:
            content = str(slide_data.get("content") or "")
            for line in self._wrap_text(content, 84)[:12]:
                draw.text((110, y), line, fill="#374151", font=body_font)
                y += 52

        footer = f"Slide {current_page}"
        if page_count:
            footer += f" / {page_count}"
        draw.text((90, 1010), footer, fill="#6b7280", font=footer_font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    def _wrap_text(self, text: str, max_chars: int) -> list[str]:
        words = text.replace("\n", " ").split()
        lines: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if len(candidate) > max_chars and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines or [""]

async def main_test():
    # Quick test
    renderer = SlideRenderer(Path("templates"))
    data = {
        "heading": "Introduction to AI",
        "content_blocks": [
            {"type": "bullets", "items": ["What is AI?", "Brief History", "Modern Applications"]},
            {"type": "callout", "text": "AI is transforming the world.", "style": "info"}
        ]
    }
    await renderer.render_slide(data, Path("test_slide.png"), course_title="AI Fundamentals", current_page=1, page_count=5)
    print("Slide rendered to test_slide.png")

if __name__ == "__main__":
    # Running test if executed directly
    # Need to handle event loop
    try:
        asyncio.run(main_test())
    except Exception as e:
        print(f"Error: {e}")
