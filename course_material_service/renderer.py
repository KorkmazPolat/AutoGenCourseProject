import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from playwright.async_api import async_playwright
import jinja2

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

    async def start(self):
        """Start the persistent browser instance."""
        if not self._browser:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)

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
        
        # Determine if we should use existing browser or start a temporary one
        should_close_browser = False
        browser = self._browser
        
        if not browser:
            # Fallback for one-off calls
            p = await async_playwright().start()
            browser = await p.chromium.launch(headless=True)
            should_close_browser = True
            
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
