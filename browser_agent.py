from typing import Any, Dict
from playwright.async_api import async_playwright, Browser, Page, Playwright
from loguru import logger
import yaml
from pathlib import Path
from contextlib import suppress

from agent_base import BaseAgent


class BrowserAgent(BaseAgent):
    """Browser Agent that uses Playwright to perform web actions.

    Playwright async objects are bound to the event loop they were created on.
    BedrockAgentCoreApp runs the lifespan handler on a different loop than
    each request handler, so the browser is launched lazily on first handle()
    call and torn down per-request via shutdown().
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.page: Page | None = None

    async def initialize(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        logger.info("BrowserAgent config loaded; browser launches lazily per request.")

    async def _ensure_browser(self) -> None:
        if self.page:
            return
        headless = self.config.get("headless", True)
        timeout_s = float(self.config.get("browser_timeout", 30))
        timeout_ms = int(timeout_s * 1000)
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()
        self.page.set_default_timeout(timeout_ms)
        self.page.set_default_navigation_timeout(timeout_ms)

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        logger.info(f"BrowserAgent received action: {action}")

        try:
            await self._ensure_browser()

            if action == "navigate":
                url = request.get("url")
                if not url:
                    return {"error": "Missing URL parameter"}
                await self.page.goto(url, wait_until="domcontentloaded")
                title = await self.page.title()
                return {"status": "success", "title": title, "url": self.page.url}

            elif action == "click":
                selector = request.get("selector")
                if not selector:
                    return {"error": "Missing selector parameter"}
                await self.page.click(selector)
                return {"status": "success", "message": f"Clicked {selector}"}

            elif action == "extract":
                selector = request.get("selector")
                if not selector:
                    return {"error": "Missing selector parameter"}
                text = await self.page.locator(selector).inner_text()
                return {"status": "success", "text": text}

            elif action == "screenshot":
                path = request.get("path", "screenshot.png")
                await self.page.screenshot(path=path)
                return {"status": "success", "path": path}

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Browser action '{action}' failed: {e}")
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> None:
        # Idempotent: called per-request after each invocation, and again at app shutdown.
        if self.page:
            with suppress(Exception):
                await self.page.close()
            self.page = None
        if self.browser:
            with suppress(Exception):
                await self.browser.close()
            self.browser = None
        if self.playwright:
            with suppress(Exception):
                await self.playwright.stop()
            self.playwright = None
