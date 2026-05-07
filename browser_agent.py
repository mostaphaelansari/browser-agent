import asyncio
from typing import Any, Dict
from playwright.async_api import async_playwright, Browser, Page, Playwright
from loguru import logger
import yaml
from pathlib import Path

from agent_base import BaseAgent

class BrowserAgent(BaseAgent):
    """Browser Agent that uses Playwright to perform web actions."""

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
        
        headless = self.config.get("headless", True)
        logger.info(f"Initializing BrowserAgent (headless={headless})")
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        logger.info(f"BrowserAgent received action: {action}")
        
        try:
            if action == "navigate":
                url = request.get("url")
                if not url:
                    return {"error": "Missing URL parameter"}
                await self.page.goto(url)
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
        logger.info("Shutting down BrowserAgent")
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
