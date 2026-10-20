"""Browser tools: the agent's hands and eyes.

Wraps Playwright so the rest of Argus can open pages, read them as
accessibility snapshots, click, and type, without knowing Playwright details.
"""

from playwright.sync_api import sync_playwright

ACTION_TIMEOUT_MS = 5_000  # Fail fast: don't wait 30s (Playwright's default) for a missing element


class BrowserSession:
    """One browser with one open tab."""

    def __init__(self, headless: bool = False, slow_mo_ms: int = 500) -> None:
        # headless=False shows the browser window so you can watch the agent.
        # slow_mo adds a pause between actions so humans can follow along.
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless, slow_mo=slow_mo_ms)
        self.page = self._browser.new_page()

    # ── Eyes ─────────────────────────────────────────────────────────
    def snapshot(self) -> str:
        """Return the page as an accessibility snapshot (what the agent 'sees')."""
        return self.page.locator("body").aria_snapshot()

    # ── Hands ────────────────────────────────────────────────────────
    def goto(self, url: str) -> None:
        self.page.goto(url)

    def click(self, role: str, name: str) -> None:
        self.page.get_by_role(role, name=name, exact=True).click(timeout=ACTION_TIMEOUT_MS)

    def type_text(self, role: str, name: str, text: str) -> None:
        self.page.get_by_role(role, name=name, exact=True).fill(text, timeout=ACTION_TIMEOUT_MS)

    # ── Cleanup ──────────────────────────────────────────────────────
    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()

    # These two methods make the class usable with `with`, which
    # guarantees close() runs even if an error happens mid-way.
    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()