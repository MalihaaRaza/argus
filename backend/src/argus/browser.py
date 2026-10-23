"""Browser tools: the agent's hands and eyes.

Wraps Playwright so the rest of Argus can open pages, read them as
element-ref snapshots, click, and type, without knowing Playwright details.
"""

import logging
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Self

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Locator, Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

ACTION_TIMEOUT_MS = (
    5_000  # Fail fast: don't wait 30s (Playwright's default) for a missing element
)
MAX_SNAPSHOT_CHARS = 6_000  # Keeps each snapshot to roughly 1.5k tokens

# The snapshot script lives in its own .js file so it gets JavaScript syntax
# highlighting and can be read on its own. Loaded once, when this module is imported.
_SNAPSHOT_JS = Path(__file__).with_name("snapshot.js").read_text(encoding="utf-8")
_REF_PATTERN = re.compile(r"e\d+")


RETRY_DELAYS_S = (1, 3)  # Wait before the 2nd and 3rd navigation attempts

# Chromium network errors that are often temporary (a blip, a server restarting).
# Anything else, like ERR_NAME_NOT_RESOLVED (usually a typo in the URL), fails at once.
TRANSIENT_NET_ERRORS = (
    "ERR_ABORTED",
    "ERR_TIMED_OUT",
    "ERR_CONNECTION_RESET",
    "ERR_CONNECTION_CLOSED",
    "ERR_CONNECTION_REFUSED",
    "ERR_NETWORK_CHANGED",
    "ERR_EMPTY_RESPONSE",
)


class ElementNotFoundError(Exception):
    """The model used a ref that isn't on the page (stale, mistyped, or made up)."""


class NavigationError(Exception):
    """The page couldn't be opened, even after retrying."""


def navigate_with_retry(
    page: Page,
    url: str,
    delays: Sequence[float] = RETRY_DELAYS_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Open `url`, retrying transient failures once per entry in `delays`.

    `page` and `sleep` are parameters so tests can pass fakes and check the
    backoff without actually waiting.
    """
    attempts = len(delays) + 1
    for attempt in range(1, attempts + 1):
        try:
            page.goto(url)
            return
        except PlaywrightError as error:
            reason = _first_line(error)
            if not _is_transient(error):
                raise NavigationError(f"Could not open {url}: {reason}") from error
            if attempt == attempts:
                raise NavigationError(
                    f"Could not open {url}: {reason} (after {attempts} attempts)"
                ) from error
            delay = delays[attempt - 1]
            logger.warning(
                "Navigation to %s failed (%s); retrying in %ss (attempt %d of %d)",
                url,
                reason,
                delay,
                attempt + 1,
                attempts,
            )
            sleep(delay)


def _is_transient(error: PlaywrightError) -> bool:
    if isinstance(error, PlaywrightTimeoutError):
        return True
    return any(code in str(error) for code in TRANSIENT_NET_ERRORS)


def _first_line(error: Exception) -> str:
    # Playwright messages end with a long call log; the first line is the useful part.
    lines = str(error).strip().splitlines()
    return lines[0] if lines else type(error).__name__


class BrowserSession:
    """One browser with one open tab."""

    def __init__(self, headless: bool = False, slow_mo_ms: int = 500) -> None:
        # headless=False shows the browser window so you can watch the agent.
        # slow_mo adds a pause between actions so humans can follow along.
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=headless, slow_mo=slow_mo_ms
        )
        self.page = self._browser.new_page()
        # Role, name and context of each ref in the latest snapshot, used to
        # describe actions in human terms (see describe()).
        self._refs: dict[str, dict[str, Any]] = {}

    # ── Eyes ─────────────────────────────────────────────────────────
    def snapshot(self) -> str:
        """Return a compact outline of the page (what the agent 'sees').

        Each interactive element is listed as `[eN] role "name"` and tagged in the
        page with a matching data-argus-ref attribute. Refs are reassigned on every
        call, so only the refs from the latest snapshot are valid.
        """
        result = self.page.evaluate(_SNAPSHOT_JS, MAX_SNAPSHOT_CHARS)
        self._refs = result["refs"]
        return result["text"]

    def describe(self, ref: str) -> str | None:
        """Describe a ref from the latest snapshot, e.g. 'button "Add to cart" in: Backpack'.

        Returns None if the ref wasn't in the latest snapshot.
        """
        info = self._refs.get(ref)
        if info is None:
            return None
        description = f'{info["role"]} "{info["name"]}"'
        if info["context"]:
            description += f" in: {info['context']}"
        return description

    def is_secret(self, ref: str) -> bool:
        """True if the ref is a password field in the latest snapshot."""
        return bool(self._refs.get(ref, {}).get("secret"))

    # ── Hands ────────────────────────────────────────────────────────
    def goto(self, url: str) -> None:
        """Open `url`, retrying transient errors. Raises NavigationError if it can't."""
        navigate_with_retry(self.page, url)

    def click(self, ref: str) -> None:
        self._ref_locator(ref).click(timeout=ACTION_TIMEOUT_MS)

    def type_text(self, ref: str, text: str) -> None:
        self._ref_locator(ref).fill(text, timeout=ACTION_TIMEOUT_MS)

    def _ref_locator(self, ref: str) -> Locator:
        """Find the element tagged with `ref`, or explain why it can't be found."""
        # Checking the format first also stops a model-supplied string from
        # breaking out of the CSS selector below.
        locator = self.page.locator(f'[data-argus-ref="{ref}"]')
        if not _REF_PATTERN.fullmatch(ref) or locator.count() == 0:
            raise ElementNotFoundError(
                f"No element with ref {ref!r}. Refs change after every snapshot; "
                "use a ref from the latest snapshot."
            )
        return locator

    # ── Cleanup ──────────────────────────────────────────────────────
    def close(self) -> None:
        self._browser.close()
        self._playwright.stop()

    # These two methods make the class usable with `with`, which
    # guarantees close() runs even if an error happens mid-way.
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
