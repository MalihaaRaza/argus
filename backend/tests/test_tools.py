from playwright.sync_api import Error as PlaywrightError

from argus.tools import ERROR_PREFIX, TOOLS, execute_tool


class FakeBrowser:
    """Stands in for BrowserSession: records calls, optionally raises."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error

    def click(self, role: str, name: str) -> None:
        self.calls.append(("click", role, name))
        if self.error:
            raise self.error

    def type_text(self, role: str, name: str, text: str) -> None:
        self.calls.append(("type_text", role, name, text))
        if self.error:
            raise self.error


def test_tool_names() -> None:
    assert [tool["name"] for tool in TOOLS] == ["click", "type_text", "finish"]


def test_click_calls_browser() -> None:
    browser = FakeBrowser()
    result = execute_tool(browser, "click", {"role": "button", "name": "Sign in"})
    assert browser.calls == [("click", "button", "Sign in")]
    assert not result.startswith(ERROR_PREFIX)


def test_type_text_calls_browser() -> None:
    browser = FakeBrowser()
    result = execute_tool(
        browser, "type_text", {"role": "textbox", "name": "Email", "text": "a@b"}
    )
    assert browser.calls == [("type_text", "textbox", "Email", "a@b")]
    assert not result.startswith(ERROR_PREFIX)


def test_playwright_error_becomes_first_line_string() -> None:
    browser = FakeBrowser(
        error=PlaywrightError("Timeout 5000ms exceeded.\nCall log:\n  - waiting")
    )
    result = execute_tool(browser, "click", {"role": "button", "name": "Nope"})
    assert result == f"{ERROR_PREFIX}Timeout 5000ms exceeded."


def test_unknown_tool_is_error() -> None:
    result = execute_tool(FakeBrowser(), "scroll", {})
    assert result.startswith(ERROR_PREFIX)


def test_missing_input_is_error() -> None:
    browser = FakeBrowser()
    result = execute_tool(browser, "click", {"role": "button"})
    assert result.startswith(ERROR_PREFIX)
    assert browser.calls == []
