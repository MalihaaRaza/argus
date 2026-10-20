from playwright.sync_api import Error as PlaywrightError

from argus.browser import ElementNotFoundError
from argus.tools import ERROR_PREFIX, TOOLS, execute_tool


class FakeBrowser:
    """Stands in for BrowserSession: records calls, optionally raises."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple] = []
        self.error = error
        self.descriptions = {
            "e1": 'textbox "Email"',
            "e3": 'button "Add to cart" in: Backpack $29.99',
        }

    def describe(self, ref: str) -> str | None:
        return self.descriptions.get(ref)

    def click(self, ref: str) -> None:
        self.calls.append(("click", ref))
        if self.error:
            raise self.error

    def type_text(self, ref: str, text: str) -> None:
        self.calls.append(("type_text", ref, text))
        if self.error:
            raise self.error


def test_tool_names() -> None:
    assert [tool["name"] for tool in TOOLS] == ["click", "type_text", "finish"]


def test_browser_tools_take_a_ref() -> None:
    schemas = {tool["name"]: tool["input_schema"] for tool in TOOLS}
    assert schemas["click"]["required"] == ["ref"]
    assert schemas["type_text"]["required"] == ["ref", "text"]


def test_click_calls_browser() -> None:
    browser = FakeBrowser()
    result = execute_tool(browser, "click", {"ref": "e3"})
    assert browser.calls == [("click", "e3")]
    assert result == 'Clicked e3 (button "Add to cart" in: Backpack $29.99).'


def test_unknown_ref_description_falls_back_to_ref() -> None:
    result = execute_tool(FakeBrowser(), "click", {"ref": "e7"})
    assert result == "Clicked e7."


def test_type_text_calls_browser() -> None:
    browser = FakeBrowser()
    result = execute_tool(browser, "type_text", {"ref": "e1", "text": "a@b"})
    assert browser.calls == [("type_text", "e1", "a@b")]
    assert result == 'Typed into e1 (textbox "Email").'


def test_playwright_error_becomes_first_line_string() -> None:
    browser = FakeBrowser(
        error=PlaywrightError("Timeout 5000ms exceeded.\nCall log:\n  - waiting")
    )
    result = execute_tool(browser, "click", {"ref": "e1"})
    assert result == f"{ERROR_PREFIX}Timeout 5000ms exceeded."


def test_unknown_ref_becomes_error_string() -> None:
    browser = FakeBrowser(error=ElementNotFoundError("No element with ref 'e9'."))
    result = execute_tool(browser, "click", {"ref": "e9"})
    assert result == f"{ERROR_PREFIX}No element with ref 'e9'."


def test_unknown_tool_is_error() -> None:
    result = execute_tool(FakeBrowser(), "scroll", {})
    assert result.startswith(ERROR_PREFIX)


def test_missing_input_is_error() -> None:
    browser = FakeBrowser()
    result = execute_tool(browser, "type_text", {"ref": "e1"})
    assert result.startswith(ERROR_PREFIX)
    assert browser.calls == []
