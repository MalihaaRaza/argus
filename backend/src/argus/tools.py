"""Tools the model can call, and the code that runs them on the browser.

TOOLS describes each tool to Claude (name, purpose, input schema).
execute_tool turns a tool call from Claude into a BrowserSession action.
"""

from typing import Any

from playwright.sync_api import Error as PlaywrightError

from argus.browser import BrowserSession, ElementNotFoundError

# Every failed tool result starts with this, so the agent loop can tell
# failures apart from successes without a separate return value.
ERROR_PREFIX = "Error: "

_REF_DESCRIPTION = (
    "The element's ref from the LATEST page snapshot, e.g. `e12` for the line "
    '`[e12] button "Add to cart"`. Refs are reassigned every time a new snapshot is '
    "taken, so never reuse a ref from an earlier snapshot."
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "click",
        "description": (
            "Click an element on the page, identified by its ref from the latest "
            "snapshot. When several elements share a name, use the `(in: ...)` hint "
            "to pick the right one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": _REF_DESCRIPTION},
            },
            "required": ["ref"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Replace the contents of a text field with the given text. Identify the "
            "field by its ref from the latest snapshot (usually a `textbox` line)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": _REF_DESCRIPTION},
                "text": {"type": "string", "description": "The text to enter."},
            },
            "required": ["ref", "text"],
        },
    },
    {
        "name": "finish",
        "description": (
            "End the test and report the verdict. Call this once you have enough evidence "
            "that the goal is met (passed=true) or cannot be met (passed=false)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "passed": {
                    "type": "boolean",
                    "description": "Whether the goal was achieved.",
                },
                "reason": {
                    "type": "string",
                    "description": "A short explanation citing what you saw on the page.",
                },
            },
            "required": ["passed", "reason"],
        },
    },
]


def execute_tool(browser: BrowserSession, name: str, tool_input: dict[str, Any]) -> str:
    """Run one browser tool and describe the outcome. Never raises.

    `finish` is not handled here: it doesn't touch the browser, so the
    agent loop deals with it directly.
    """
    try:
        if name == "click":
            browser.click(tool_input["ref"])
            return f"Clicked {tool_input['ref']}."
        if name == "type_text":
            browser.type_text(tool_input["ref"], tool_input["text"])
            return f"Typed into {tool_input['ref']}."
        return f"{ERROR_PREFIX}unknown tool {name!r}."
    except ElementNotFoundError as error:
        return f"{ERROR_PREFIX}{error}"
    except PlaywrightError as error:
        # Playwright messages include a long multi-line call log; the first
        # line says what went wrong and is all the model needs.
        first_line = (
            str(error).strip().splitlines()[0] if str(error).strip() else "unknown"
        )
        return f"{ERROR_PREFIX}{first_line}"
    except KeyError as error:
        return f"{ERROR_PREFIX}missing input field {error}."
