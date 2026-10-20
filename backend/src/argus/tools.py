"""Tools the model can call, and the code that runs them on the browser.

TOOLS describes each tool to Claude (name, purpose, input schema).
execute_tool turns a tool call from Claude into a BrowserSession action.
"""

from typing import Any

from playwright.sync_api import Error as PlaywrightError

from argus.browser import BrowserSession

# Every failed tool result starts with this, so the agent loop can tell
# failures apart from successes without a separate return value.
ERROR_PREFIX = "Error: "

_ROLE_DESCRIPTION = (
    "The element's ARIA role, exactly as shown in the accessibility snapshot. "
    'For the snapshot line `- button "Sign in"`, the role is `button`.'
)
_NAME_DESCRIPTION = (
    "The element's accessible name, exactly as shown in quotes in the accessibility snapshot. "
    'For the snapshot line `- button "Sign in"`, the name is `Sign in`.'
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "click",
        "description": (
            "Click an element on the page. Use the exact role and name shown in the "
            "accessibility snapshot; do not guess or paraphrase them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": _ROLE_DESCRIPTION},
                "name": {"type": "string", "description": _NAME_DESCRIPTION},
            },
            "required": ["role", "name"],
        },
    },
    {
        "name": "type_text",
        "description": (
            "Replace the contents of a text field with the given text. Use the exact role "
            "(usually `textbox`) and name shown in the accessibility snapshot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": _ROLE_DESCRIPTION},
                "name": {"type": "string", "description": _NAME_DESCRIPTION},
                "text": {"type": "string", "description": "The text to enter."},
            },
            "required": ["role", "name", "text"],
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
            browser.click(tool_input["role"], tool_input["name"])
            return f"Clicked {tool_input['role']} {tool_input['name']!r}."
        if name == "type_text":
            browser.type_text(
                tool_input["role"], tool_input["name"], tool_input["text"]
            )
            return f"Typed into {tool_input['role']} {tool_input['name']!r}."
        return f"{ERROR_PREFIX}unknown tool {name!r}."
    except PlaywrightError as error:
        # Playwright messages include a long multi-line call log; the first
        # line says what went wrong and is all the model needs.
        first_line = (
            str(error).strip().splitlines()[0] if str(error).strip() else "unknown"
        )
        return f"{ERROR_PREFIX}{first_line}"
    except KeyError as error:
        return f"{ERROR_PREFIX}missing input field {error}."
