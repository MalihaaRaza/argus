"""The agent loop: Claude looks at the page, picks a tool, we run it, repeat."""

from typing import Any

import anthropic
from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from pydantic import ValidationError

from argus.browser import BrowserSession
from argus.models import RunResult
from argus.tools import ERROR_PREFIX, TOOLS, execute_tool

# Reads ANTHROPIC_API_KEY from .env into the environment, where the SDK looks for it.
load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are Argus, a QA agent testing a web page against a goal.

Work step by step. After each action you receive a fresh snapshot of the page:
- The first line shows the page title and URL.
- Interactive elements look like `[e12] button "Add to cart" (in: Sauce Labs Backpack $29.99)`:
  a ref, a role, a name, optional state (value="...", checked, disabled), and, when
  several elements share a name, an `(in: ...)` hint describing their surroundings.
- `heading "..."` and `text "..."` lines show what the page says; use them as evidence.

Rules:
- Pass an element's ref (e.g. `e12`) to click or type_text.
- Refs change with every snapshot. Only use refs from the latest snapshot.
- When names repeat, use the `(in: ...)` hint to pick the right element.
- If an action fails, read the error and try a different approach.
- When you have enough evidence, call finish with passed=true or passed=false and a
  short reason that cites what you saw on the page."""


def run_agent(url: str, goal: str, max_steps: int = 15) -> RunResult:
    """Open the page and let Claude test it against the goal."""
    client = anthropic.Anthropic()
    with BrowserSession() as browser:
        browser.goto(url)
        return _run_loop(client, browser, goal, max_steps)


def _run_loop(
    client: anthropic.Anthropic, browser: BrowserSession, goal: str, max_steps: int
) -> RunResult:
    """The decision loop. Takes its dependencies as arguments so tests can pass fakes."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": f"Goal: {goal}\n\nCurrent page snapshot:\n{_snapshot_text(browser)}",
        }
    ]

    for step in range(1, max_steps + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [block for block in response.content if block.type == "tool_use"]

        if not tool_calls:
            # Claude answered with text only. Nudge it so the conversation
            # keeps alternating user -> assistant.
            messages.append(
                {
                    "role": "user",
                    "content": "Use click or type_text to continue, or call finish with your verdict.",
                }
            )
            continue

        results: list[dict[str, Any]] = []
        failed = False

        for call in tool_calls:
            if failed:
                # An earlier action this turn failed, so the page may not be in the
                # state Claude expected. Don't run later actions (or trust a finish).
                result, is_error = (
                    "skipped: an earlier action in this turn failed",
                    True,
                )
            elif call.name == "finish":
                try:
                    # Our step count goes last so the model can't override it.
                    verdict = RunResult.model_validate({**call.input, "steps": step})
                except ValidationError as error:
                    # Malformed verdict: tell Claude what was wrong so it can retry.
                    result = f"{ERROR_PREFIX}invalid finish input: {_describe(error)}"
                    is_error = failed = True
                else:
                    _print_step(step, call.name, call.input, "run finished")
                    return verdict
            else:
                result = execute_tool(browser, call.name, call.input)
                is_error = result.startswith(ERROR_PREFIX)
                if is_error:
                    failed = True

            _print_step(step, call.name, _loggable(browser, call.input), result)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result,
                    "is_error": is_error,
                }
            )

        # One user message: every tool result first (the API requires them to
        # come before other content), then ONE fresh snapshot of the page.
        snapshot_block = {
            "type": "text",
            "text": f"Current page snapshot:\n{_snapshot_text(browser)}",
        }
        messages.append({"role": "user", "content": results + [snapshot_block]})

    return RunResult(passed=False, reason="step limit reached", steps=max_steps)


def _snapshot_text(browser: BrowserSession) -> str:
    """Take a snapshot, turning failures (e.g. mid-navigation) into text for the model."""
    try:
        return browser.snapshot()
    except PlaywrightError as error:
        return f"{ERROR_PREFIX}could not take snapshot: {error}"


def _describe(error: ValidationError) -> str:
    """One short line per problem, e.g. "reason: Field required" (str(error) is verbose)."""
    return "; ".join(
        f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
        for issue in error.errors()
    )


def _loggable(browser: BrowserSession, tool_input: dict[str, Any]) -> dict[str, Any]:
    """The tool input with text typed into password fields masked, safe to print.

    Uses the refs of the latest snapshot, which is still the one the model chose
    from: a new snapshot is only taken after all of this turn's tools have run.
    """
    if "text" in tool_input and browser.is_secret(tool_input.get("ref", "")):
        return {**tool_input, "text": "••••"}
    return tool_input


def _print_step(step: int, name: str, tool_input: dict[str, Any], result: str) -> None:
    print(f"[step {step}] {name}({tool_input}) -> {result}")
