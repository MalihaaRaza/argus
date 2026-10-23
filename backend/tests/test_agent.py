from typing import Any, Self

import anthropic
import pytest
from fakes import (
    INPUT_TOKENS_PER_CALL,
    OUTPUT_TOKENS_PER_CALL,
    FakeBrowser,
    FakeClient,
    text,
    tool_use,
)
from playwright.sync_api import Error as PlaywrightError

import argus.agent
from argus.agent import _run_loop, run_agent
from argus.browser import NavigationError


def tokens(calls: int) -> dict[str, int]:
    """The token totals FakeClient reports after `calls` model calls."""
    return {
        "input_tokens": calls * INPUT_TOKENS_PER_CALL,
        "output_tokens": calls * OUTPUT_TOKENS_PER_CALL,
    }


def test_finish_returns_verdict() -> None:
    client = FakeClient(
        [[tool_use("t1", "finish", passed=True, reason="OK button visible")]]
    )
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=5)
    assert result.status == "passed"
    assert result.passed is True
    assert result.reason == "OK button visible"
    assert result.steps == 1


def test_first_message_has_goal_and_snapshot() -> None:
    client = FakeClient([[tool_use("t1", "finish", passed=True, reason="done")]])
    _run_loop(client, FakeBrowser(), "Log in", max_steps=5)
    first = client.history[0]
    assert first["role"] == "user"
    assert "Log in" in first["content"]
    assert '[e1] button "OK"' in first["content"]


def test_step_limit_reached() -> None:
    client = FakeClient([[tool_use("t1", "click", ref="e1")]], repeat_last=True)
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=3)
    # No verdict is an Argus problem, not an app failure.
    assert result.model_dump() == {
        "status": "error",
        "reason": "step limit reached",
        "steps": 3,
        **tokens(3),
    }
    assert len(client.calls) == 3


def test_results_then_one_snapshot_in_single_user_message() -> None:
    client = FakeClient(
        [
            [
                tool_use("a", "type_text", ref="e2", text="x@y"),
                tool_use("b", "click", ref="e1"),
            ],
            [tool_use("c", "finish", passed=True, reason="done")],
        ]
    )
    browser = FakeBrowser()
    result = _run_loop(client, browser, "goal", max_steps=5)

    assert result.steps == 2
    assert browser.calls == [
        ("type_text", "e2", "x@y"),
        ("click", "e1"),
    ]
    reply = client.history[2]  # [first user, assistant, tool results]
    assert reply["role"] == "user"
    *results, snapshot = reply["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert all(r["type"] == "tool_result" and not r["is_error"] for r in results)
    assert snapshot["type"] == "text"
    assert '[e1] button "OK"' in snapshot["text"]


def test_failure_skips_rest_of_turn_including_finish() -> None:
    client = FakeClient(
        [
            [
                tool_use("a", "click", ref="e99"),
                tool_use("b", "click", ref="e1"),
                tool_use("c", "finish", passed=True, reason="too early"),
            ],
            [tool_use("d", "finish", passed=False, reason="button broken")],
        ]
    )
    browser = FakeBrowser()
    result = _run_loop(client, browser, "goal", max_steps=5)

    # The finish in turn 1 came after a failure, so it must not end the run.
    assert result.model_dump() == {
        "status": "failed",
        "reason": "button broken",
        "steps": 2,
        **tokens(2),
    }
    assert browser.calls == [("click", "e99")]
    results = client.history[2]["content"][:-1]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]
    assert results[0]["content"] == "Error: Timeout 5000ms exceeded."
    assert all(r["is_error"] for r in results)
    assert all(r["content"].startswith("skipped") for r in results[1:])


def test_text_only_turn_gets_nudge() -> None:
    client = FakeClient(
        [
            [text("Let me think about this page.")],
            [tool_use("t1", "finish", passed=True, reason="done")],
        ]
    )
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=5)
    assert result.steps == 2
    nudge = client.history[2]
    assert nudge["role"] == "user"
    assert "finish" in nudge["content"]


def test_malformed_finish_is_returned_as_error() -> None:
    client = FakeClient(
        [
            [tool_use("a", "finish", passed=True)],  # reason missing
            [tool_use("b", "finish", passed=True, reason="retried with a reason")],
        ]
    )
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=5)

    assert result.model_dump() == {
        "status": "passed",
        "reason": "retried with a reason",
        "steps": 2,
        **tokens(2),
    }
    error_result = client.history[2]["content"][0]
    assert error_result["tool_use_id"] == "a"
    assert error_result["is_error"] is True
    assert (
        error_result["content"] == "Error: invalid finish input: reason: Field required"
    )


def test_finish_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    client = FakeClient([[tool_use("t1", "finish", passed=True, reason="done")]])
    _run_loop(client, FakeBrowser(), "goal", max_steps=5)
    assert "[step 1] finish(" in capsys.readouterr().out


def test_password_is_masked_in_step_log(capsys: pytest.CaptureFixture[str]) -> None:
    client = FakeClient(
        [
            [
                tool_use("a", "type_text", ref="e4", text="standard_user"),
                tool_use("b", "type_text", ref="e5", text="hunter2"),
            ],
            [tool_use("c", "finish", passed=True, reason="done")],
        ]
    )
    browser = FakeBrowser()
    _run_loop(client, browser, "goal", max_steps=5)

    out = capsys.readouterr().out
    assert "hunter2" not in out
    assert "'text': '••••'" in out
    assert "standard_user" in out  # Only password fields are masked
    assert ("type_text", "e5", "hunter2") in browser.calls  # The real text is typed


def test_api_error_becomes_error_result() -> None:
    client = FakeClient(
        [
            [tool_use("a", "click", ref="e1")],
            anthropic.AnthropicError("Overloaded\nmore details"),
        ]
    )
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=5)
    assert result.model_dump() == {
        "status": "error",
        "reason": "Anthropic API error: Overloaded",
        "steps": 1,  # Only the first model call completed
        **tokens(1),  # ...but what it spent is still reported
    }


def test_tokens_add_up_across_calls() -> None:
    client = FakeClient(
        [
            [tool_use("a", "click", ref="e1")],
            [text("Thinking...")],
            [tool_use("b", "finish", passed=True, reason="done")],
        ]
    )
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=5)
    assert (result.input_tokens, result.output_tokens) == (300, 60)


def test_on_step_receives_each_step_masked() -> None:
    client = FakeClient(
        [
            [
                tool_use("a", "type_text", ref="e5", text="hunter2"),
                tool_use("b", "click", ref="e99"),
            ],
            [tool_use("c", "finish", passed=True, reason="done")],
        ]
    )
    steps: list[tuple[int, str, dict[str, Any], str, bool]] = []

    def on_step(*step: Any) -> None:
        steps.append(step)

    _run_loop(client, FakeBrowser(), "goal", max_steps=5, on_step=on_step)

    assert steps == [
        (1, "type_text", {"ref": "e5", "text": "••••"}, "Typed into e5.", False),
        (1, "click", {"ref": "e99"}, "Error: Timeout 5000ms exceeded.", True),
        (2, "finish", {"passed": True, "reason": "done"}, "run finished", False),
    ]


class FailingSession:
    """Stands in for BrowserSession in run_agent: navigation always fails."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def goto(self, url: str) -> None:
        raise NavigationError(
            f"Could not open {url}: net::ERR_ABORTED (after 3 attempts)"
        )


class UnlaunchableSession:
    def __init__(self) -> None:
        raise PlaywrightError(
            "Executable doesn't exist at /chromium\nRun playwright install"
        )


def test_run_agent_navigation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(argus.agent, "BrowserSession", FailingSession)
    result = run_agent("https://shop.test", "goal")
    assert result.model_dump() == {
        "status": "error",
        "reason": "Could not open https://shop.test: net::ERR_ABORTED (after 3 attempts)",
        "steps": 0,
        **tokens(0),
    }


def test_run_agent_browser_launch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(argus.agent, "BrowserSession", UnlaunchableSession)
    result = run_agent("https://shop.test", "goal")
    assert result.status == "error"
    assert result.reason == "Browser error: Executable doesn't exist at /chromium"


def test_run_agent_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    # Would fail the test if reached: the key check must come before the browser.
    monkeypatch.setattr(argus.agent, "BrowserSession", UnlaunchableSession)
    result = run_agent("https://shop.test", "goal")
    assert result.status == "error"
    assert "ANTHROPIC_API_KEY" in result.reason
