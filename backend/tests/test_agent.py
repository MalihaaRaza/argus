from types import SimpleNamespace
from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError

from argus.agent import _run_loop


def tool_use(id: str, name: str, /, **tool_input: Any) -> SimpleNamespace:
    """Mimics an SDK ToolUseBlock: the loop only reads these four attributes.

    The `/` makes id and name positional-only, so a tool input field called
    `name` would land in **tool_input instead of clashing with the tool's name.
    """
    return SimpleNamespace(type="tool_use", id=id, name=name, input=tool_input)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


class FakeClient:
    """Stands in for anthropic.Anthropic: replays scripted turns, one per create()."""

    def __init__(
        self, turns: list[list[SimpleNamespace]], repeat_last: bool = False
    ) -> None:
        self.turns = turns
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # so client.messages.create(...) resolves to self.create

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if self.repeat_last:
            index = min(index, len(self.turns) - 1)
        return SimpleNamespace(content=self.turns[index])

    @property
    def history(self) -> list[dict[str, Any]]:
        # The loop passes the same list object on every call and keeps appending to it,
        # so the last call's messages is the full conversation.
        return self.calls[-1]["messages"]


class FakeBrowser:
    """Stands in for BrowserSession. Clicking ref "e99" fails; "e5" is a password field."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def snapshot(self) -> str:
        return '[e1] button "OK"'

    def describe(self, ref: str) -> str | None:
        return None  # Descriptions are covered in test_tools.py

    def is_secret(self, ref: str) -> bool:
        return ref == "e5"

    def click(self, ref: str) -> None:
        self.calls.append(("click", ref))
        if ref == "e99":
            raise PlaywrightError("Timeout 5000ms exceeded.\nCall log: ...")

    def type_text(self, ref: str, text: str) -> None:
        self.calls.append(("type_text", ref, text))


def test_finish_returns_verdict() -> None:
    client = FakeClient(
        [[tool_use("t1", "finish", passed=True, reason="OK button visible")]]
    )
    result = _run_loop(client, FakeBrowser(), "goal", max_steps=5)
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
    assert result.model_dump() == {
        "passed": False,
        "reason": "step limit reached",
        "steps": 3,
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
        "passed": False,
        "reason": "button broken",
        "steps": 2,
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
        "passed": True,
        "reason": "retried with a reason",
        "steps": 2,
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
