"""Stand-ins for the Anthropic client and the browser, shared by the agent and worker tests.

Test modules import this as `from fakes import ...`: pytest puts the tests/ folder
on sys.path because it has no __init__.py.
"""

from types import SimpleNamespace
from typing import Any

from playwright.sync_api import Error as PlaywrightError

# Token usage FakeClient reports for every model call.
INPUT_TOKENS_PER_CALL = 100
OUTPUT_TOKENS_PER_CALL = 20


def tool_use(id: str, name: str, /, **tool_input: Any) -> SimpleNamespace:
    """Mimics an SDK ToolUseBlock: the loop only reads these four attributes.

    The `/` makes id and name positional-only, so a tool input field called
    `name` would land in **tool_input instead of clashing with the tool's name.
    """
    return SimpleNamespace(type="tool_use", id=id, name=name, input=tool_input)


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


class FakeClient:
    """Stands in for anthropic.Anthropic: replays scripted turns, one per create().

    A turn can also be an exception, which create() raises (to simulate API errors).
    """

    def __init__(
        self, turns: list[list[SimpleNamespace] | Exception], repeat_last: bool = False
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
        turn = self.turns[index]
        if isinstance(turn, Exception):
            raise turn
        usage = SimpleNamespace(
            input_tokens=INPUT_TOKENS_PER_CALL, output_tokens=OUTPUT_TOKENS_PER_CALL
        )
        return SimpleNamespace(content=turn, usage=usage)

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


class FakeRedis:
    """Stands in for the worker's redis.Redis: records publish() calls.

    Set `error` to make publish() raise it instead (to simulate Redis being down).
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def publish(self, channel: str, message: str) -> int:
        if self.error is not None:
            raise self.error
        self.published.append((channel, message))
        return 0  # Number of subscribers that received it
