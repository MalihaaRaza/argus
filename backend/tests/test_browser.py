"""Navigation retry logic, tested with a fake page (no real browser needed)."""

import pytest
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from argus.browser import NavigationError, navigate_with_retry

ABORTED = PlaywrightError(
    "Page.goto: net::ERR_ABORTED at https://shop.test\nCall log: ..."
)


class FakePage:
    """goto() raises the scripted errors in order, then succeeds."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.goto_calls = 0

    def goto(self, url: str) -> None:
        self.goto_calls += 1
        if self.errors:
            raise self.errors.pop(0)


class FakeSleep:
    """Records requested delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def test_success_needs_no_retry() -> None:
    page, sleep = FakePage([]), FakeSleep()
    navigate_with_retry(page, "https://shop.test", sleep=sleep)
    assert page.goto_calls == 1
    assert sleep.delays == []


def test_transient_error_then_success(capsys: pytest.CaptureFixture[str]) -> None:
    page, sleep = FakePage([ABORTED]), FakeSleep()
    navigate_with_retry(page, "https://shop.test", sleep=sleep)

    assert page.goto_calls == 2
    assert sleep.delays == [1]
    out = capsys.readouterr().out
    assert "ERR_ABORTED" in out
    assert "retrying in 1s (attempt 2 of 3)" in out
    assert "Call log" not in out  # Only the first line of the error is shown


def test_timeout_is_transient() -> None:
    page = FakePage([PlaywrightTimeoutError("Timeout 30000ms exceeded.")])
    sleep = FakeSleep()
    navigate_with_retry(page, "https://shop.test", sleep=sleep)
    assert page.goto_calls == 2


def test_fails_every_time() -> None:
    page, sleep = FakePage([ABORTED, ABORTED, ABORTED]), FakeSleep()
    with pytest.raises(NavigationError, match=r"ERR_ABORTED .*after 3 attempts"):
        navigate_with_retry(page, "https://shop.test", sleep=sleep)
    assert page.goto_calls == 3
    assert sleep.delays == [1, 3]


def test_permanent_error_is_not_retried() -> None:
    dns = PlaywrightError("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://shpo.test")
    page, sleep = FakePage([dns]), FakeSleep()
    with pytest.raises(NavigationError, match="ERR_NAME_NOT_RESOLVED"):
        navigate_with_retry(page, "https://shpo.test", sleep=sleep)
    assert page.goto_calls == 1
    assert sleep.delays == []
