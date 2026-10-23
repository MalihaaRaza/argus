import logging
from collections.abc import Iterator

import pytest

import argus
import argus.agent
from argus.models import RunResult


@pytest.fixture(autouse=True)
def restore_argus_logger() -> Iterator[None]:
    """main() adds a handler to the "argus" logger; remove it after each test."""
    logger = logging.getLogger("argus")
    handlers, level = logger.handlers[:], logger.level
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)


@pytest.mark.parametrize(
    ("status", "exit_code"), [("passed", 0), ("failed", 1), ("error", 2)]
)
def test_exit_code_matches_status(
    status: str,
    exit_code: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_run_agent(url: str, goal: str, max_steps: int) -> RunResult:
        return RunResult(status=status, reason="because", steps=4)

    # main() imports run_agent from argus.agent when it runs, so patching the
    # module attribute is enough.
    monkeypatch.setattr(argus.agent, "run_agent", fake_run_agent)
    monkeypatch.setattr("sys.argv", ["argus", "https://shop.test", "Buy a hat"])

    with pytest.raises(SystemExit) as exit_info:
        argus.main()

    assert exit_info.value.code == exit_code
    out = capsys.readouterr().out
    assert f"Result: {status.upper()} after 4 steps" in out
    assert "Reason: because" in out


def test_bad_arguments_exit_with_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["argus", "https://shop.test"])  # goal missing
    with pytest.raises(SystemExit) as exit_info:
        argus.main()
    assert exit_info.value.code == 2


def test_step_lines_are_shown_on_the_terminal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_agent(url: str, goal: str, max_steps: int) -> RunResult:
        # Logged the way the agent logs a step.
        logging.getLogger("argus.agent").info("[step 1] click({'ref': 'e1'}) -> ok")
        return RunResult(status="passed", reason="because", steps=1)

    monkeypatch.setattr(argus.agent, "run_agent", fake_run_agent)
    monkeypatch.setattr("sys.argv", ["argus", "https://shop.test", "Buy a hat"])

    with pytest.raises(SystemExit):
        argus.main()

    captured = capsys.readouterr()
    assert "[step 1] click({'ref': 'e1'}) -> ok\n" in captured.err  # Plain line
    assert "Result: PASSED" in captured.out  # The result stays on stdout
