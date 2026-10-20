import pytest

import argus
import argus.agent
from argus.models import RunResult


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
