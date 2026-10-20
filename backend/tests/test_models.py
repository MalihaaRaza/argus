import pytest
from pydantic import ValidationError

from argus.models import RunResult


@pytest.mark.parametrize(
    ("status", "passed"), [("passed", True), ("failed", False), ("error", False)]
)
def test_passed_follows_status(status: str, passed: bool) -> None:
    assert RunResult(status=status, reason="r", steps=1).passed is passed


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunResult(status="skipped", reason="r", steps=1)
