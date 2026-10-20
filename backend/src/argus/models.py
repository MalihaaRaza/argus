"""Structured data passed between parts of Argus."""

from typing import Literal

from pydantic import BaseModel


class Verdict(BaseModel):
    """What the model reports through the `finish` tool."""

    passed: bool
    reason: str


class RunResult(BaseModel):
    """The outcome of one agent run.

    status:
      - "passed": the app met the goal.
      - "failed": the app did not meet the goal (a bug in the app under test).
      - "error":  Argus couldn't complete the test (navigation, API, or browser
                  failure, or no verdict within the step limit).
    """

    status: Literal["passed", "failed", "error"]
    reason: str
    steps: int  # Number of model calls the run completed

    @property
    def passed(self) -> bool:
        # Derived from status rather than stored, so the two can never disagree.
        return self.status == "passed"
