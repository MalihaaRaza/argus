"""Structured data passed between parts of Argus."""

from pydantic import BaseModel


class RunResult(BaseModel):
    """The verdict of one agent run."""

    passed: bool
    reason: str
    steps: int  # Number of model calls the run used
