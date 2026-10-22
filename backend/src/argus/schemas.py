"""Request and response shapes for the HTTP API.

Kept separate from the ORM models in db.py, so the API's shape and the table layout
can change independently (e.g. a new column doesn't automatically become public).
"""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, HttpUrl, StringConstraints

from argus.db import RunStatus

# Whitespace is stripped before the length check, so a blank goal counts as empty.
Goal = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class RunCreate(BaseModel):
    """Body of POST /runs."""

    url: HttpUrl  # Only http:// and https:// URLs are accepted
    goal: Goal


class RunStepOut(BaseModel):
    # from_attributes: build from an ORM object's attributes, not only from a dict.
    model_config = ConfigDict(from_attributes=True)

    step_number: int
    tool_name: str
    tool_input: dict[str, Any]
    result: str
    is_error: bool
    created_at: datetime


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    url: str
    goal: str
    status: RunStatus
    reason: str | None
    step_count: int
    input_tokens: int
    output_tokens: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class RunDetailOut(RunOut):
    """A run with its steps, for GET /runs/{run_id}."""

    steps: list[RunStepOut]
