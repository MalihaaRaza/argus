"""The API and the worker agree on the task name, and the API stays lightweight."""

import subprocess
import sys

from argus.celery_app import EXECUTE_RUN_TASK, celery_app
from argus.worker import execute_run


def test_worker_registers_task_under_shared_name() -> None:
    # The API sends EXECUTE_RUN_TASK by name, so the worker must register it so.
    assert execute_run.name == EXECUTE_RUN_TASK
    # (Not an `is` check: @celery_app.task returns a lazy proxy for the task.)
    assert EXECUTE_RUN_TASK in celery_app.tasks


def test_api_does_not_import_worker_or_agent() -> None:
    # A fresh interpreter, because this test process has already imported the
    # worker (above) and everything it pulls in.
    heavy = ["argus.worker", "argus.agent", "playwright", "anthropic"]
    code = f"import sys, argus.api; print([m for m in {heavy!r} if m in sys.modules])"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
