"""The Celery app and task names, shared by the API (which queues runs) and the worker.

Kept separate from worker.py on purpose: the API only needs to publish a message to
Redis, so it imports this small module instead of the worker, which would drag in
the agent, Playwright and the Anthropic SDK.
"""

from typing import Any

from celery import Celery

from argus.config import get_settings

# The API sends tasks by name, and worker.py registers execute_run under the same
# name. Both use this constant, so the two can't drift apart.
EXECUTE_RUN_TASK = "argus.worker.execute_run"

celery_app = Celery("argus")
celery_app.conf.update(
    task_ignore_result=True,  # Results are saved in Postgres, not sent back via Celery
    worker_prefetch_multiplier=1,  # Runs are slow: don't reserve tasks we can't start yet
    broker_connection_retry_on_startup=True,  # Keep retrying if Redis isn't up yet
)


@celery_app.on_configure.connect
def _set_broker(sender: Celery, **kwargs: Any) -> None:
    """Read REDIS_URL only when Celery first needs its configuration.

    Like the engines in db.py, this means importing this module (which api.py does)
    doesn't need a .env.
    """
    sender.conf.broker_url = get_settings().redis_url
