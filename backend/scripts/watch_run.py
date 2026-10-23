"""Watch a run live in the terminal: prints each event from the API's WebSocket.

Run from backend/ while the API is running:
  uv run python scripts/watch_run.py <run_id>
  uv run python scripts/watch_run.py <run_id> --api ws://127.0.0.1:8000

Exit codes: 0 when the run finished, 1 if there is no such run, 2 if the API
can't be reached or the connection broke.
"""

import argparse
import json
import sys
import uuid
from typing import Any

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

NOT_FOUND_CLOSE_CODE = 4404  # Matches argus.api


def describe_step(step: dict[str, Any]) -> str:
    # Same shape as the agent's own step log line.
    line = f"[step {step['step_number']}] {step['tool_name']}({step['tool_input']})"
    line += f" -> {step['result']}"
    return line + ("  (tool error)" if step["is_error"] else "")


def describe(message: dict[str, Any]) -> str:
    data = message["data"]
    match message["type"]:
        case "snapshot":
            lines = [
                f"Run {data['id']}: {data['status']}",
                f"  {data['url']}",
                f"  Goal: {data['goal']}",
            ]
            lines += [describe_step(step) for step in data["steps"]]
            return "\n".join(lines)
        case "status":
            return f"Status: {data['status']}"
        case "step":
            return describe_step(data)
        case "finished":
            return (
                f"Finished: {data['status'].upper()}: {data['reason']}\n"
                f"  {data['step_count']} steps, {data['input_tokens']} input / "
                f"{data['output_tokens']} output tokens"
            )
        case _:  # A newer API sending something this script doesn't know yet
            return json.dumps(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an Argus run live.")
    parser.add_argument("run_id", type=uuid.UUID, help="The run's id")
    parser.add_argument(
        "--api",
        default="ws://127.0.0.1:8000",
        help="API base URL (default %(default)s)",
    )
    args = parser.parse_args()
    url = f"{args.api}/ws/runs/{args.run_id}"

    try:
        with connect(url) as ws:
            try:
                for raw in ws:
                    print(describe(json.loads(raw)), flush=True)
            except ConnectionClosed:
                pass  # Abnormal close codes (like 4404) end up here; checked below
            code = ws.close_code
    except OSError as error:
        print(f"Could not connect to {url}: {error}", file=sys.stderr)
        sys.exit(2)

    if code == NOT_FOUND_CLOSE_CODE:
        print(f"No run with id {args.run_id}", file=sys.stderr)
        sys.exit(1)
    if code != 1000:
        print(f"Connection closed unexpectedly (code {code})", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
