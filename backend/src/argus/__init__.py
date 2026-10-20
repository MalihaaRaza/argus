import argparse
import sys

# 0 = the app met the goal, 1 = it didn't (an app bug), 2 = Argus couldn't run
# the test. argparse also exits with 2 for bad arguments, which fits "couldn't run".
EXIT_CODES = {"passed": 0, "failed": 1, "error": 2}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="argus", description="Test a web page against a plain-English goal."
    )
    parser.add_argument("url", help="The page to open, e.g. https://example.com")
    parser.add_argument("goal", help="What to check, in plain English")
    parser.add_argument(
        "--max-steps", type=int, default=15, help="Model calls allowed (default 15)"
    )
    args = parser.parse_args()

    # Imported here so `argus --help` stays fast and works without an API key.
    from argus.agent import run_agent

    result = run_agent(args.url, args.goal, max_steps=args.max_steps)
    print()
    print(f"Result: {result.status.upper()} after {result.steps} steps")
    print(f"Reason: {result.reason}")
    # Distinct exit codes let scripts and CI tell an app bug from a broken run.
    sys.exit(EXIT_CODES[result.status])
