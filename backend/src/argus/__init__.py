import argparse
import sys


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
    print(result)
    # A non-zero exit code on failure lets scripts and CI react to the verdict.
    sys.exit(0 if result.passed else 1)
