"""Manual check: drive the browser by hand, the same way the agent will."""

import re

from argus.browser import BrowserSession


def find_ref(snapshot: str, line_text: str) -> str:
    """Return the ref of the first snapshot line containing `line_text`, e.g. 'e2'."""
    for line in snapshot.splitlines():
        match = re.match(r"\[(e\d+)\] (.*)", line)
        if match and line_text in match.group(2):
            return match.group(1)
    raise LookupError(f"No snapshot line contains {line_text!r}")


with BrowserSession() as browser:
    browser.goto("https://www.saucedemo.com")

    print("=== What the agent sees on the login page ===")
    snapshot = browser.snapshot()
    print(snapshot)

    browser.type_text(find_ref(snapshot, 'textbox "Username"'), "standard_user")
    browser.type_text(find_ref(snapshot, 'textbox "Password"'), "secret_sauce")
    browser.click(find_ref(snapshot, 'button "Login"'))

    print("\n=== What the agent sees after logging in ===")
    snapshot = browser.snapshot()
    print(snapshot)

    print()
    print(len(browser.page.content()), "characters of HTML")
    print(
        len(browser.page.locator("body").aria_snapshot()), "characters of aria snapshot"
    )
    print(len(snapshot), "characters of Argus snapshot")
