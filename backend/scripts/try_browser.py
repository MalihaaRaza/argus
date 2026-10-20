"""Manual check: drive the browser by hand, the same way the agent will."""

from argus.browser import BrowserSession

with BrowserSession() as browser:
    browser.goto("https://www.saucedemo.com")

    print("=== What the agent sees on the login page ===")
    print(browser.snapshot())

    browser.type_text("textbox", "Username", "standard_user")
    browser.type_text("textbox", "Password", "secret_sauce")
    browser.click("button", "Login")

    print("\n=== What the agent sees after logging in ===")
    print(browser.snapshot()[:1500])  # First part only; the product page is long

    print(len(browser.page.content()), "characters of HTML")
    print(len(browser.snapshot()), "characters of snapshot")