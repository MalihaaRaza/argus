"""Integration tests: run snapshot.js and ref targeting in a real (headless) Chromium."""

import re
from collections.abc import Iterator

import pytest
from playwright.sync_api import Error as PlaywrightError

from argus.browser import MAX_SNAPSHOT_CHARS, BrowserSession, ElementNotFoundError

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser() -> Iterator[BrowserSession]:
    # One browser for the whole file: launching Chromium takes most of the test time.
    try:
        session = BrowserSession(headless=True, slow_mo_ms=0)
    except PlaywrightError as error:
        pytest.skip(f"Chromium not available: {error}")
    with session:
        yield session


def ref_lines(snapshot: str) -> dict[str, str]:
    """Map each ref to the rest of its line, e.g. {'e1': 'button "OK"'}."""
    return dict(re.findall(r"^\[(e\d+)\] (.*)$", snapshot, flags=re.MULTILINE))


def ref_for(snapshot: str, text: str) -> str:
    """The ref of the first element line containing `text`."""
    return next(ref for ref, line in ref_lines(snapshot).items() if text in line)


# Over 80 chars: too long for a text line, and for a context hint (40-char limit).
LONG = (
    "A long product description that is far too wordy to be a useful hint, "
    "much like the real ones on saucedemo.com."
)

PRODUCTS = f"""
<h1>Products</h1>
<div class="list">
  <div class="card">
    <a href="#backpack">Backpack</a><p>{LONG}</p><div>$29.99</div>
    <button onclick="document.body.dataset.added = 'Backpack'">Add to cart</button>
  </div>
  <div class="card">
    <a href="#light">Bike Light</a><p>{LONG}</p><div>$<!-- -->9.99</div>
    <button onclick="document.body.dataset.added = 'Bike Light'">Add to cart</button>
  </div>
</div>
"""


def test_duplicate_names_get_distinct_refs_and_context(browser: BrowserSession) -> None:
    browser.page.set_content(PRODUCTS)
    snapshot = browser.snapshot()

    add_buttons = {
        ref: line for ref, line in ref_lines(snapshot).items() if "Add to cart" in line
    }
    assert list(add_buttons.values()) == [
        'button "Add to cart" (in: Backpack $29.99)',
        'button "Add to cart" (in: Bike Light $9.99)',
    ]
    # Unique names don't need a hint, so none is added (saves tokens).
    assert 'link "Backpack"' in snapshot
    assert 'link "Backpack" (in:' not in snapshot

    second = list(add_buttons)[1]
    browser.click(second)
    assert browser.page.evaluate("document.body.dataset.added") == "Bike Light"


def test_headings_and_short_text_are_listed(browser: BrowserSession) -> None:
    browser.page.set_content(
        PRODUCTS + '<p class="error">Epic sadface: Username is required</p>'
    )
    snapshot = browser.snapshot()
    lines = snapshot.splitlines()
    assert lines[0].startswith("Page: ")
    assert 'heading "Products"' in lines
    assert 'text "$29.99"' in lines
    assert 'text "$9.99"' in lines  # Split into two text nodes, like React renders it
    assert 'text "Epic sadface: Username is required"' in lines
    assert LONG not in snapshot  # Too long for a text line


def test_hidden_elements_are_skipped(browser: BrowserSession) -> None:
    browser.page.set_content(
        """
        <button>Visible</button>
        <button style="display:none">DisplayNone</button>
        <div style="display:none"><button>InsideHidden</button></div>
        <button style="visibility:hidden">VisibilityHidden</button>
        <button aria-hidden="true">AriaHidden</button>
        <button style="width:0;height:0;padding:0;border:0;overflow:hidden">ZeroSize</button>
        <input type="hidden" name="token" value="secret">
        """
    )
    snapshot = browser.snapshot()
    assert list(ref_lines(snapshot).values()) == ['button "Visible"']
    for hidden in ["DisplayNone", "InsideHidden", "VisibilityHidden", "AriaHidden"]:
        assert hidden not in snapshot
    assert "ZeroSize" not in snapshot


def test_accessible_names(browser: BrowserSession) -> None:
    browser.page.set_content(
        """
        <label for="email">Email</label><input id="email" type="email">
        <label>Password <input type="password"></label>
        <label>Country <select><option>Norway</option><option>Peru</option></select></label>
        <button aria-label="Close">×</button>
        <input type="submit" value="Sign up">
        <input placeholder="Search products">
        <a href="/home"><img src="logo.png" alt="Home"></a>
        """
    )
    assert list(ref_lines(browser.snapshot()).values()) == [
        'textbox "Email"',
        'textbox "Password"',
        'combobox "Country" value="Norway"',
        'button "Close"',
        'button "Sign up"',
        'textbox "Search products"',
        'link "Home"',
    ]


def test_type_text_fills_the_ref_and_shows_value(browser: BrowserSession) -> None:
    browser.page.set_content(
        '<label for="email">Email</label><input id="email">'
        '<label for="pw">Password</label><input id="pw" type="password">'
    )
    snapshot = browser.snapshot()
    browser.type_text(ref_for(snapshot, '"Email"'), "a@b.c")
    browser.type_text(ref_for(snapshot, '"Password"'), "hunter2")

    assert browser.page.input_value("#email") == "a@b.c"
    after = browser.snapshot()
    assert 'textbox "Email" value="a@b.c"' in after
    assert 'textbox "Password" value="••••"' in after  # Passwords are masked
    assert "hunter2" not in after


def test_refs_are_renumbered_and_stale_refs_fail(browser: BrowserSession) -> None:
    browser.page.set_content("<button id='first'>First</button><button>Second</button>")
    assert ref_lines(browser.snapshot()) == {
        "e1": 'button "First"',
        "e2": 'button "Second"',
    }

    browser.page.evaluate("document.getElementById('first').remove()")
    with pytest.raises(ElementNotFoundError, match="latest snapshot"):
        browser.click("e1")  # Its element is gone

    assert ref_lines(browser.snapshot()) == {"e1": 'button "Second"'}
    # The old e2 tag was cleared, so only one element carries a ref now.
    assert browser.page.locator("[data-argus-ref]").count() == 1


def test_malformed_ref_is_rejected(browser: BrowserSession) -> None:
    browser.page.set_content("<button>OK</button>")
    browser.snapshot()
    with pytest.raises(ElementNotFoundError, match="latest snapshot"):
        browser.click('e1"], button, [x="')


def test_snapshot_is_capped(browser: BrowserSession) -> None:
    buttons = "".join(f"<button>Button number {i}</button>" for i in range(1000))
    browser.page.set_content(buttons)
    snapshot = browser.snapshot()
    lines = snapshot.splitlines()

    assert re.fullmatch(r"\.\.\. \(\d+ more items not shown\)", lines[-1])
    assert len(snapshot) <= MAX_SNAPSHOT_CHARS + len(lines[-1]) + 1
    # Only the elements we showed are tagged, so every tagged ref is one the model saw.
    assert browser.page.locator("[data-argus-ref]").count() == len(ref_lines(snapshot))
