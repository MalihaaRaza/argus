(maxChars) => {
  // Builds the page outline the agent "sees" and tags each visible interactive
  // element with data-argus-ref="eN", so Python can target exactly that element.
  // Run from browser.py via page.evaluate(script, maxChars). Plain JS, no dependencies.
  // (This comment lives inside the function because Playwright only recognises the
  // script as a function when the text starts with one.)
  // Not covered yet: elements inside iframes or shadow DOM.

  const REF_ATTR = "data-argus-ref";
  const INTERACTIVE_SELECTOR = [
    "button",
    "a[href]",
    "input:not([type=hidden])",
    "select",
    "textarea",
    "summary",
    "[contenteditable=true]",
    "[contenteditable='']",
  ].join(", ");
  const INTERACTIVE_ROLES = new Set([
    "button", "link", "checkbox", "radio", "tab", "menuitem", "option", "switch",
    "combobox", "textbox", "searchbox", "slider", "spinbutton", "treeitem",
  ]);
  const HEADING_SELECTOR = "h1, h2, h3, h4, h5, h6, [role=heading]";
  const SKIPPED_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SVG"]);
  const BUTTON_INPUT_TYPES = ["submit", "button", "reset", "image"];

  const NAME_MAX = 60; // accessible names
  const TEXT_MAX = 80; // headings and text lines
  const CONTEXT_PIECE_MAX = 40; // longer text (e.g. product descriptions) is skipped
  const CONTEXT_MAX = 60;
  const CONTEXT_MAX_LEVELS = 6; // how far up to look for an element's "card"

  const clean = (text) => (text || "").replace(/\s+/g, " ").trim();
  const truncate = (text, max) => (text.length > max ? text.slice(0, max - 1) + "…" : text);

  function isInteractive(el) {
    return el.matches(INTERACTIVE_SELECTOR) || INTERACTIVE_ROLES.has(el.getAttribute("role"));
  }

  function insideInteractive(el) {
    for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
      if (isInteractive(node)) return true;
    }
    return false;
  }

  function isVisible(el) {
    if (el.closest("[aria-hidden='true']")) return false;
    const style = getComputedStyle(el);
    if (style.display === "none" || style.visibility !== "visible") return false;
    // Zero size also catches elements whose ancestor has display:none.
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function roleOf(el) {
    const explicit = el.getAttribute("role");
    if (explicit) return explicit.split(" ")[0];
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const type = el.type;
      if (type === "checkbox" || type === "radio") return type;
      if (BUTTON_INPUT_TYPES.includes(type)) return "button";
      if (type === "range") return "slider";
      if (type === "number") return "spinbutton";
      if (type === "search") return "searchbox";
      return "textbox";
    }
    if (el.isContentEditable) return "textbox";
    return tag;
  }

  // Text of `container`, leaving out anything inside `exclude` (e.g. a <label>
  // that wraps its <select> shouldn't include the option texts).
  function textOutside(container, exclude) {
    const parts = [];
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (!exclude.contains(node)) parts.push(node.textContent);
    }
    return clean(parts.join(" "));
  }

  // Simplified version of the browser's accessible-name rules: first match wins.
  function accessibleName(el) {
    const isTextField = el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT";
    const sources = [
      () => (el.getAttribute("aria-labelledby") || "")
        .split(/\s+/)
        .map((id) => document.getElementById(id)?.innerText || "")
        .join(" "),
      () => el.getAttribute("aria-label"),
      () => Array.from(el.labels || []).map((label) => textOutside(label, el)).join(" "),
      () => el.getAttribute("placeholder"),
      () => el.getAttribute("alt") || el.querySelector("img[alt]")?.getAttribute("alt"),
      () => (el.tagName === "INPUT" && BUTTON_INPUT_TYPES.includes(el.type) ? el.value : ""),
      () => (isTextField ? "" : el.innerText),
      () => el.getAttribute("title"),
    ];
    for (const source of sources) {
      const name = clean(source());
      if (name) return truncate(name, NAME_MAX);
    }
    return "";
  }

  // Short state markers that help the agent verify what it did.
  function stateOf(el) {
    const parts = [];
    const role = roleOf(el);
    if (el.tagName === "SELECT") {
      const selected = clean(el.selectedOptions[0]?.text);
      if (selected) parts.push(`value="${truncate(selected, 40)}"`);
    } else if ((role === "textbox" || role === "searchbox" || role === "spinbutton") && el.value) {
      // Never echo passwords back into the model's context.
      const value = el.type === "password" ? "••••" : truncate(clean(el.value), 40);
      parts.push(`value="${value}"`);
    }
    if (el.checked || el.getAttribute("aria-checked") === "true") parts.push("checked");
    if (el.disabled || el.getAttribute("aria-disabled") === "true") parts.push("disabled");
    return parts.join(" ");
  }

  // Describe the element's surroundings, e.g. the product card a button sits in.
  // We climb while the parent does NOT contain another element with the same name,
  // so we stop at the largest container that belongs to this element alone.
  function contextHint(el, sameName) {
    let container = el;
    for (let level = 0; level < CONTEXT_MAX_LEVELS; level++) {
      const parent = container.parentElement;
      if (!parent || parent === document.body) break;
      if (sameName.some((other) => other !== el && parent.contains(other))) break;
      container = parent;
    }
    if (container === el) return "";

    // Keep only short visible text, so long descriptions don't crowd out the
    // product name and price. Text nodes with the same parent are joined without
    // a space, since frameworks often split "$29.99" into "$" and "29.99".
    const pieces = [];
    let lastParent = null;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = clean(node.textContent);
      if (!text || el.contains(node) || !isVisible(node.parentElement)) continue;
      if (node.parentElement === lastParent) {
        pieces[pieces.length - 1] += text;
      } else {
        pieces.push(text);
        lastParent = node.parentElement;
      }
    }
    const short = pieces.filter((piece) => piece.length <= CONTEXT_PIECE_MAX);
    return truncate([...new Set(short)].join(" "), CONTEXT_MAX);
  }

  // Text that belongs to this element itself, not to its children. Joined with
  // no separator so a split "$" + "29.99" reads "$29.99"; real spaces live in the
  // text nodes themselves.
  function directText(el) {
    return clean(
      Array.from(el.childNodes)
        .filter((node) => node.nodeType === Node.TEXT_NODE)
        .map((node) => node.textContent)
        .join(""),
    );
  }

  // ── 1. Clear refs from the previous snapshot ──────────────────────
  document.querySelectorAll(`[${REF_ATTR}]`).forEach((el) => el.removeAttribute(REF_ATTR));

  // ── 2. Collect items in document order ────────────────────────────
  const items = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, {
    // Skipping a <script>/<svg> also skips everything inside it.
    acceptNode: (el) =>
      SKIPPED_TAGS.has(el.tagName.toUpperCase()) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  for (let el = walker.nextNode(); el; el = walker.nextNode()) {
    if (isInteractive(el)) {
      if (!insideInteractive(el) && isVisible(el)) {
        items.push({ kind: "element", el, role: roleOf(el), name: accessibleName(el) });
      }
      continue;
    }
    const heading = el.closest(HEADING_SELECTOR);
    if (insideInteractive(el) || (heading && heading !== el)) {
      continue; // Already described by its interactive element or heading.
    }
    const isHeading = heading === el;
    const text = isHeading ? clean(el.innerText) : directText(el);
    if (!text || (!isHeading && text.length > TEXT_MAX) || !isVisible(el)) continue;
    const line = `${isHeading ? "heading" : "text"} "${truncate(text, TEXT_MAX)}"`;
    if (items.length && items[items.length - 1].line === line) continue; // consecutive duplicate
    items.push({ kind: "text", line });
  }

  // ── 3. Group elements by name, to know which ones need a context hint ─
  const byName = new Map();
  for (const item of items) {
    if (item.kind !== "element") continue;
    if (!byName.has(item.name)) byName.set(item.name, []);
    byName.get(item.name).push(item.el);
  }

  // ── 4. Build the output, tagging only the elements we actually show ──
  const lines = [`Page: ${clean(document.title) || "(no title)"} (${location.href})`];
  let length = lines[0].length;
  let nextRef = 1;
  let shown = 0;
  for (const item of items) {
    let line = item.line;
    const ref = `e${nextRef}`;
    if (item.kind === "element") {
      const sameName = byName.get(item.name);
      const state = stateOf(item.el);
      const context = sameName.length > 1 || !item.name ? contextHint(item.el, sameName) : "";
      line = `[${ref}] ${item.role} "${item.name}"`;
      if (state) line += ` ${state}`;
      if (context) line += ` (in: ${context})`;
    }
    if (length + 1 + line.length > maxChars) break;
    if (item.kind === "element") {
      item.el.setAttribute(REF_ATTR, ref);
      nextRef++;
    }
    lines.push(line);
    length += 1 + line.length;
    shown++;
  }
  if (shown < items.length) lines.push(`... (${items.length - shown} more items not shown)`);
  return lines.join("\n");
};
