"use strict";

// Exercise the real key listener in a minimal VM, without plan or model requests.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/qa.html"), "utf8");
const originalScript = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const script = originalScript.replace(/\nloadPlan\(\);\s*$/, "\n");
assert.notEqual(script, originalScript, "only the automatic plan fetch is suppressed by the harness");

class Element {
  constructor(attributes = "") {
    this.value = ""; this.children = []; this.listeners = {}; this.selectionStart = 0; this.selectionEnd = 0;
    this.disabled = /\bdisabled\b/.test(attributes); this.readOnly = /\breadonly\b/i.test(attributes);
    this.placeholder = attributes.match(/\bplaceholder="([^"]*)"/)?.[1] || "";
  }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  focus() { this.focused = true; }
}
function harness({ ready = true, busy = false, value = "" } = {}) {
  const elements = new Map([...page.matchAll(/<[a-z][\w-]*\b([^>]*\bid="([^"]+)"[^>]*)>/gi)]
    .map(match => [match[2], new Element(match[1])]));
  const calls = [];
  const context = vm.createContext({ document: { getElementById: id => elements.get(id) },
    fetch: (...args) => { calls.push(args); throw new Error("The example shortcut must not fetch"); } });
  vm.runInContext(script, context);
  vm.runInContext("state.plan = " + (ready ? "{}" : "null") + "; state.busy = " + busy + ";", context);
  const query = elements.get("query"); query.disabled = false; query.value = value;
  let prevented = 0;
  return { query, calls, context, elements,
    key(options = {}) {
      const event = { key: "Tab", shiftKey: false, ctrlKey: false, altKey: false, metaKey: false,
        isComposing: false, target: query, currentTarget: query,
        preventDefault() { prevented++; }, ...options };
      const listeners = query.listeners.keydown || [];
      assert(listeners.length > 0, "query must register a keydown listener");
      for (const listener of listeners) listener(event);
      return prevented;
    } };
}
function assertUnchanged(h, options = {}) {
  const value = h.query.value;
  assert.equal(h.key(options), 0, "normal keyboard focus behavior must be preserved");
  assert.equal(h.query.value, value);
  assert.equal(h.query.selectionStart, 0);
  assert.equal(h.query.selectionEnd, 0);
  assert.equal(h.calls.length, 0);
}

test("plain Tab fills the actual placeholder example, keeps the caret at the end and does not submit", () => {
  const h = harness();
  assert.equal(vm.runInContext("typeof fillQueryExample", h.context), "function");
  const expected = h.query.placeholder.replace(/^例如：/, "").trim();
  assert(expected.length > 0);
  assert.equal(h.key(), 1);
  assert.equal(h.query.value, expected);
  assert.equal(h.query.selectionStart, expected.length);
  assert.equal(h.query.selectionEnd, expected.length);
  assert.equal(h.calls.length, 0);
  assert.equal(vm.runInContext("state.busy", h.context), false);
  assert.equal(vm.runInContext("state.currentTurn", h.context), null);
  assert.equal(h.key(), 1, "a second Tab must not be intercepted once the query has content");
});

test("whitespace-only input is considered empty and the current placeholder supplies the example", () => {
  const h = harness({ value: " \n\t " });
  h.query.placeholder = "例如：Service 的类型是什么？";
  assert.equal(h.key(), 1);
  assert.equal(h.query.value, "Service 的类型是什么？");
  assert.equal(h.calls.length, 0);
});

test("existing text, other keys, modifiers and IME composition preserve native focus behavior", () => {
  assertUnchanged(harness({ value: "我已有自己的问题" }));
  assertUnchanged(harness({ value: "  DNS怎么查？  " }));
  for (const key of ["Enter", "Escape", "ArrowRight", " "]) assertUnchanged(harness(), { key });
  for (const modifier of ["shiftKey", "ctrlKey", "altKey", "metaKey", "isComposing"]) {
    assertUnchanged(harness(), { [modifier]: true });
  }
});

test("not-ready, busy, disabled and readonly textareas never consume Tab", () => {
  assertUnchanged(harness({ ready: false }));
  assertUnchanged(harness({ busy: true }));
  const disabled = harness(); disabled.query.disabled = true; assertUnchanged(disabled);
  const readonly = harness(); readonly.query.readOnly = true; assertUnchanged(readonly);
});

test("empty or prefix-only placeholders never trap keyboard navigation", () => {
  for (const placeholder of ["", "  ", "例如：", "例如：  "]) {
    const h = harness(); h.query.placeholder = placeholder; assertUnchanged(h);
  }
});

test("the keyboard hint is visible and connected to the textarea for assistive technology", () => {
  const textarea = page.match(/<textarea\b[^>]*\bid="query"[^>]*>/)[0];
  const descriptionIds = textarea.match(/\baria-describedby="([^"]+)"/)?.[1].trim().split(/\s+/);
  assert(descriptionIds?.length, "the textarea must describe its Tab shortcut");
  const descriptions = descriptionIds.map(id => {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const match = page.match(new RegExp('<([a-z][\\w-]*)\\b([^>]*\\bid="' + escaped + '"[^>]*)>([\\s\\S]*?)<\\/\\1>'));
    assert(match, "described-by IDs must refer to real markup");
    assert(!/\bhidden\b/.test(match[2]), "the keyboard hint should be visible");
    return match[3];
  }).join(" ");
  assert(/Tab/i.test(descriptions));
  assert(/示例/.test(descriptions));
});
