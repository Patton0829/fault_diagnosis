"use strict";

// Synthetic plan data only; the VM has no browser or real network access.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/qa.html"), "utf8");
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];

function plan(knowledge) {
  return { schema_version: "rag-live-plan/v1", default_model: "qwen3.8-flash",
    models: [{ id: "qwen3.8-flash", base_url: "https://example.invalid/v1", transport_notice: "HTTPS" }],
    reranker: { model: "test-rerank", url: "https://example.invalid/rerank" },
    pipeline: ["retrieve", "rerank", "answer"], notice: "合成配置，不调用模型。",
    ...(knowledge === undefined ? {} : { knowledge }) };
}
const knowledge = () => ({ index_name: "kubernetes_full_preflight_v1", document_count: 1920,
  parent_count: 755, dimension: 4096, notice: "资料尚未逐条审核。" });
class Element {
  constructor() { this.children = []; this.textContent = ""; this.hidden = false; this.value = ""; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener() {}
}
async function ready(initial) {
  const elements = new Map([...page.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)].map(match => {
    const element = new Element(); element.hidden = /\bhidden\b/.test(match[2]);
    return [match[3], element];
  }));
  const calls = [];
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: () => new Element() },
    fetch: async (url, options) => {
      calls.push({ url, options });
      assert.equal(url, "/api/qa/plan");
      return { ok: true, json: async () => JSON.parse(JSON.stringify(initial)) };
    } });
  vm.runInContext(script, context);
  await new Promise(resolve => setImmediate(resolve));
  return { calls, get: id => elements.get(id), eval: code => vm.runInContext(code, context) };
}

test("the optional knowledge summary is hidden for legacy plans", async () => {
  const h = await ready(plan());
  assert(h.get("knowledge-summary").hidden);
  assert.equal(h.get("knowledge-summary").textContent, "");
  assert.equal(h.get("ask").disabled, false);
  assert.equal(h.calls.length, 1);
});

test("knowledge summary shows the active scope from one existing plan request", async () => {
  const h = await ready(plan(knowledge()));
  assert.equal(h.get("knowledge-summary").hidden, false);
  assert.equal(h.get("knowledge-summary").textContent, "当前知识库：755篇 · 1920块 · 4096维。资料尚未逐条审核。");
  assert.equal(h.get("ask").disabled, false);
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].options.method, "GET");
});

test("the notice stays plain text and the label cannot carry a path", async () => {
  const info = knowledge(); info.notice = '<img src=x onerror="throw 1">尚未逐条审核';
  const h = await ready(plan(info));
  assert(h.get("knowledge-summary").textContent.includes(info.notice));
  assert.equal(h.get("knowledge-summary").children.length, 0);
  assert(!page.includes("innerHTML"));
  for (const index_name of ["/Users/private/index", "../index", "index.db", "a".repeat(81), "<script>", ""]) {
    const failed = await ready(plan({ ...knowledge(), index_name }));
    assert.equal(failed.get("ask").disabled, true, index_name);
    assert.equal(failed.calls.length, 1);
  }
});

test("knowledge uses exact fields and positive integer scope values", async () => {
  const malformed = [null, [], {}, { ...knowledge(), secret: "not allowed" }, { ...knowledge(), notice: "" },
    { ...knowledge(), notice: " " }, { ...knowledge(), notice: "x".repeat(401) },
    { ...knowledge(), notice: null }];
  for (const key of ["document_count", "parent_count", "dimension"]) {
    for (const value of [0, -1, 0.5, "1920", null, true, Number.MAX_SAFE_INTEGER + 1]) malformed.push({ ...knowledge(), [key]: value });
    const missing = knowledge(); delete missing[key]; malformed.push(missing);
  }
  for (const info of malformed) {
    const h = await ready(plan(info));
    assert.equal(h.get("ask").disabled, true, JSON.stringify(info));
    assert.equal(h.get("knowledge-summary").hidden, true);
    assert.equal(h.calls.length, 1);
  }
});
