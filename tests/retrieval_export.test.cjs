"use strict";

// All responses are synthetic. No browser, server, corpus, model, or packages.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const helperPath = path.join(__dirname, "../web/retrieval_export.js");
const pagePath = path.join(__dirname, "../web/retrieval_lab.html");
const helper = require(helperPath);
const TIMES = {
  receivedAt: "2026-10-18T09:10:11.012+08:00",
  exportedAt: "2026-10-18T09:12:13.456+08:00",
};

function fixture(mode = "hybrid") {
  const records = {};
  for (let index = 1; index <= 10; index++) {
    const id = `chunk-${index}`;
    records[id] = {
      chunk_id: id,
      text: `${id} 完整原文：引号"与'、竖线|、反斜杠\\、emoji🧪\n\`\`\`yaml\ncommand: "$(do-not-run)"\n\`\`\`\n${"尾段不可截断。".repeat(170)}END-${index}`,
      metadata: {
        chunk_id: id,
        title: `章节${index} | <img src=x onerror=alert(1)>`,
        sections: [["父章节|", `子章节${index}\n换行`]],
        source: "/Users/synthetic/private-source.md",
        source_url: "https://example.invalid/synthetic?x=1&y=2",
        snapshot_commit: "a".repeat(40),
        review_status: "needs_review",
        arbitrary: { unicode: "中文\u0000原样", values: [false, null, 0, 1.2345678901234567] },
      },
    };
  }
  const bm25 = Array.from({ length: 8 }, (_, index) => ({
    chunk_id: `chunk-${index + 1}`, rank: index + 1,
    score: index < 2 ? -0.12345678901234567 : -index,
    matched_terms: ["dns", "引号\"|"],
  }));
  const dense = Array.from({ length: 8 }, (_, index) => ({
    chunk_id: `chunk-${index + 3}`, rank: index + 1,
    score: index < 2 ? 0.9876543210123456 : 0.8 - index / 100,
  }));
  const rrf = [3, 4, 5, 6, 7].map((index, position) => {
    const ranks = { bm25: index, dense: index - 2 };
    const contributions = { bm25: 1 / (60 + ranks.bm25), dense: 1 / (60 + ranks.dense) };
    return { chunk_id: `chunk-${index}`, rank: position + 1,
      ranks, contributions, score: contributions.bm25 + contributions.dense };
  });
  return {
    query: 'DNS "如何排查" | <script>throw 1</script>\n$(do-not-run) 🧪',
    mode, scope_count: 42, recall_k: 8, top_k: 5, rrf_constant: 60,
    tokens: ["dns", "如何排查", "a|b"],
    routes: { bm25, dense: mode === "hybrid" ? dense : [], rrf: mode === "hybrid" ? rrf : [] },
    records,
    request_counts: { embedding: mode === "hybrid" ? 1 : 0, http: mode === "hybrid" ? 3 : 0, reranker: 0, chat: 0 },
    notice: "合成响应；不是概率或准确率。",
    extra_response_field: { nested: ["完整服务响应", { retained: true }] },
  };
}

function deepFreeze(value) {
  if (value && typeof value === "object") {
    Object.freeze(value);
    for (const child of Object.values(value)) deepFreeze(child);
  }
  return value;
}

test("report is a complete independent JSON snapshot, including hidden ranks and full metadata", () => {
  const original = deepFreeze(fixture());
  const report = helper.buildReport(original, { ...TIMES, selectedChunkId: "chunk-8" });
  assert.equal(report.schema_version, "retrieval-lab-report/v1");
  assert.equal(report.browser_received_at, "2026-10-18T01:10:11.012Z");
  assert.equal(report.exported_at, "2026-10-18T01:12:13.456Z");
  assert.equal(report.selected_chunk_id, "chunk-8");
  assert.deepEqual(report.result, original);
  assert.notEqual(report.result, original);
  assert.notEqual(report.result.records["chunk-8"].metadata, original.records["chunk-8"].metadata);
  const decoded = JSON.parse(JSON.stringify(report));
  assert.deepEqual(decoded.result, original);
  assert.equal(decoded.result.routes.bm25[7].score, original.routes.bm25[7].score);
  assert.equal(decoded.result.routes.dense[0].score, original.routes.dense[0].score);
  assert.ok(decoded.result.records["chunk-8"].text.endsWith("END-8"));
  assert.equal(decoded.result.records["chunk-8"].metadata.source, "/Users/synthetic/private-source.md");
  report.result.records["chunk-8"].metadata.arbitrary.values.push("changed snapshot");
  assert.equal(original.records["chunk-8"].metadata.arbitrary.values.length, 4);
});

test("Chinese summary contains the complete route union, original ranks, RRF contributions, counts, provenance, and boundaries", () => {
  const result = deepFreeze(fixture());
  const before = JSON.stringify(result);
  const summary = helper.summaryText(helper.buildReport(result, { ...TIMES, selectedChunkId: "chunk-8" }));
  assert.match(summary, /问题：DNS/);
  assert.match(summary, /模式：BM25 \+ Dense \+ RRF/);
  assert.match(summary, /范围42块.*8块.*5块.*60/);
  assert.match(summary, /Embedding 1；HTTP 3；Reranker 0；聊天 0/);
  assert.match(summary, /当前选中块：chunk-8/);
  assert.match(summary, /BM25 8；Dense 8；RRF 5；去重并集10块/);
  const rows = summary.split("\n").filter(line => line.startsWith("| chunk-"));
  assert.equal(rows.length, 10);
  assert.deepEqual(rows.map(line => line.split(" | ")[0]), [3, 4, 5, 6, 7, 1, 2, 8, 9, 10].map(index => `| chunk-${index}`));
  for (const index of [6, 7, 8]) {
    const row = rows.find(line => line.startsWith(`| chunk-${index} |`));
    assert.ok(row, `folded rank ${index} must be in the summary`);
    assert.ok(row.includes(`#${index} / -${index - 1}.000000`));
  }
  assert.match(summary, /#1 \/ -0\.123457/);
  assert.match(summary, /#2 \/ -0\.123457/); // ties keep the server's rank
  assert.match(summary, /bm25 #3：1\/\(60\+3\)=0\.01587302/);
  assert.match(summary, /dense #1：1\/\(60\+1\)=0\.01639344/);
  assert.match(summary, /https:\/\/example\.invalid\/synthetic\?x=1&amp;y=2/);
  assert.match(summary, new RegExp("a".repeat(40)));
  assert.match(summary, /needs_review/);
  assert.match(summary, /不是概率或准确率/);
  assert.match(summary, /同分次序不代表内容优劣/);
  assert.match(summary, /不重新检索或请求模型/);
  assert.ok(summary.includes("\\|"));
  assert.ok(summary.includes("&lt;script&gt;"));
  assert.ok(!summary.includes("<script>"));
  assert.ok(!summary.includes("/Users/synthetic/private-source.md"));
  assert.ok(!summary.includes("END-8")); // full bodies belong to JSON
  assert.equal(JSON.stringify(result), before);
});

test("empty BM25 hybrid and BM25-only summaries retain explicit route semantics", () => {
  const emptyLexical = fixture();
  emptyLexical.routes.bm25 = [];
  const denseSummary = helper.summaryText(helper.buildReport(emptyLexical, TIMES));
  assert.match(denseSummary, /BM25 0；Dense 8；RRF 5/);
  assert.match(denseSummary, /未入召回前8/);
  const lexical = fixture("bm25");
  const summary = helper.summaryText(helper.buildReport(lexical, TIMES));
  assert.match(summary, /仅BM25（Dense\/RRF未运行）/);
  assert.match(summary, /Embedding 0；HTTP 0；Reranker 0；聊天 0/);
  assert.ok(summary.split("\n").filter(line => line.startsWith("| chunk-")).every(line => line.endsWith("| 未运行 | 未运行 |")));
  lexical.routes.bm25 = [];
  lexical.records = {};
  assert.match(helper.summaryText(helper.buildReport(lexical, TIMES)), /\| 无候选 \|/);
});

test("download filename is generated exclusively from the canonical timestamp", () => {
  const report = helper.buildReport(fixture(), TIMES);
  assert.equal(helper.downloadName(report), "k8s-retrieval-20261018T011213456Z.json");
  report.result.query = '../../evil/"\n<script>|？';
  assert.equal(helper.downloadName(report), "k8s-retrieval-20261018T011213456Z.json");
  assert.match(helper.downloadName(report), /^k8s-retrieval-[0-9]{8}T[0-9]{9}Z\.json$/);
});

test("incomplete results, invalid time/selection, and lossy non-JSON values are rejected", () => {
  const invalid = [
    data => { data.routes.bm25[0].score = NaN; },
    data => { data.routes.dense[0].score = Infinity; },
    data => { data.records["chunk-8"].metadata.nested = { score: -Infinity }; },
    data => { data.records["chunk-8"].metadata.nested = undefined; },
    data => { data.records["chunk-8"].metadata.nested = () => 1; },
    data => { data.records["chunk-8"].metadata.nested = Symbol("lost"); },
    data => { data.records["chunk-8"].metadata.nested = 1n; },
    data => { data.routes.bm25[1].rank = 1; },
    data => { data.routes.dense[1].chunk_id = data.routes.dense[0].chunk_id; },
    data => { delete data.records["chunk-8"]; },
    data => { delete data.request_counts; },
    data => { data.mode = "unsupported"; },
  ];
  for (const mutate of invalid) {
    const data = fixture();
    mutate(data);
    assert.throws(() => helper.buildReport(data, TIMES), undefined, mutate.toString());
  }
  assert.throws(() => helper.buildReport(fixture(), { ...TIMES, selectedChunkId: "previous-result" }));
  assert.throws(() => helper.buildReport(fixture(), { ...TIMES, receivedAt: "invalid" }));
  assert.throws(() => helper.buildReport(fixture(), { ...TIMES, exportedAt: "invalid" }));
  assert.throws(() => helper.summaryText({ schema_version: "unknown", result: fixture() }));
});

test("browser UMD helper executes without HTTP or other external APIs", () => {
  let networkCalls = 0;
  const deny = () => { networkCalls++; throw new Error("Unexpected network access"); };
  const sandbox = { fetch: deny, XMLHttpRequest: deny, WebSocket: deny };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(helperPath, "utf8"), sandbox, { filename: helperPath });
  const report = sandbox.RetrievalExport.buildReport(fixture(), TIMES);
  assert.equal(sandbox.RetrievalExport.downloadName(report), helper.downloadName(report));
  assert.equal(sandbox.RetrievalExport.summaryText(report), helper.summaryText(report));
  assert.equal(networkCalls, 0);
});

class FakeNode {
  constructor(tagName = "div") {
    Object.assign(this, { tagName, children: [], dataset: {}, attributes: {}, listeners: {}, value: "", textContent: "", hidden: false, disabled: false, open: false });
    const classes = new Set();
    this.classList = { toggle: (name, on) => on ? classes.add(name) : classes.delete(name) };
  }
  append(...nodes) { for (const item of nodes) { item.parent = this; this.children.push(item); } }
  replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  focus() { this.focused = true; }
  select() { this.selected = true; }
  click() { this.clicked = true; this.listeners.click?.(); }
  remove() { this.removed = true; if (this.parent) this.parent.children = this.parent.children.filter(child => child !== this); }
  querySelector(selector) { return this.children.find(child => selector === "button" && child.tagName === "button") || null; }
}

function uiHarness() {
  const elements = new Map();
  const created = [];
  const body = new FakeNode("body");
  const events = { fetch: [], blobs: [], revoked: [], timers: [], clipboard: [] };
  const document = {
    body,
    getElementById(id) { if (!elements.has(id)) elements.set(id, new FakeNode()); return elements.get(id); },
    createElement(tag) { const element = new FakeNode(tag); created.push(element); return element; },
    querySelectorAll() { return []; },
  };
  const sandbox = {
    document, Blob,
    navigator: { clipboard: { writeText: async text => { events.clipboard.push(text); } } },
    RetrievalExport: helper,
    fetch: (...args) => { events.fetch.push(args); throw new Error("Export must not fetch"); },
    URL: class extends URL {
      static createObjectURL(blob) { events.blobs.push(blob); return `blob:synthetic-${events.blobs.length}`; }
      static revokeObjectURL(url) { events.revoked.push(url); }
    },
    setTimeout(callback, milliseconds) { events.timers.push({ callback, milliseconds }); return events.timers.length; },
  };
  sandbox.window = sandbox;
  const inline = [...fs.readFileSync(pagePath, "utf8").matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(match => match[1]).find(source => source.includes("function copySummary"));
  assert.ok(inline, "page must contain the export handlers");
  assert.match(inline, /\ninit\(\);\s*$/);
  vm.createContext(sandbox);
  // Test the production handlers; only the unrelated automatic plan GET is omitted.
  vm.runInContext(inline.replace(/\ninit\(\);\s*$/, "") + "\nglobalThis.ui={state,renderResults,clearResults,setBusy,hasExportResult,updateExportButtons,copySummary,downloadJson,compare};", sandbox, { filename: pagePath });
  const element = id => document.getElementById(id);
  function ready(data = fixture()) {
    sandbox.ui.state.plan = { document_count: 42 };
    element("query").value = data.query;
    sandbox.ui.renderResults(data);
    return data;
  }
  return { sandbox, ui: sandbox.ui, element, events, created, ready };
}

test("UI buttons follow completed/current/busy/cleared states and clear old notices on edits", async () => {
  const harness = uiHarness();
  const { ui, element, ready, events } = harness;
  ui.updateExportButtons();
  assert.equal(element("copy-result").disabled, true);
  assert.equal(element("export-json").disabled, true);
  ready();
  assert.equal(element("copy-result").disabled, false);
  assert.equal(element("export-json").disabled, false);
  ui.setBusy(true);
  assert.equal(element("copy-result").disabled, true);
  assert.equal(element("export-json").disabled, true);
  await ui.copySummary();
  ui.downloadJson();
  assert.equal(events.clipboard.length, 0);
  assert.equal(events.blobs.length, 0);
  ui.setBusy(false);
  element("export-fallback").hidden = false;
  element("export-fallback-text").value = "old summary";
  element("export-status").textContent = "实验摘要已复制";
  element("query").value = "different question";
  element("query").listeners.input();
  assert.equal(ui.state.data, null);
  assert.equal(element("copy-result").disabled, true);
  assert.equal(element("export-json").disabled, true);
  assert.equal(element("export-fallback").hidden, true);
  assert.equal(element("export-fallback-text").value, "");
  assert.ok(!element("export-status").textContent.includes("已复制"));
  assert.equal(events.fetch.length, 0);
});

test("clipboard success and failure use the completed response without fetch or rerun", async () => {
  const harness = uiHarness();
  const { ui, sandbox, element, events, ready } = harness;
  ready();
  await ui.copySummary();
  assert.match(element("export-status").textContent, /已复制/);
  assert.ok(events.clipboard[0].includes("| chunk-8 |"));
  assert.equal(element("export-fallback").hidden, true);
  sandbox.navigator.clipboard.writeText = async () => { throw new Error("permission denied"); };
  await ui.copySummary();
  assert.equal(element("export-fallback").hidden, false);
  assert.match(element("export-fallback-text").value, /完整候选排名/);
  assert.equal(element("export-fallback-text").focused, true);
  assert.equal(element("export-fallback-text").selected, true);
  assert.match(element("export-status").textContent, /手动复制/);
  assert.equal(element("copy-result").disabled, false);
  assert.equal(events.fetch.length, 0);
  const html = fs.readFileSync(pagePath, "utf8");
  assert.match(html, /<textarea[^>]*id="export-fallback-text"[^>]*readonly/);
});

for (const resolves of [true, false]) {
  test(`stale clipboard ${resolves ? "success" : "failure"} does not overwrite a subsequent result`, async () => {
    const { ui, sandbox, element, events, ready } = uiHarness();
    ready();
    let resolveCopy, rejectCopy;
    sandbox.navigator.clipboard.writeText = () => new Promise((resolve, reject) => { resolveCopy = resolve; rejectCopy = reject; });
    const pending = ui.copySummary();
    assert.equal(element("copy-result").disabled, true);
    ui.state.sequence++;
    ui.clearResults();
    const next = fixture("bm25"); next.query = "新一轮问题";
    ready(next);
    const newStatus = element("export-status").textContent;
    if (resolves) resolveCopy(); else rejectCopy(new Error("old failure"));
    await pending;
    assert.equal(ui.state.data, next);
    assert.equal(element("export-status").textContent, newStatus);
    assert.equal(element("export-fallback").hidden, true);
    assert.equal(element("export-fallback-text").value, "");
    assert.equal(element("copy-result").disabled, false);
    assert.equal(events.fetch.length, 0);
  });
}

test("download Blob keeps complete records, timestamp filename, link removal and URL cleanup without fetch", async () => {
  const { ui, element, events, created, ready } = uiHarness();
  const original = ready();
  ui.state.selected = "chunk-8";
  ui.downloadJson();
  assert.equal(events.blobs.length, 1);
  assert.match(events.blobs[0].type, /^application\/json/);
  const report = JSON.parse(await events.blobs[0].text());
  assert.deepEqual(report.result, original);
  assert.equal(report.selected_chunk_id, "chunk-8");
  assert.ok(report.result.records["chunk-8"].text.endsWith("END-8"));
  const link = created.find(item => item.tagName === "a" && item.download);
  assert.ok(link.clicked);
  assert.ok(link.removed);
  assert.match(link.download, /^k8s-retrieval-[0-9]{8}T[0-9]{9}Z\.json$/);
  assert.equal(link.href, "blob:synthetic-1");
  assert.equal(events.timers.length, 1);
  events.timers[0].callback();
  assert.deepEqual(events.revoked, ["blob:synthetic-1"]);
  assert.match(element("export-status").textContent, /完整JSON下载/);
  assert.equal(events.fetch.length, 0);
});

test("failed next retrieval clears export eligibility and old manual copy content", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  ready();
  element("export-fallback").hidden = false;
  element("export-fallback-text").value = "stale manual copy";
  sandbox.fetch = async (...args) => { events.fetch.push(args); return { ok: false, json: async () => ({ detail: "synthetic failure" }) }; };
  await ui.compare("hybrid");
  assert.equal(events.fetch.length, 1);
  assert.equal(ui.state.data, null);
  assert.equal(element("copy-result").disabled, true);
  assert.equal(element("export-json").disabled, true);
  assert.equal(element("export-fallback").hidden, true);
  assert.equal(element("export-fallback-text").value, "");
});
