"use strict";

// 合成合同与轻量 DOM；不启动数据库、浏览器或模型服务。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/workflow_lab.html"), "utf8");
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const ORDER = ["retrieve", "fuse", "rerank", "prepare_evidence", "generate", "validate"];
const clone = value => JSON.parse(JSON.stringify(value));

function fixture() {
  const nodes = ORDER.map((id, index) => ({ id, label: `合成步骤 ${index + 1}`,
    reads: index ? ["query"] : ["query"], writes: [["bm25", "dense"], ["rrf_pool"], ["reranked"], ["evidence"], ["draft"], ["answer"]][index],
    operation: ["读取合成快照", "本地计算", "读取合成快照", "本地计算", "读取合成快照", "本地校验"][index] }));
  const common = { schema_version: "rag-workflow-replay/v1", langgraph_version: "1.2.16",
    fixture: { id: "dns-teaching-v1", title: "合成快照 <img onerror=alert(1)>", origin: "synthetic", query: "Pod 中 DNS 如何排查？", notice: "只用合成数据，模型请求 0。" },
    nodes, request_counts: { embedding: 0, http: 0, reranker: 0, chat: 0 }, notice: "教学回放，无模型调用。" };
  const updates = [{ bm25: ["chunk-a"], dense: ["chunk-b"] }, { rrf_pool: ["chunk-a", "chunk-b"] },
    { reranked: ["chunk-b", "chunk-a"] }, { evidence: { S1: "示例证据" } },
    { draft: { text: "合成回答 <script>throw 1</script>", source: "S1" } },
    { answer: { status: "answered", text: "合成回答 <script>throw 1</script>", source: "S1" } }];
  let shared = { query: common.fixture.query };
  const events = updates.map((update, index) => {
    const before = clone(shared); shared = { ...shared, ...clone(update) };
    return { step: index + 1, node: ORDER[index], before, update, after: clone(shared) };
  });
  const ids = ["__start__", ...ORDER, "__end__"];
  return { ...common, graph: { nodes: ids.map(id => ({ id, label: id })), edges: ids.slice(0, -1).map((id, index) => ({ source: id, target: ids[index + 1] })) },
    plain: { engine: "python", events: clone(events), final_state: clone(shared) },
    langgraph: { engine: "langgraph", events: clone(events), final_state: clone(shared) },
    comparison: { same_steps: true, same_updates: true, same_final_state: true, same_answer: true } };
}
function plan() { const data = fixture(); return { ...data, available: true }; }

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this._text = ""; this.className = "";
    this.dataset = {}; this.attributes = {}; this.listeners = {}; this.disabled = false; this.hidden = false; this.open = false;
    this.classList = { toggle: (name, enabled) => {
      const names = new Set(this.className.split(/\s+/).filter(Boolean));
      if (enabled) names.add(name); else names.delete(name); this.className = [...names].join(" ");
    } };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = children; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  querySelectorAll(selector) {
    const found = [];
    for (const child of this.children) {
      if (selector === "button" && child.tagName === "BUTTON" || selector === ".graph-node" && child.className.split(/\s+/).includes("graph-node")) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }
}
function harness({ planData = plan(), postData = fixture(), failPlan = false, failPost = false, deferPost = false } = {}) {
  const elements = new Map([...page.matchAll(/id="([^"]+)"/g)].map(match => [match[1], new Element()]));
  const calls = []; let finishPost;
  const fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === "/api/workflow/plan") return { ok: !failPlan, json: async () => clone(planData) };
    if (url === "/api/workflow/replay") {
      if (deferPost) await new Promise(resolve => { finishPost = resolve; });
      return { ok: !failPost, json: async () => clone(postData) };
    }
    throw new Error("unexpected network target");
  };
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: tag => new Element(tag) }, fetch });
  vm.runInContext(script, context);
  return { context, calls, elements, get: id => elements.get(id), eval: code => vm.runInContext(code, context), finish: () => finishPost() };
}
async function settle() { await new Promise(resolve => setImmediate(resolve)); }

test("page has no external assets or executable HTML injection sinks", () => {
  assert(!page.includes("innerHTML")); assert(!page.includes("insertAdjacentHTML"));
  assert(!page.match(/<script[^>]+src=/)); assert(!page.match(/<link[^>]+href="https?:/));
  assert(page.includes('href="/"')); assert(page.includes("不是实时检索"));
  assert(page.includes("逐步查看已经执行的轨迹")); assert(page.includes("没有实现持久化"));
});

test("initial load only GETs the plan and leaves replay behind a click", async () => {
  const h = harness(); await settle();
  assert.equal(h.calls.length, 1); assert.equal(h.calls[0].url, "/api/workflow/plan");
  assert.equal(h.calls[0].options.method, "GET"); assert.equal(h.get("run-button").disabled, false);
  assert(h.get("status").textContent.includes("点击按钮")); assert.equal(h.eval("state.data"), null);
});

test("missing pinned dependency disables replay and explains why", async () => {
  const data = plan(); data.available = false; delete data.langgraph_version; data.dependency_message = "请安装锁定依赖";
  const h = harness({ planData: data }); await settle(); await h.eval("runReplay()");
  assert.equal(h.get("run-button").disabled, true); assert.equal(h.calls.length, 1);
  assert.equal(h.get("error").textContent, data.dependency_message);
});

test("plan load failure makes no automatic retry", async () => {
  const h = harness({ failPlan: true }); await settle();
  assert.equal(h.get("run-button").disabled, true); assert.equal(h.calls.length, 1); assert.equal(h.get("error").hidden, false);
});

test("one click executes one same-origin guarded POST with an empty JSON object", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()");
  assert.equal(h.calls.length, 2); const call = h.calls[1]; assert.equal(call.url, "/api/workflow/replay");
  assert.equal(call.options.method, "POST"); assert.equal(call.options.mode, "same-origin");
  assert.equal(call.options.credentials, "same-origin"); assert.equal(call.options.headers["X-Retrieval-Lab"], "1");
  assert.equal(call.options.headers["Content-Type"], "application/json"); assert.equal(call.options.body, "{}");
  assert.equal(h.get("results").hidden, false); assert.equal(h.get("checks").children.length, 4);
});

test("two displayed routes expose identical before, update and after with code highlighting", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()");
  assert.equal(h.get("plain-event").children.length, 3); assert.equal(h.get("langgraph-event").children.length, 3);
  assert.equal(h.get("plain-event").textContent, h.get("langgraph-event").textContent);
  assert(h.get("plain-code").children.find(item => item.className.includes("active")).textContent.includes("retrieve(state)"));
  assert(h.get("graph-code").textContent.includes("builder.compile()")); assert(h.get("graph-code").textContent.includes("app.stream"));
});

test("steps and previous/next inspect memory only, not network or model calls", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()"); const count = h.calls.length;
  assert.equal(h.get("previous-button").disabled, true); assert.equal(h.get("next-button").disabled, false);
  h.eval("selectStep(5)"); assert.equal(h.get("step-count").textContent, "6 / 6");
  assert.equal(h.get("next-button").disabled, true); assert.equal(h.get("previous-button").disabled, false);
  h.get("previous-button").listeners.click(); assert.equal(h.get("step-count").textContent, "5 / 6");
  h.get("step-list").children[1].children[0].listeners.click(); assert.equal(h.get("step-count").textContent, "2 / 6");
  assert.equal(h.calls.length, count); assert(h.get("step-description").textContent.includes("rrf_pool"));
});

test("diagram uses returned node metadata and graph edges and selects the corresponding node", async () => {
  const data = fixture(); data.nodes.find(item => item.id === "fuse").label = "服务端返回的融合节点"; data.graph.edges.reverse();
  const h = harness({ postData: data }); await settle(); await h.eval("runReplay()");
  const boxes = h.get("diagram").querySelectorAll(".graph-node"); assert.equal(boxes.length, 8);
  const fuse = boxes.find(item => item.dataset.node === "fuse"); assert(fuse.textContent.includes("服务端返回的融合节点"));
  fuse.listeners.click(); assert.equal(h.get("step-count").textContent, "2 / 6"); assert(fuse.className.includes("selected"));
  assert(h.get("graph-edges").textContent.includes("__start__ → retrieve"));
});

test("source and answer strings stay literal text, even with hostile HTML", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()");
  assert(h.get("fixture-title").textContent.includes("<img onerror"));
  assert(h.get("answer").textContent.includes("<script>throw 1</script>"));
  assert.equal(h.get("answer").children.length, 0);
});

test("busy guard prevents duplicate clicks and navigation during server execution", async () => {
  const h = harness({ deferPost: true }); await settle(); const first = h.eval("runReplay()");
  assert.equal(h.get("run-button").disabled, true); await h.eval("runReplay()"); h.eval("selectStep(5)");
  assert.equal(h.calls.length, 2); assert.equal(h.get("previous-button").disabled, true);
  h.finish(); await first; assert.equal(h.get("run-button").disabled, false); assert.equal(h.get("step-count").textContent, "1 / 6");
});

test("failed run clears old replay and does not retry or fabricate model errors", async () => {
  const h = harness({ failPost: true }); await settle(); await h.eval("runReplay()");
  assert.equal(h.get("results").hidden, true); assert.equal(h.get("error").hidden, false);
  assert.equal(h.calls.length, 2); assert.equal(h.eval("state.data"), null);
  assert(h.get("status").textContent.includes("没有自动重试"));
});

test("stale response cannot overwrite the current UI", async () => {
  const h = harness({ deferPost: true }); await settle(); const first = h.eval("runReplay()");
  h.eval("state.sequence += 1"); h.finish(); await first;
  assert.equal(h.eval("state.data"), null); assert.equal(h.get("results").hidden, true);
});

test("rerun resets selected step and issues exactly one additional POST", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()"); h.eval("selectStep(5)"); await h.eval("runReplay()");
  assert.equal(h.calls.filter(call => call.url === "/api/workflow/replay").length, 2);
  assert.equal(h.get("step-count").textContent, "1 / 6"); assert.equal(h.get("error").hidden, true);
});

for (const [label, mutate] of [
  ["schema version", data => { data.schema_version = "future"; }],
  ["real data label", data => { data.fixture.origin = "live"; }],
  ["nonzero model counts", data => { data.request_counts.chat = 1; }],
  ["node order", data => { data.nodes.reverse(); }],
  ["graph branch", data => { data.graph.edges[1].target = "generate"; }],
  ["duplicate graph node", data => { data.graph.nodes[1].id = "__start__"; }],
  ["missing event", data => { data.langgraph.events.pop(); }],
  ["false comparison", data => { data.comparison.same_updates = false; }],
  ["forged equal verdict", data => { data.langgraph.final_state.answer.text = "不相同"; }],
  ["broken state merge", data => { data.plain.events[0].after.extra = "未由 update 写入"; }],
  ["broken continuity", data => { data.plain.events[1].before.query = "换了问题"; }],
  ["wrong initial query", data => { data.fixture.query = "其他问题"; }],
]) {
  test(`rejects ${label} without presenting a successful comparison`, async () => {
    const data = fixture(); mutate(data); const h = harness({ postData: data }); await settle(); await h.eval("runReplay()");
    assert.equal(h.get("results").hidden, true); assert.equal(h.get("error").hidden, false);
    assert.equal(h.eval("state.data"), null); assert.equal(h.calls.length, 2);
  });
}

test("object key order does not create a false mismatch", async () => {
  const data = fixture(); const event = data.langgraph.events[0]; event.update = { dense: event.update.dense, bm25: event.update.bm25 };
  const h = harness({ postData: data }); await settle(); await h.eval("runReplay()");
  assert.equal(h.get("results").hidden, false); assert.equal(h.get("error").hidden, true);
});

test("mobile layout and reduced desktop assumptions are declared", () => {
  assert(page.includes('@media(max-width:600px)')); assert(page.includes("grid-template-columns:1fr"));
  assert(page.includes("overflow-wrap:anywhere")); assert(page.includes('role="status"')); assert(page.includes('aria-pressed'));
});
