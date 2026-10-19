"use strict";

// All inputs are synthetic contracts. Fake fetch and a small DOM keep these tests offline.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/approval_lab.html"), "utf8");
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const PREFIX = ["retrieve", "fuse", "rerank", "prepare_evidence"];
const NODES = [...PREFIX, "review", "generate", "cancel", "validate"];
const SOURCES = ["state", "review", "router", "graph_builder", "start", "resume"];
const SESSION = "a".repeat(32);
const clone = value => JSON.parse(JSON.stringify(value));

function definitions() {
  const make = (name, code) => ({ name, code, file: "src/qa_platform/workflows/approval.py",
    start_line: 10, end_line: 10 + code.split("\n").length - 1 });
  return {
    state: make("ApprovalState", "class ApprovalState(ReplayState, total=False):\n    approved: bool\n    cancelled: bool"),
    review: make("review_evidence", "def review_evidence(state):\n    approved = interrupt({\"kind\": \"review_evidence\"})\n    return {\"approved\": approved}"),
    router: make("route_after_review", "def route_after_review(state):\n    return \"generate\" if state[\"approved\"] else \"cancel\""),
    graph_builder: make("build_graph", "def build_graph(nodes, checkpointer):\n    builder.add_conditional_edges(\"review\", route_after_review)\n    return builder.compile(checkpointer=checkpointer)"),
    start: make("start", "def start(self):\n    graph = build_graph(nodes, InMemorySaver())\n    _drive(session, {\"query\": fixture[\"query\"]})"),
    resume: make("resume", "def resume(self, session_id, approved):\n    _drive(session, Command(resume=approved))"),
  };
}

function fixture(phase = "idle", session = SESSION) {
  const query = "Pod 中 DNS 如何排查？";
  const evidence = {
    S1: { chunk_id: "demo-namespace", text: "合成命名空间原文 <img onerror=alert(1)>", quotes: { Q1: "合成命名空间原文 <img onerror=alert(1)>" } },
    S2: { chunk_id: "demo-dns-config", text: "合成 DNS 配置原文", quotes: { Q1: "合成 DNS 配置原文" } },
  };
  const candidates = [{ chunk_id: "demo-namespace", score: 1 }, { chunk_id: "demo-dns-config", score: 0.5 }];
  const draft = { status: "partial", claims: [{ text: "教学草稿 <script>throw 1</script>", evidence: [{ source_id: "S1", quote_id: "Q1" }] }], missing_info: ["缺少现场信息"] };
  const graphEdges = [["__start__", "retrieve"], ["retrieve", "fuse"], ["fuse", "rerank"], ["rerank", "prepare_evidence"],
    ["prepare_evidence", "review"], ["review", "generate"], ["review", "cancel"], ["generate", "validate"], ["validate", "__end__"], ["cancel", "__end__"]];
  const counts = Object.fromEntries(NODES.map(id => [id, { entered: 0, completed: 0 }]));
  const data = { schema_version: "rag-workflow-approval/v1", available: true, langgraph_version: "1.2.16",
    fixture: { id: "dns-teaching-v1", title: "合成教学", origin: "synthetic", query, notice: "非现场规程。" },
    nodes: NODES.map(id => ({ id, label: id, reads: ["evidence"], writes: id === "review" ? ["approved"] : [], operation: "合成回放" })),
    definitions: definitions(), request_counts: { embedding: 0, http: 0, reranker: 0, chat: 0 }, notice: "真实暂停；仅内存检查点；零模型调用。",
    phase, session_id: null, graph: null, values: {}, events: [], run_counts: counts, checkpoint: null, interrupt: null };
  if (phase === "idle") return data;
  data.session_id = session;
  data.graph = { nodes: ["__start__", ...NODES, "__end__"].map(id => ({ id, label: id })),
    edges: graphEdges.map(([source, target]) => ({ source, target, conditional: source === "review" })) };
  let values = { query };
  const add = (node, update) => {
    const before = clone(values); values = { ...values, ...clone(update) };
    data.events.push({ step: data.events.length + 1, node, before, update: clone(update), after: clone(values) });
    counts[node] = { entered: 1, completed: 1 };
  };
  add("retrieve", { bm25: candidates, dense: candidates });
  add("fuse", { rrf_pool: candidates }); add("rerank", { reranked: candidates }); add("prepare_evidence", { evidence });
  counts.review = { entered: 1, completed: 0 };
  data.checkpoint = { thread_id: session, checkpoint_id: "checkpoint-synthetic", next: ["review"] };
  data.interrupt = { id: "interrupt-synthetic", value: { kind: "review_evidence", question: "是否继续教学回放？", evidence_count: 2, source_ids: ["S1", "S2"] } };
  if (phase !== "paused") {
    add("review", { approved: phase === "completed" }); counts.review.entered = 2;
    if (phase === "completed") { add("generate", { draft }); add("validate", { answer: clone(draft) }); }
    else add("cancel", { cancelled: true });
    data.checkpoint.next = []; data.interrupt = null;
  }
  data.values = clone(values);
  return data;
}

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this._text = ""; this.className = "";
    this.dataset = {}; this.attributes = {}; this.listeners = {}; this.disabled = false; this.hidden = false;
    this.open = false; this.value = ""; this.clientWidth = 400;
    this.classList = { toggle: (name, enabled) => { const names = new Set(this.className.split(/\s+/).filter(Boolean));
      if (enabled) names.add(name); else names.delete(name); this.className = [...names].join(" "); },
    add: name => { this.className += ` ${name}`; }, remove: name => { this.className = this.className.split(/\s+/).filter(item => item !== name).join(" "); } };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this._text = ""; this.children = children; }
  setAttribute(key, value) { this.attributes[key] = String(value); if (key === "class") this.className = String(value);
    if (key.startsWith("data-")) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value); }
  getAttribute(key) { return this.attributes[key] ?? null; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  querySelectorAll(selector) { const found = []; for (const child of this.children) {
    if (selector.startsWith(".") ? child.className.split(/\s+/).includes(selector.slice(1)) : child.tagName.toLowerCase() === selector.toLowerCase()) found.push(child);
    found.push(...child.querySelectorAll(selector)); } return found; }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

function harness({ initial = fixture(), startData = fixture("paused"), resumeData = null,
  failGet = false, failStart = false, failResume = false, deferGet = false, deferPost = false } = {}) {
  const elements = new Map([...page.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)].map(match => {
    const item = new Element(match[1]); item.hidden = /\bhidden\b/.test(match[2]); item.disabled = /\bdisabled\b/.test(match[2]);
    return [match[3], item]; }));
  elements.get("source-select").value = "state";
  const calls = [], pending = [];
  const fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === "/api/workflow/approval/state") {
      if (deferGet) await new Promise(resolve => pending.push(resolve));
      return { ok: !failGet, status: failGet ? 500 : 200, json: async () => clone(initial) };
    }
    if (deferPost) await new Promise(resolve => pending.push(resolve));
    if (url === "/api/workflow/approval/start") return { ok: !failStart, status: failStart ? 500 : 200,
      json: async () => clone(failStart ? { error: "合成开始失败" } : startData) };
    if (url === "/api/workflow/approval/resume") {
      const approved = JSON.parse(options.body).approved;
      return { ok: !failResume, status: failResume ? 409 : 200,
        json: async () => clone(failResume ? { error: "当前会话不可恢复" } : resumeData || fixture(approved ? "completed" : "cancelled", JSON.parse(options.body).session_id)) };
    }
    throw new Error(`unexpected network target ${url}`);
  };
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: tag => new Element(tag),
    createElementNS: (namespace, tag) => { const item = new Element(tag); item.namespaceURI = namespace; return item; } }, fetch });
  vm.runInContext(script, context);
  return { context, calls, elements, get: id => elements.get(id), eval: code => vm.runInContext(code, context),
    finish: (index = 0) => pending.splice(index, 1)[0]() };
}
async function settle() { await new Promise(resolve => setImmediate(resolve)); }
async function paused(options = {}) { const h = harness(options); await settle(); await h.eval("startRun()"); return h; }

test("lesson is independent, compact and has no external scripts or HTML execution sinks", () => {
  assert(!page.includes("innerHTML")); assert(!page.includes("insertAdjacentHTML"));
  assert(!page.match(/<script[^>]+src=/)); assert(!page.match(/<link[^>]+href="https?:/));
  assert(page.includes('href="/workflow/branching"')); assert(page.includes('role="status"'));
  const details = [...page.matchAll(/<details\b([^>]*)>/g)]; assert(details.length >= 1);
  assert(details.every(match => !/\bopen\b/.test(match[1]))); assert(!page.includes('id="plain-update"'));
});

test("initial load uses only read-only state GET, never auto-starts or resumes", async () => {
  const h = harness(); await settle(); assert.equal(h.calls.length, 1);
  const call = h.calls[0]; assert.equal(call.url, "/api/workflow/approval/state"); assert.equal(call.options.method, "GET");
  assert.equal(h.get("start-button").disabled, false); assert.equal(h.get("approve-button").disabled, true);
  assert.equal(h.get("cancel-button").disabled, true); assert.equal(h.eval("state.data.phase"), "idle");
});

test("start sends one guarded same-origin POST with no client state or model configuration", async () => {
  const h = await paused(); assert.equal(h.calls.length, 2); const call = h.calls[1];
  assert.equal(call.url, "/api/workflow/approval/start"); assert.deepEqual(JSON.parse(call.options.body), {});
  assert.equal(call.options.method, "POST"); assert.equal(call.options.mode, "same-origin");
  assert.equal(call.options.credentials, "same-origin"); assert.equal(call.options.headers["X-Retrieval-Lab"], "1");
  assert.equal(call.options.headers["Content-Type"], "application/json");
});

test("paused view shows four committed nodes, an entered unfinished review and no draft or answer", async () => {
  const h = await paused(); const data = JSON.parse(h.get("full-json").textContent);
  assert.equal(data.phase, "paused"); assert.deepEqual(data.events.map(item => item.node), PREFIX);
  assert.deepEqual(data.run_counts.review, { entered: 1, completed: 0 });
  assert(!Object.hasOwn(data.values, "approved")); assert(!Object.hasOwn(data.values, "draft")); assert(!Object.hasOwn(data.values, "answer"));
  assert(h.get("phase-title").textContent.includes("暂停")); assert.equal(h.get("approve-button").disabled, false);
  assert.equal(h.get("cancel-button").disabled, false); assert(h.get("evidence-list").textContent.includes("demo-namespace") || h.get("evidence-list").textContent.includes("S1"));
});

for (const approved of [true, false]) {
  test(`${approved ? "approval" : "cancellation"} resumes only the current session with a strict boolean`, async () => {
    const h = await paused(); await h.eval(`resumeRun(${approved})`); const call = h.calls[2];
    assert.equal(h.calls.length, 3); assert.equal(call.url, "/api/workflow/approval/resume");
    assert.deepEqual(JSON.parse(call.options.body), { session_id: SESSION, approved });
    assert.equal(call.options.headers["X-Retrieval-Lab"], "1"); assert.equal(call.options.mode, "same-origin");
    assert.equal(call.options.credentials, "same-origin");
    assert.equal(h.eval("state.data.phase"), approved ? "completed" : "cancelled");
    assert.equal(h.get("approve-button").disabled, true); assert.equal(h.get("cancel-button").disabled, true);
    const data = JSON.parse(h.get("full-json").textContent);
    for (const id of PREFIX) assert.deepEqual(data.run_counts[id], { entered: 1, completed: 1 });
    assert.deepEqual(data.run_counts.review, { entered: 2, completed: 1 });
    assert.equal(data.interrupt, null); assert.deepEqual(data.checkpoint.next, []);
    assert.equal(data.events.length, approved ? 7 : 6);
    assert.equal(Object.hasOwn(data.values, "answer"), approved); assert.equal(Object.hasOwn(data.values, "draft"), approved);
    if (!approved) { assert.equal(data.values.cancelled, true); assert(h.get("phase-title").textContent.includes("取消")); }
    await h.eval("resumeRun(true)"); await h.eval("resumeRun(false)"); assert.equal(h.calls.length, 3);
  });
}

for (const phase of ["paused", "completed", "cancelled"]) {
  test(`refresh displays server ${phase} checkpoint without replaying any node or decision`, async () => {
    const h = harness({ initial: fixture(phase) }); await settle(); assert.equal(h.calls.length, 1);
    assert.equal(h.eval("state.data.phase"), phase); assert.equal(h.get("approve-button").disabled, phase !== "paused");
    assert.equal(h.get("cancel-button").disabled, phase !== "paused");
    assert.deepEqual(JSON.parse(h.get("full-json").textContent).run_counts, fixture(phase).run_counts);
  });
}

test("double start and resume clicks while pending do not duplicate requests or decisions", async () => {
  const h = harness({ deferPost: true }); await settle(); const start = h.eval("startRun()");
  assert.equal(h.get("start-button").disabled, true); await h.eval("startRun()"); await h.eval("resumeRun(true)"); assert.equal(h.calls.length, 2);
  h.finish(); await start; const resume = h.eval("resumeRun(true)");
  assert.equal(h.get("approve-button").disabled, true); assert.equal(h.get("cancel-button").disabled, true);
  assert.equal(h.get("start-button").disabled, true); await h.eval("resumeRun(false)"); await h.eval("startRun()"); assert.equal(h.calls.length, 3);
  h.finish(); await resume; assert.equal(h.eval("state.data.phase"), "completed");
});

test("new start clears the old completed state while pending and creates a different session", async () => {
  const next = fixture("paused", "b".repeat(32)); const h = harness({ initial: fixture("completed"), startData: next, deferPost: true }); await settle();
  const start = h.eval("startRun()"); assert(!h.get("full-json").textContent.includes("教学草稿"));
  assert.equal(h.get("approve-button").disabled, true); h.finish(); await start;
  assert.equal(h.eval("state.data.session_id"), "b".repeat(32)); assert(!h.get("full-json").textContent.includes('"answer"'));
  const cancel = h.eval("resumeRun(false)"); h.finish(); await cancel;
});

test("source reading has six explicit construction stages and never changes state, counts or HTTP", async () => {
  const h = await paused(); const calls = h.calls.length; const saved = h.get("full-json").textContent;
  assert.equal(h.get("source-select").value, "state"); assert.equal(h.get("source-prev").disabled, true);
  const options = [...page.matchAll(/<option value="([^"]+)">/g)].map(match => match[1]).filter(key => SOURCES.includes(key));
  assert.deepEqual(options, SOURCES);
  for (const key of SOURCES) {
    assert.equal(h.get("source-select").value, key); assert(h.get("source-code").textContent.includes(definitions()[key].name));
    assert(h.get("source-goal").textContent.length > 10); h.get("source-next").listeners.click();
  }
  assert.equal(h.get("source-next").disabled, true); assert.equal(h.get("full-json").textContent, saved); assert.equal(h.calls.length, calls);
  for (const key of SOURCES.slice(0, -1).reverse()) { h.get("source-prev").listeners.click(); assert.equal(h.get("source-select").value, key); }
});

test("source and evidence hostile strings remain literal text, not DOM elements", async () => {
  const initial = fixture(); initial.definitions.review.code += '\n    # <svg onload="throw 1">'; initial.definitions.review.end_line += 1;
  const h = await paused({ initial }); h.get("source-select").value = "review"; h.get("source-select").listeners.change();
  // The POST contract also supplies definitions, so use its current record for the XSS probe.
  const data = fixture("paused"); data.definitions.review.code += '\n    # <svg onload="throw 1">'; data.definitions.review.end_line += 1;
  const literal = harness({ initial: data }); await settle(); literal.get("source-select").value = "review"; literal.get("source-select").listeners.change();
  assert(literal.get("source-code").textContent.includes('<svg onload="throw 1">'));
  assert.equal(literal.get("source-code").querySelectorAll("svg").length, 0);
  assert(h.get("evidence-list").textContent.includes("<img onerror=alert(1)>"));
  assert.equal(h.get("evidence-list").querySelectorAll("img").length, 0); assert.equal(h.calls.length, 2);
});

test("actual graph draws all ten nodes and ten edges with no out-of-view narrow boxes", async () => {
  const data = fixture("paused"); data.graph.edges.reverse(); const h = await paused({ startData: data });
  const diagram = h.get("diagram"), svg = diagram.querySelector("svg"); assert(svg);
  assert.equal(diagram.querySelectorAll(".graph-node").length, 10); assert.equal(diagram.querySelectorAll(".graph-edge").length, 10);
  const box = svg.getAttribute("viewBox").split(/\s+/).map(Number); assert.equal(box.length, 4);
  for (const rect of diagram.querySelectorAll("rect")) {
    const x = Number(rect.getAttribute("x")), y = Number(rect.getAttribute("y")); assert(x >= 0 && y >= 0);
    assert(x + Number(rect.getAttribute("width")) <= box[2]); assert(y + Number(rect.getAttribute("height")) <= box[3]);
  }
});

for (const [label, change] of [
  ["wrong schema", data => { data.schema_version = "future"; }],
  ["live fixture", data => { data.fixture.origin = "live"; }],
  ["nonzero request count", data => { data.request_counts.chat = 1; }],
  ["source outside lesson", data => { data.definitions.resume.file = "/etc/passwd"; }],
  ["source binding mismatch", data => { data.definitions.review.name = "arbitrary"; }],
  ["source text mismatch", data => { data.definitions.review.code = "def arbitrary(state):\n    return {}"; }],
  ["invalid source line range", data => { data.definitions.review.end_line = 1; }],
  ["missing guided definition", data => { delete data.definitions.start; }],
  ["unknown node", data => { data.nodes[0].id = "arbitrary"; }],
]) {
  test(`invalid initial ${label} disables mutations without auto-retry`, async () => {
    const initial = fixture(); change(initial); const h = harness({ initial }); await settle(); await h.eval("startRun()");
    assert.equal(h.calls.length, 1); assert.equal(h.get("start-button").disabled, true); assert.equal(h.get("error").hidden, false);
  });
}

for (const [label, change] of [
  ["invalid session", data => { data.session_id = "../../session"; }],
  ["unknown phase", data => { data.phase = "loading"; }],
  ["missing checkpoint", data => { data.checkpoint = null; }],
  ["checkpoint thread mismatch", data => { data.checkpoint.thread_id = "b".repeat(32); }],
  ["wrong pending node", data => { data.checkpoint.next = ["generate"]; }],
  ["missing interrupt", data => { data.interrupt = null; }],
  ["wrong interrupt kind", data => { data.interrupt.value.kind = "other"; }],
  ["forged interrupt evidence count", data => { data.interrupt.value.evidence_count = 0; }],
  ["review incorrectly completed", data => { data.run_counts.review.completed = 1; }],
  ["prefix incorrectly rerun", data => { data.run_counts.retrieve.entered = 2; }],
  ["generation before approval", data => { data.run_counts.generate.entered = 1; }],
  ["preapproved state", data => { data.values.approved = true; }],
  ["draft smuggled before approval", data => { data.values.draft = { status: "partial" }; }],
  ["answer smuggled before approval", data => { data.values.answer = { status: "partial" }; }],
  ["broken event continuity", data => { data.events[1].before.query = "other"; }],
  ["broken merged event", data => { data.events[0].after.extra = true; }],
  ["forged current values", data => { data.values.evidence.S1.text = "other"; }],
  ["unknown graph edge", data => { data.graph.edges[0].target = "cancel"; }],
  ["missing graph node", data => { data.graph.nodes.pop(); }],
]) {
  test(`invalid paused ${label} is not shown as a resumable snapshot`, async () => {
    const data = fixture("paused"); change(data); const h = await paused({ startData: data });
    assert.equal(h.get("approve-button").disabled, true); assert.equal(h.get("cancel-button").disabled, true);
    assert.equal(h.get("error").hidden, false); assert.equal(h.calls.length, 2);
  });
}

test("resume response must belong to submitted session, and failure cannot remain resumable", async () => {
  const h = await paused({ resumeData: fixture("completed", "b".repeat(32)) }); await h.eval("resumeRun(true)");
  assert.equal(h.get("error").hidden, false); assert.equal(h.get("approve-button").disabled, true);
  const failed = await paused({ failResume: true }); await failed.eval("resumeRun(false)");
  assert.equal(failed.calls.length, 3); assert.equal(failed.get("error").hidden, false);
  assert.equal(failed.get("approve-button").disabled, true); assert.equal(failed.get("cancel-button").disabled, true);
  await failed.eval("resumeRun(true)"); assert.equal(failed.calls.length, 3);
  assert.equal(failed.get("start-button").disabled, false);
});

for (const [label, phase, change] of [
  ["completed before review returned", "completed", data => { data.run_counts.review.completed = 0; }],
  ["completed prefix rerun", "completed", data => { data.run_counts.retrieve.entered = 2; }],
  ["completed with pending next node", "completed", data => { data.checkpoint.next = ["generate"]; }],
  ["completed with interrupt", "completed", data => { data.interrupt = clone(fixture("paused").interrupt); }],
  ["completed missing answer", "completed", data => { delete data.values.answer; }],
  ["completed with negative decision", "completed", data => { data.values.approved = false; }],
  ["cancelled with generated answer", "cancelled", data => { data.values.answer = clone(fixture("completed").values.answer); }],
  ["cancelled but validate executed", "cancelled", data => { data.run_counts.validate = { entered: 1, completed: 1 }; }],
]) {
  test(`invalid ${label} cannot be restored as a successful server snapshot`, async () => {
    const data = fixture(phase); change(data); const h = harness({ initial: data }); await settle();
    assert.equal(h.get("error").hidden, false); assert.equal(h.get("approve-button").disabled, true);
    assert.equal(h.get("cancel-button").disabled, true); assert.equal(h.calls.length, 1);
  });
}

test("a resume response cannot silently substitute cancellation for the submitted approval", async () => {
  const h = await paused({ resumeData: fixture("cancelled") }); await h.eval("resumeRun(true)");
  assert.equal(h.get("error").hidden, false); assert.equal(h.get("approve-button").disabled, true);
  assert.equal(h.calls.length, 3);
});

test("initial GET and start errors remain explicit, do not retry or pretend to be paused", async () => {
  const get = harness({ failGet: true }); await settle(); assert.equal(get.calls.length, 1); assert.equal(get.get("start-button").disabled, true);
  const h = await paused({ failStart: true }); assert.equal(h.calls.length, 2); assert.equal(h.get("error").hidden, false);
  assert.equal(h.get("approve-button").disabled, true); assert.equal(h.get("cancel-button").disabled, true);
  assert(!h.get("full-json").textContent.includes('"phase": "paused"'));
});

test("missing LangGraph dependency disables start and resume without fallback", async () => {
  const initial = fixture(); initial.available = false; initial.langgraph_version = null;
  const h = harness({ initial }); await settle(); await h.eval("startRun()"); await h.eval("resumeRun(true)");
  assert.equal(h.calls.length, 1); assert.equal(h.get("start-button").disabled, true);
  assert.equal(h.get("approve-button").disabled, true); assert.equal(h.get("cancel-button").disabled, true);
});

test("read-only recovery fetches current server state without sending another decision", async () => {
  const h = harness({ initial: fixture("paused") }); await settle();
  await h.eval("loadState()"); assert.equal(h.calls.length, 2);
  assert(h.calls.every(call => call.url === "/api/workflow/approval/state" && call.options.method === "GET"));
  assert.equal(h.eval("state.data.phase"), "paused");
});

test("explicit recovery after a conflicting resume reads terminal server state without resending approval", async () => {
  const h = await paused({ initial: fixture("completed"), failResume: true });
  await h.eval("resumeRun(true)"); assert.equal(h.calls.length, 3);
  await h.eval("loadState()"); assert.equal(h.calls.length, 4);
  assert.equal(h.calls.at(-1).url, "/api/workflow/approval/state");
  assert.equal(h.eval("state.data.phase"), "completed"); assert.equal(h.get("approve-button").disabled, true);
  assert.equal(h.calls.filter(call => call.url.endsWith("/resume")).length, 1);
});

test("pending initial GET cannot authorize a premature start or stale resume", async () => {
  const h = harness({ initial: fixture("paused"), deferGet: true });
  await h.eval("startRun()"); await h.eval("resumeRun(true)"); assert.equal(h.calls.length, 1);
  assert.equal(h.get("approve-button").disabled, true); h.finish(); await settle();
  assert.equal(h.eval("state.data.phase"), "paused"); assert.equal(h.calls.length, 1);
});

test("programmatic string or numeric approval is rejected before sending a POST", async () => {
  const h = await paused(); const count = h.calls.length;
  await h.eval("resumeRun('true')"); await h.eval("resumeRun(1)"); await h.eval("resumeRun(null)");
  assert.equal(h.calls.length, count);
});

test("existing branching lesson remains independent and links forward without changing its APIs", () => {
  const branch = fs.readFileSync(path.join(__dirname, "../web/branching_lab.html"), "utf8");
  assert(branch.includes('"/api/workflow/branching/plan"')); assert(branch.includes('"/api/workflow/branching/replay"'));
  assert(!script.includes('fetch("/api/workflow/branching/replay"'));
});
