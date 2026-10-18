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

function definitions(nodes) {
  const source = (name, code, start_line = 10) => ({ name, code, start_line, end_line: start_line + code.split("\n").length - 1 });
  const fieldTypes = { query: "str", bm25: "list[dict]", dense: "list[dict]", rrf_pool: "list[dict]", reranked: "list[dict]", evidence: "dict", draft: "dict", answer: "dict" };
  return { schema_version: "workflow-definitions/v1", source_file: "src/qa_platform/workflows/replay.py",
    state: { ...source("ReplayState", "class ReplayState(TypedDict, total=False):\n    query: str\n    evidence: dict"),
      fields: Object.entries(fieldTypes).map(([name, type]) => ({ name, type, required: false,
        read_by: nodes.filter(node => node.reads.includes(name)).map(node => node.id),
        written_by: nodes.filter(node => node.writes.includes(name)).map(node => node.id) })) },
    factory: source("make_nodes", "def make_nodes(fixture: dict) -> dict:\n    return {\"retrieve\": retrieve, \"fuse\": fuse}"),
    setup: source("_execute_shared", "def _execute_shared(fixture: dict):\n    shared_nodes = make_nodes(fixture)\n    plain = _run_plain(fixture, shared_nodes)\n    graph = build_graph(fixture, nodes=shared_nodes)"),
    nodes: Object.fromEntries(nodes.map((node, index) => [node.id, source(node.id,
      `def ${node.id}(state: ReplayState) -> dict:\n    return {\"${node.writes[0]}\": \"合成增量\"}`, 30 + index * 5)])),
    graph_builder: source("build_graph", "def build_graph(fixture: dict, *, nodes: dict | None = None):\n    builder = StateGraph(ReplayState)\n    return builder.compile(checkpointer=None, cache=None)"),
    notice: "真实源码的只读展示；shared_nodes 是 Python 字典，不是 LangGraph 内置变量。" };
}

function fixture() {
  const nodes = ORDER.map((id, index) => ({ id, label: `合成步骤 ${index + 1}`,
    reads: [["query"], ["bm25", "dense"], ["rrf_pool"], ["reranked"], ["evidence"], ["draft", "evidence"]][index], writes: [["bm25", "dense"], ["rrf_pool"], ["reranked"], ["evidence"], ["draft"], ["answer"]][index],
    operation: ["读取合成快照", "本地计算", "读取合成快照", "本地计算", "读取合成快照", "本地校验"][index] }));
  const common = { schema_version: "rag-workflow-replay/v1", langgraph_version: "1.2.16",
    fixture: { id: "dns-teaching-v1", title: "合成快照 <img onerror=alert(1)>", origin: "synthetic", query: "Pod 中 DNS 如何排查？", notice: "只用合成数据，模型请求 0。" },
    nodes, definitions: definitions(nodes), request_counts: { embedding: 0, http: 0, reranker: 0, chat: 0 }, notice: "教学回放，无模型调用。" };
  const evidence = { S1: { chunk_id: "chunk-b", text: "如果完全限定名称可解析，但相对名称不能解析，需要检查 Pod 中的\n`/etc/resolv.conf` 文件。",
    quotes: { Q1: "如果完全限定名称可解析，但相对名称不能解析，需要检查 Pod 中的\n`/etc/resolv.conf` 文件。" } },
    S2: { chunk_id: "chunk-a", text: "命名空间学习示例 <img onerror=alert(1)>，并非实际运维规程。", quotes: { Q1: "命名空间学习示例 <img onerror=alert(1)>，并非实际运维规程。" } } };
  const draft = { status: "partial", claims: [
    { text: "合成回答 <script>throw 1</script>：需要检查 /etc/resolv.conf。", evidence: [{ source_id: "S1", quote_id: "Q1" }] },
    { text: "另一条命名空间学习示例。", evidence: [{ source_id: "S2", quote_id: "Q1" }] }],
    missing_info: ["缺少现场资料；合成数据不能作为维护依据。"] };
  const updates = [{ bm25: ["chunk-a"], dense: ["chunk-b"] }, { rrf_pool: ["chunk-a", "chunk-b"] },
    { reranked: ["chunk-b", "chunk-a"] }, { evidence }, { draft }, { answer: clone(draft) }];
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
  setAttribute(key, value) {
    this.attributes[key] = String(value);
    if (key === "class") this.className = String(value);
    if (key.startsWith("data-")) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = String(value);
  }
  getAttribute(key) { return this.attributes[key] ?? null; }
  addEventListener(type, fn) { this.listeners[type] = fn; }
  querySelectorAll(selector) {
    const found = [];
    for (const child of this.children) {
      if (selector.startsWith(".") ? child.className.split(/\s+/).includes(selector.slice(1)) : child.tagName.toLowerCase() === selector.toLowerCase()) found.push(child);
      found.push(...child.querySelectorAll(selector));
    }
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
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
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: tag => new Element(tag),
    createElementNS: (namespace, tag) => { const item = new Element(tag); item.namespaceURI = namespace; return item; } }, fetch });
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
  ["forged equal verdict", data => { data.langgraph.final_state.answer.claims[0].text = "不相同"; }],
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

test("plan exposes real source definitions as literal read-only numbered text", async () => {
  const data = plan(); data.definitions.state.code += '\n    # <svg onload="throw 1">'; data.definitions.state.end_line += 1;
  const h = harness({ planData: data }); await settle();
  assert.equal(h.get("definition-tabs").children.length, 4);
  assert(h.get("definition-code").textContent.includes("class ReplayState(TypedDict, total=False):"));
  assert(h.get("definition-code").textContent.includes('<svg onload="throw 1">'));
  assert.equal(h.get("definition-code").children.length, 4);
  assert(h.get("definition-location").textContent.includes("src/qa_platform/workflows/replay.py:10–13"));
  assert(h.get("definitions-status").textContent.includes("只读"));
  assert(h.get("state-field-detail").textContent.includes("不伪造字段值"));
  assert.equal(h.calls.length, 1);
});

test("shared_nodes inspector shows ordinary Python dictionary setup and full factory without executing it", async () => {
  const h = harness(); await settle(); h.eval('selectDefinition("setup")');
  assert(h.get("definition-code").textContent.includes("shared_nodes = make_nodes(fixture)"));
  assert(h.get("definition-code").textContent.includes("_run_plain(fixture, shared_nodes)"));
  assert(h.get("definition-code").textContent.includes("build_graph(fixture, nodes=shared_nodes)"));
  assert(h.get("definition-explanation").textContent.includes("不是 LangGraph"));
  assert(h.get("factory-code").textContent.includes("def make_nodes"));
  assert.equal(h.get("node-registry").children.length, 6); assert.equal(h.calls.length, 1);
});

test("old plan and replay without optional definitions remain usable but never fabricate source", async () => {
  const oldPlan = plan(), oldReplay = fixture(); delete oldPlan.definitions; delete oldReplay.definitions;
  const h = harness({ planData: oldPlan, postData: oldReplay }); await settle();
  assert(h.get("definitions-status").textContent.includes("未提供真实源码"));
  assert.equal(h.get("definition-code").textContent, "");
  assert.equal(h.get("factory-details").hidden, true);
  await h.eval("runReplay()"); assert.equal(h.get("results").hidden, false);
  assert.equal(h.get("definition-code").textContent, ""); assert.equal(h.calls.length, 2);
});

for (const [label, mutate] of [
  ["definition schema", data => { data.definitions.schema_version = "future"; }],
  ["arbitrary source path", data => { data.definitions.source_file = "/private/secret.py"; }],
  ["missing registered node", data => { delete data.definitions.nodes.fuse; }],
  ["node identity", data => { data.definitions.nodes.fuse.name = "retrieve"; }],
  ["missing source code", data => { data.definitions.factory.code = ""; }],
  ["invalid line number", data => { data.definitions.state.start_line = 0; }],
  ["reversed source range", data => { data.definitions.setup.end_line = 1; }],
  ["duplicate state field", data => { data.definitions.state.fields[1].name = "query"; }],
  ["required state field", data => { data.definitions.state.fields[0].required = true; }],
  ["forged writer mapping", data => { data.definitions.state.fields.find(field => field.name === "evidence").written_by = ["generate"]; }],
]) {
  test(`malformed ${label} fails closed during plan loading`, async () => {
    const data = plan(); mutate(data); const h = harness({ planData: data }); await settle();
    assert.equal(h.get("run-button").disabled, true); assert.equal(h.get("error").hidden, false);
    assert.equal(h.eval("state.plan"), null); assert.equal(h.calls.length, 1);
    assert.equal(h.get("definition-code").textContent, "");
  });
}

test("malformed replay definitions do not reuse the plan source as if the result were valid", async () => {
  const data = fixture(); data.definitions.graph_builder.name = "other";
  const h = harness({ postData: data }); await settle(); await h.eval("runReplay()");
  assert.equal(h.get("results").hidden, true); assert.equal(h.eval("state.data"), null);
  assert.equal(h.get("error").hidden, false); assert.equal(h.calls.length, 2);
});

test("field inspector distinguishes declared, absent, newly written and already present fields", async () => {
  const h = harness(); await settle();
  assert.equal(h.get("state-fields").children.length, 8);
  assert(h.get("state-fields").children.every(button => button.textContent.includes("只有定义")));
  await h.eval("runReplay()"); h.eval('selectStateField("evidence")');
  assert(h.get("state-field-detail").textContent.includes("尚未产生"));
  assert.equal(h.get("state-field-detail").querySelectorAll("pre").length, 0);
  h.eval("selectStep(3)");
  assert(h.get("state-fields").children.find(button => button.dataset.field === "evidence").textContent.includes("本步新增"));
  assert(h.get("state-field-detail").textContent.includes("prepare_evidence"));
  assert(h.get("state-field-detail").textContent.includes("generate、validate"));
  assert(h.get("state-field-detail").querySelector("pre").textContent.includes('"S1"'));
  h.eval("selectStep(4)");
  assert(h.get("state-fields").children.find(button => button.dataset.field === "evidence").textContent.includes("本步已存在"));
  assert(h.get("state-field-detail").textContent.includes("此前产生或初始输入"));
  h.eval('selectStateField("query")');
  assert(h.get("state-field-detail").textContent.includes("由初始输入提供"));
  assert.equal(h.calls.length, 2);
});

test("registry and current-node source follow step selection using returned source snippets", async () => {
  const h = harness(); await settle();
  h.get("node-registry").children.find(button => button.dataset.node === "fuse").listeners.click();
  assert(h.get("definition-code").textContent.includes("def fuse(state: ReplayState)"));
  assert.equal(h.eval("state.data"), null); assert.equal(h.calls.length, 1);
  await h.eval("runReplay()"); h.eval('selectDefinition("node")'); h.eval("selectStep(4)");
  assert(h.get("definition-code").textContent.includes("def generate(state: ReplayState)"));
  assert(h.get("definition-explanation").textContent.includes('shared_nodes["generate"]'));
  h.get("node-registry").children.find(button => button.dataset.node === "validate").listeners.click();
  assert.equal(h.get("step-count").textContent, "6 / 6");
  assert(h.get("definition-code").textContent.includes("def validate(state: ReplayState)"));
  h.eval('selectDefinition("graph_builder")');
  assert(h.get("definition-code").textContent.includes("builder = StateGraph(ReplayState)"));
  assert.equal(h.calls.length, 2);
});

test("SVG renders exact compiled edges with forward arrows even when server arrays are shuffled", async () => {
  const data = fixture(); data.graph.edges.reverse(); data.graph.nodes.reverse();
  const h = harness({ postData: data }); await settle(); await h.eval("runReplay()");
  const svg = h.get("diagram").children[0];
  assert.equal(svg.tagName, "SVG"); assert.equal(svg.namespaceURI, "http://www.w3.org/2000/svg");
  const boxes = svg.querySelectorAll(".graph-node"), edges = svg.querySelectorAll(".edge-line");
  assert.deepEqual(boxes.map(box => box.dataset.node), ["__start__", ...ORDER, "__end__"]);
  assert.equal(edges.length, 7);
  assert.deepEqual(edges.map(edge => [edge.dataset.source, edge.dataset.target]), data.graph.edges.map(edge => [edge.source, edge.target]));
  for (const edge of edges) {
    assert.equal(edge.getAttribute("marker-end"), "url(#workflow-arrow)");
    const [fromX, fromY, toX, toY] = edge.getAttribute("d").match(/-?\d+(?:\.\d+)?/g).map(Number);
    assert(fromX < toX); assert.equal(fromY, toY);
  }
  for (const terminal of boxes.filter(box => box.className.includes("terminal"))) {
    assert.equal(terminal.getAttribute("role"), null); assert.equal(terminal.listeners.click, undefined);
  }
  assert.equal(h.calls.length, 2);
});

test("SVG node is keyboard operable and progress labels are inspection status only", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()"); const count = h.calls.length;
  const boxes = h.get("diagram").querySelectorAll(".graph-node"), generate = boxes.find(box => box.dataset.node === "generate");
  assert.equal(generate.getAttribute("role"), "button"); assert.equal(generate.getAttribute("tabindex"), "0");
  let prevented = 0; generate.listeners.keydown({ key: "Enter", preventDefault: () => { prevented += 1; } });
  assert.equal(h.get("step-count").textContent, "5 / 6"); assert.equal(prevented, 1);
  assert.equal(generate.getAttribute("aria-pressed"), "true");
  assert.equal(generate.querySelector(".node-status").textContent, "当前查看");
  assert(boxes.find(box => box.dataset.node === "fuse").querySelector(".node-status").textContent.includes("均已执行"));
  const fuse = boxes.find(box => box.dataset.node === "fuse");
  fuse.listeners.keydown({ key: " ", preventDefault: () => { prevented += 1; } });
  assert.equal(h.get("step-count").textContent, "2 / 6"); assert.equal(prevented, 2);
  fuse.listeners.keydown({ key: "Escape", preventDefault: () => { prevented += 1; } });
  assert.equal(prevented, 2); assert.equal(h.calls.length, count);
});

test("narrow diagram uses a complete vertical graph inside its viewBox and remains locally interactive", async () => {
  const h = harness(); h.get("diagram").clientWidth = 420; await settle(); await h.eval("runReplay()");
  const count = h.calls.length, svg = h.get("diagram").children[0];
  const [viewX, viewY, viewWidth, viewHeight] = svg.getAttribute("viewBox").split(/\s+/).map(Number);
  assert.equal(viewX, 0); assert.equal(viewY, 0); assert(viewWidth > 0 && viewWidth <= 420); assert(viewHeight > viewWidth);
  const boxes = svg.querySelectorAll(".graph-node"), edges = svg.querySelectorAll(".edge-line");
  assert.equal(boxes.length, 8); assert.equal(edges.length, 7);
  const centers = [];
  for (const box of boxes) {
    const rect = box.querySelector("rect"), x = Number(rect.getAttribute("x")), y = Number(rect.getAttribute("y"));
    const width = Number(rect.getAttribute("width")), height = Number(rect.getAttribute("height"));
    assert(x >= 0 && y >= 0); assert(x + width <= viewWidth); assert(y + height <= viewHeight);
    centers.push(x + width / 2);
    for (const label of box.querySelectorAll("text")) {
      const labelX = Number(label.getAttribute("x")), labelY = Number(label.getAttribute("y"));
      assert(labelX >= x && labelX <= x + width); assert(labelY >= y && labelY <= y + height);
    }
  }
  assert(centers.every(center => center === centers[0]));
  for (const edge of edges) {
    const [fromX, fromY, toX, toY] = edge.getAttribute("d").match(/-?\d+(?:\.\d+)?/g).map(Number);
    assert.equal(fromX, toX); assert(fromY < toY);
    assert.equal(edge.getAttribute("marker-end"), "url(#workflow-arrow)");
  }
  const generate = boxes.find(box => box.dataset.node === "generate");
  let prevented = false; generate.listeners.keydown({ key: "Enter", preventDefault: () => { prevented = true; } });
  assert(prevented); assert.equal(h.get("step-count").textContent, "5 / 6");
  assert.equal(generate.getAttribute("aria-pressed"), "true");
  assert(h.get("claim-cards").textContent.includes("合成回答")); assert.equal(h.calls.length, count);
});

test("reference inspector uses current after only and never reveals future evidence or draft", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()");
  for (const index of [0, 1, 2]) {
    h.eval(`selectStep(${index})`);
    assert.equal(h.get("source-cards").querySelectorAll(".source-card").length, 0);
    assert.equal(h.get("claim-cards").querySelectorAll(".claim-card").length, 0);
    assert(!h.get("source-cards").textContent.includes("/etc/resolv.conf"));
    assert(!h.get("claim-cards").textContent.includes("合成回答"));
  }
  h.eval("selectStep(3)");
  assert.equal(h.get("source-cards").querySelectorAll(".source-card").length, 2);
  assert.equal(h.get("claim-cards").querySelectorAll(".claim-card").length, 0);
  assert(h.get("reference-path").textContent.includes("当前尚无草稿引用"));
  h.eval("selectStep(4)");
  assert.equal(h.get("claim-cards").querySelectorAll(".claim-card").length, 2);
  assert(h.get("reference-path").textContent.includes("draft.claims[0].evidence[0]"));
  assert(h.get("draft-role").textContent.includes("草稿还不是已校验回答"));
  h.eval("selectStep(5)"); assert(h.get("draft-role").textContent.includes("answer 已产生"));
  assert(h.get("draft-role").textContent.includes("不等于语义支持")); assert.equal(h.calls.length, 2);
});

test("claim citation links select exact S/Q source and highlight original multiline text literally", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()"); h.eval("selectStep(4)"); const count = h.calls.length;
  const claims = h.get("claim-cards").querySelectorAll(".claim-card"); claims[1].querySelector("button").listeners.click();
  assert.equal(h.eval("state.reference.source_id"), "S2"); assert.equal(h.eval("state.reference.quote_id"), "Q1");
  assert.equal(h.eval("state.reference.claim_index"), 1);
  assert(h.get("reference-path").textContent.includes("draft.claims[1].evidence[0] → evidence.S2.quotes.Q1"));
  const selected = h.get("source-cards").querySelectorAll(".source-card").find(card => card.className.includes("selected"));
  assert.equal(selected.dataset.source, "S2");
  assert(selected.querySelector("mark").textContent.includes("<img onerror=alert(1)>"));
  assert.equal(selected.querySelectorAll("img").length, 0);
  const sourceButton = h.get("source-cards").querySelectorAll("button").find(button => button.dataset.source === "S1");
  sourceButton.listeners.click();
  const original = h.get("source-cards").querySelectorAll(".source-card").find(card => card.dataset.source === "S1").querySelector("mark");
  assert(original.textContent.includes("中的\n`/etc/resolv.conf`"));
  assert.equal(h.get("claim-cards").querySelectorAll(".claim-card").find(card => card.className.includes("selected")).dataset.claim, "0");
  assert.equal(h.calls.length, count);
});

test("unknown S/Q reference reports missing location without inventing quote or model errors", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()"); h.eval("selectStep(4)");
  const sources = h.get("source-cards").textContent;
  h.eval('selectReference("S999", "Q999", 0, 0)');
  assert(h.get("reference-path").textContent.includes("不存在；未伪造原文"));
  assert.equal(h.get("source-cards").textContent, sources); assert.equal(h.get("error").hidden, true);
  assert.equal(h.calls.length, 2);
});

test("rerunning clears old reference selection and state values before returning to step one", async () => {
  const h = harness(); await settle(); await h.eval("runReplay()"); h.eval("selectStep(4)"); h.eval('selectReference("S2", "Q1", 1, 0)');
  h.eval('selectStateField("draft")'); assert(h.get("state-field-detail").textContent.includes("合成回答"));
  await h.eval("runReplay()");
  assert.equal(h.eval("state.reference"), null); assert.equal(h.get("step-count").textContent, "1 / 6");
  assert.equal(h.get("source-cards").querySelectorAll(".source-card").length, 0);
  assert.equal(h.get("claim-cards").querySelectorAll(".claim-card").length, 0);
  assert(!h.get("state-field-detail").textContent.includes("合成回答"));
  assert.equal(h.calls.filter(call => call.options.method === "POST").length, 2);
});

test("in-flight rerun immediately removes previous field values from the visible definition inspector", async () => {
  const h = harness({ deferPost: true }); await settle();
  const first = h.eval("runReplay()"); h.finish(); await first;
  h.eval("selectStep(4)"); h.eval('selectStateField("draft")'); h.eval('selectReference("S2", "Q1", 1, 0)');
  assert(h.get("state-field-detail").textContent.includes("合成回答"));
  const second = h.eval("runReplay()");
  assert.equal(h.eval("state.reference"), null); assert.equal(h.eval("state.data"), null);
  assert.equal(h.get("results").hidden, true); assert(!h.get("state-field-detail").textContent.includes("合成回答"));
  assert(h.get("state-field-detail").textContent.includes("尚未执行回放"));
  h.finish(); await second; assert.equal(h.calls.length, 3);
});
