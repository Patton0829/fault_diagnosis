"use strict";

// 只用合成合同、内存 DOM 和假 fetch：不启动浏览器、数据库或模型服务。
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/branching_lab.html"), "utf8");
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const PREFIX = ["retrieve", "fuse", "rerank", "prepare_evidence"];
const NODES = [...PREFIX, "generate", "refuse", "validate"];
const clone = value => JSON.parse(JSON.stringify(value));
const SCENARIOS = [
  { id: "with_evidence", label: "有候选资料", description: "非空不等于可靠。" },
  { id: "empty_evidence", label: "召回为空", description: "两路召回置空，固定拒答。" },
];

function definitions() {
  const source = (name, code, file = "branching.py", start_line = 10) => ({ name, code,
    file: `src/qa_platform/workflows/${file}`, start_line, end_line: start_line + code.split("\n").length - 1 });
  return {
    state: source("ReplayState", "class ReplayState(TypedDict, total=False):\n    evidence: dict", "replay.py"),
    router: source("route_after_evidence", 'def route_after_evidence(state):\n    return "generate" if state["evidence"] else "refuse"'),
    factory: source("make_nodes", "def make_nodes(fixture):\n    return shared_nodes"),
    plain: source("_run_plain", 'def _run_plain(fixture, nodes):\n    if target == "generate":\n        call("generate")\n    else:\n        call("refuse")'),
    graph_builder: source("build_graph", 'def build_graph(fixture):\n    builder.add_conditional_edges("prepare_evidence", route_after_evidence)\n    return builder.compile()'),
    nodes: Object.fromEntries(NODES.map(id => [id, source(id, `def ${id}(state):\n    return {}`, id === "refuse" ? "branching.py" : "replay.py")])),
  };
}

function fixture(scenario = "with_evidence") {
  const target = scenario === "with_evidence" ? "generate" : "refuse";
  const evidence = scenario === "with_evidence" ? {
    S1: { chunk_id: "demo-namespace", text: "合成原文 <img onerror=alert(1)>", quotes: { Q1: "合成原文 <img onerror=alert(1)>" } },
    S2: { chunk_id: "demo-dns-config", text: "另一条合成原文", quotes: { Q1: "另一条合成原文" } },
  } : {};
  const draft = target === "generate" ? { status: "partial", claims: [{ text: "合成回答 <script>throw 1</script>", evidence: [{ source_id: "S1", quote_id: "Q1" }] }], missing_info: ["缺少现场信息"] }
    : { status: "insufficient_evidence", claims: [], missing_info: ["召回为空，需要补充可检索资料"] };
  const candidate = { chunk_id: "demo-namespace", score: 1 };
  const candidates = target === "generate" ? [candidate] : [];
  const updates = [{ bm25: candidates, dense: candidates }, { rrf_pool: candidates }, { reranked: candidates }, { evidence }, { draft }, { answer: clone(draft) }];
  const order = [...PREFIX, target, "validate"];
  let current = { query: "Pod 中 DNS 如何排查？" };
  const events = updates.map((update, index) => {
    const before = clone(current); current = { ...current, ...clone(update) };
    return { step: index + 1, node: order[index], before, update, after: clone(current) };
  });
  const decisions = [{ after_node: "prepare_evidence", target, evidence_count: Object.keys(evidence).length,
    reason: target === "generate" ? "证据字典非空，进入草稿回放；尚未判定质量。" : "证据字典为空，跳过草稿回放，使用固定拒答。" }];
  const edges = [["__start__", "retrieve"], ["retrieve", "fuse"], ["fuse", "rerank"], ["rerank", "prepare_evidence"],
    ["prepare_evidence", "generate"], ["prepare_evidence", "refuse"], ["generate", "validate"], ["refuse", "validate"], ["validate", "__end__"]];
  return { schema_version: "rag-workflow-branching/v1", langgraph_version: "1.2.16", scenarios: clone(SCENARIOS),
    scenario: clone(SCENARIOS.find(item => item.id === scenario)),
    fixture: { id: "dns-teaching-v1", title: "教学合成 <img onerror=alert(1)>", origin: "synthetic", query: "Pod 中 DNS 如何排查？", notice: "合成数据，非现场规程。" },
    nodes: NODES.map((id, index) => ({ id, label: `节点 ${id}`, reads: [index === 0 ? "query" : "evidence"], writes: [id === "validate" ? "answer" : "evidence"], operation: "合成回放" })),
    definitions: definitions(), notice: "证据非空不是质量判定；零模型。",
    request_counts: { embedding: 0, http: 0, reranker: 0, chat: 0 },
    graph: { nodes: ["__start__", ...NODES, "__end__"].map(id => ({ id, label: id })),
      edges: edges.map(([source, target]) => ({ source, target, conditional: source === "prepare_evidence" })) },
    plain: { engine: "python", events: clone(events), decisions: clone(decisions), final_state: clone(current) },
    langgraph: { engine: "langgraph", events: clone(events), decisions: clone(decisions), final_state: clone(current) },
    comparison: { same_steps: true, same_updates: true, same_final_state: true, same_answer: true, same_decisions: true },
  };
}

function plan() {
  const data = fixture();
  for (const key of ["scenario", "graph", "plain", "langgraph", "comparison"]) delete data[key];
  return { ...data, available: true, dependency_message: "沿用验证版本，不自动安装。" };
}

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this._text = ""; this.className = "";
    this.dataset = {}; this.attributes = {}; this.listeners = {}; this.disabled = false; this.hidden = false; this.open = false;
    this.value = ""; this.clientWidth = 400;
    this.classList = { toggle: (name, enabled) => {
      const names = new Set(this.className.split(/\s+/).filter(Boolean));
      if (enabled) names.add(name); else names.delete(name); this.className = [...names].join(" ");
    }, add: name => { this.className += ` ${name}`; }, remove: name => { this.className = this.className.split(/\s+/).filter(item => item !== name).join(" "); } };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.children.push(child); return child; }
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

function harness({ planData = plan(), postData = null, failPlan = false, failPost = false, deferPost = false } = {}) {
  const elements = new Map([...page.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)].map(match => {
    const item = new Element(match[1]); item.hidden = /\bhidden\b/.test(match[2]); item.disabled = /\bdisabled\b/.test(match[2]);
    return [match[3], item];
  }));
  elements.get("scenario-select").value = "with_evidence";
  elements.get("source-select").value = "router";
  const calls = []; const pending = [];
  const fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === "/api/workflow/branching/plan") return { ok: !failPlan, json: async () => clone(planData) };
    if (url === "/api/workflow/branching/replay") {
      const requested = JSON.parse(options.body).scenario;
      if (deferPost) await new Promise(resolve => pending.push(resolve));
      return { ok: !failPost, json: async () => clone(postData || fixture(requested)) };
    }
    throw new Error(`unexpected network target ${url}`);
  };
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: tag => new Element(tag),
    createElementNS: (namespace, tag) => { const item = new Element(tag); item.namespaceURI = namespace; return item; } }, fetch });
  vm.runInContext(script, context);
  return { context, calls, elements, get: id => elements.get(id), eval: code => vm.runInContext(code, context), finish: () => pending.shift()(),
    scenario: id => { elements.get("scenario-select").value = id; elements.get("scenario-select").listeners.change(); } };
}
async function settle() { await new Promise(resolve => setImmediate(resolve)); }
async function run(options) { const h = harness(options); await settle(); await h.eval("runReplay()"); return h; }

test("independent learning page has no external scripts or HTML execution sinks", () => {
  assert(!page.includes("innerHTML")); assert(!page.includes("insertAdjacentHTML"));
  assert(!page.match(/<script[^>]+src=/)); assert(!page.match(/<link[^>]+href="https?:/));
  assert(page.includes('href="/workflow"')); assert(page.includes('role="status"'));
});

test("initial load fetches only the plan, not a replay or any model endpoint", async () => {
  const h = harness(); await settle();
  assert.equal(h.calls.length, 1); assert.equal(h.calls[0].url, "/api/workflow/branching/plan");
  assert.equal(h.calls[0].options.method, "GET"); assert.equal(h.get("run-button").disabled, false);
  assert.equal(h.get("results").hidden, true); assert.equal(h.eval("state.data"), null);
  assert(page.includes('<option value="with_evidence">')); assert(page.includes('<option value="empty_evidence">'));
});

test("source and complete states are folded by default, keeping the main exercise compact", () => {
  const details = [...page.matchAll(/<details\b([^>]*)>/g)];
  assert(details.length >= 2); assert(details.every(match => !/\bopen\b/.test(match[1])));
  assert(page.includes('id="plain-update"')); assert(page.includes('id="langgraph-update"'));
});

test("one click sends exactly one guarded, scenario-scoped same-origin POST", async () => {
  const h = await run(); const call = h.calls[1];
  assert.equal(h.calls.length, 2); assert.equal(call.url, "/api/workflow/branching/replay");
  assert.equal(call.options.method, "POST"); assert.equal(call.options.mode, "same-origin");
  assert.equal(call.options.credentials, "same-origin"); assert.equal(call.options.headers["X-Retrieval-Lab"], "1");
  assert.equal(call.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(call.options.body), { scenario: "with_evidence" });
  assert.equal(h.get("results").hidden, false); assert(h.get("checks").textContent.includes("路由记录一致"));
  assert.equal(h.get("step-count").textContent, "4 / 6");
});

for (const [scenario, used, skipped] of [["with_evidence", "generate", "refuse"], ["empty_evidence", "refuse", "generate"]]) {
  test(`${scenario} draws both actual graph branches but executes only ${used}`, async () => {
    const h = harness(); await settle(); h.scenario(scenario); await h.eval("runReplay()");
    const boxes = h.get("diagram").querySelectorAll(".graph-node");
    assert.equal(boxes.length, 9); assert.equal(h.get("diagram").querySelectorAll(".graph-edge").length, 9);
    assert.equal(h.get("diagram").querySelectorAll(".graph-edge").filter(edge => edge.className.split(/\s+/).includes("used")).length, 7);
    const usedNode = boxes.find(item => item.dataset.node === used), skippedNode = boxes.find(item => item.dataset.node === skipped);
    assert(usedNode); assert(skippedNode); assert.equal(typeof usedNode.listeners.click, "function");
    assert.equal(skippedNode.listeners.click, undefined); assert(skippedNode.textContent.includes("未执行"));
    usedNode.listeners.click(); assert.equal(h.get("step-count").textContent, "5 / 6");
    assert(h.get("step-title").textContent.includes(used)); assert.equal(h.calls.length, 2);
    assert.equal(h.get("plain-update").textContent, h.get("langgraph-update").textContent);
    assert(h.get("decision").textContent.includes(scenario === "with_evidence" ? "非空" : "为空"));
  });
}

test("the narrow graph keeps both branches and all arrows inside its SVG viewBox", async () => {
  const h = await run(); const svg = h.get("diagram").querySelector("svg");
  const viewBox = svg.getAttribute("viewBox").split(/\s+/).map(Number);
  assert.equal(viewBox.length, 4); assert(viewBox[2] > 0 && viewBox[3] > 0);
  const graphNodes = h.get("diagram").querySelectorAll(".graph-node");
  for (const node of graphNodes) {
    for (const rect of node.querySelectorAll("rect")) {
      const x = Number(rect.getAttribute("x")), y = Number(rect.getAttribute("y"));
      assert(x >= 0 && y >= 0); assert(x + Number(rect.getAttribute("width")) <= viewBox[2]);
      assert(y + Number(rect.getAttribute("height")) <= viewBox[3]);
    }
  }
  const branches = ["generate", "refuse"].map(id => graphNodes.find(node => node.dataset.node === id).querySelector("rect"));
  assert.notEqual(branches[0].getAttribute("x"), branches[1].getAttribute("x"));
  assert.equal(branches[0].getAttribute("y"), branches[1].getAttribute("y"));
});

test("narrow graph labels, IDs and execution status have separate readable baselines", async () => {
  const h = await run();
  for (const node of h.get("diagram").querySelectorAll(".graph-node")) {
    const lines = node.querySelectorAll("text");
    const baselines = lines.map(line => Number(line.getAttribute("y")));
    for (let index = 1; index < baselines.length; index++) {
      assert(baselines[index] - baselines[index - 1] >= 12, node.dataset.node);
    }
    const rect = node.querySelector("rect");
    assert(baselines.at(-1) < Number(rect.getAttribute("y")) + Number(rect.getAttribute("height")));
  }
});

test("graph construction consumes actual unordered edges, rather than assuming output order", async () => {
  const data = fixture(); data.graph.edges.reverse(); const h = await run({ postData: data });
  assert.equal(h.get("results").hidden, false);
  assert.equal(h.get("diagram").querySelectorAll(".graph-edge").length, 9);
});

test("step buttons and keyboard inspect local data only", async () => {
  const h = await run(); const count = h.calls.length;
  h.get("next-button").listeners.click(); assert.equal(h.get("step-count").textContent, "5 / 6");
  h.get("previous-button").listeners.click(); assert.equal(h.get("step-count").textContent, "4 / 6");
  h.eval("selectStep(0)"); assert.equal(h.get("previous-button").disabled, true);
  const generate = h.get("diagram").querySelectorAll(".graph-node").find(item => item.dataset.node === "generate");
  generate.listeners.keydown({ key: "Enter", preventDefault() {} });
  assert.equal(h.get("step-count").textContent, "5 / 6"); assert.equal(h.calls.length, count);
});

test("changing scenario immediately clears old answer, steps, graph and complete JSON without fetching", async () => {
  const h = await run(); const count = h.calls.length;
  h.scenario("empty_evidence");
  assert.equal(h.calls.length, count); assert.equal(h.eval("state.data"), null);
  assert.equal(h.get("results").hidden, true); assert.equal(h.get("diagram").querySelectorAll(".graph-node").length, 0);
  assert(!h.get("full-json").textContent.includes("demo-namespace")); assert.equal(h.get("plain-update").textContent, "");
  await h.eval("runReplay()"); assert.equal(JSON.parse(h.calls.at(-1).options.body).scenario, "empty_evidence");
  assert(h.get("decision").textContent.includes("为空"));
});

test("a scenario change invalidates an already pending response, so old results never flash into the new scenario", async () => {
  const h = harness({ deferPost: true }); await settle(); const pending = h.eval("runReplay()");
  h.scenario("empty_evidence"); h.finish(); await pending;
  assert.equal(h.eval("state.data"), null); assert.equal(h.get("results").hidden, true);
  assert.equal(h.get("scenario-select").value, "empty_evidence"); assert.equal(h.calls.length, 2);
  assert.equal(h.get("run-button").disabled, false);
});

test("an invalidated old response cannot replace a newer completed scenario", async () => {
  const h = harness({ deferPost: true }); await settle(); const oldRequest = h.eval("runReplay()");
  h.scenario("empty_evidence"); const newRequest = h.eval("runReplay()");
  assert.equal(h.calls.length, 3); h.finish(); await oldRequest;
  assert.equal(h.get("results").hidden, true); assert.equal(h.get("run-button").disabled, true);
  h.finish(); await newRequest;
  assert.equal(h.get("results").hidden, false); assert.equal(h.eval("state.data.scenario.id"), "empty_evidence");
  assert(h.get("decision").textContent.includes("refuse"));
});

test("pending request freezes the submitted scenario and double clicks do not add requests", async () => {
  const h = harness({ deferPost: true }); await settle(); const pending = h.eval("runReplay()");
  assert.equal(h.get("run-button").disabled, true); assert.equal(h.get("scenario-select").disabled, true);
  await h.eval("runReplay()"); assert.equal(h.calls.length, 2);
  assert.equal(JSON.parse(h.calls[1].options.body).scenario, "with_evidence");
  h.finish(); await pending; assert.equal(h.get("run-button").disabled, false); assert.equal(h.get("scenario-select").disabled, false);
});

test("HTTP failure clears the old state, resets controls and does not automatically retry", async () => {
  const h = await run({ failPost: true });
  assert.equal(h.eval("state.data"), null); assert.equal(h.get("results").hidden, true);
  assert.equal(h.get("error").hidden, false); assert.equal(h.get("run-button").disabled, false);
  assert.equal(h.get("scenario-select").disabled, false); assert.equal(h.calls.length, 2);
});

test("read-only definitions preserve literal hostile text without HTML interpretation", async () => {
  const data = plan(); data.definitions.router.code += '\n    # <svg onload="throw 1">'; data.definitions.router.end_line += 1;
  const h = harness({ planData: data }); await settle();
  assert(h.get("source-code").textContent.includes('<svg onload="throw 1">'));
  assert(h.get("source-location").textContent.includes("src/qa_platform/workflows/branching.py"));
  assert(h.get("fixture-title").textContent.includes("<img onerror")); assert.equal(h.calls.length, 1);
});

test("source selector shows actual router, Python branch and conditional builder without network", async () => {
  const h = harness(); await settle(); const count = h.calls.length;
  for (const [key, text] of [["plain", "if target"], ["graph_builder", "add_conditional_edges"], ["state", "total=False"]]) {
    h.get("source-select").value = key; h.get("source-select").listeners.change();
    assert(h.get("source-code").textContent.includes(text));
  }
  assert.equal(h.calls.length, count);
});

test("current-node source follows the executed refusal branch, not the skipped generate function", async () => {
  const h = harness(); await settle(); h.scenario("empty_evidence"); await h.eval("runReplay()");
  h.get("source-select").value = "node"; h.get("source-select").listeners.change();
  assert(h.get("source-code").textContent.includes("def prepare_evidence"));
  h.eval("selectStep(4)");
  assert(h.get("source-code").textContent.includes("def refuse"));
  assert(!h.get("source-code").textContent.includes("def generate")); assert.equal(h.calls.length, 2);
});

test("literal answer strings remain data when displayed in updates or complete JSON", async () => {
  const h = await run(); h.eval("selectStep(4)");
  assert(h.get("plain-update").textContent.includes("<script>throw 1</script>"));
  assert.equal(h.get("plain-update").children.length, 0);
  assert(h.get("full-json").textContent.includes("<img onerror=alert(1)>"));
  assert.equal(h.get("full-json").children.length, 0);
});

for (const [label, mutate] of [
  ["schema", data => { data.schema_version = "future"; }],
  ["nonzero HTTP count", data => { data.request_counts.http = 1; }],
  ["unknown scenario", data => { data.scenarios[0].id = "arbitrary"; }],
  ["missing router definition", data => { delete data.definitions.router; }],
  ["external source path", data => { data.definitions.state.file = "../../secrets.py"; }],
]) {
  test(`invalid plan ${label} disables execution without a POST`, async () => {
    const data = plan(); mutate(data); const h = harness({ planData: data }); await settle(); await h.eval("runReplay()");
    assert.equal(h.get("run-button").disabled, true); assert.equal(h.calls.length, 1);
    assert.equal(h.eval("state.plan"), null); assert.equal(h.get("error").hidden, false);
  });
}

for (const [label, mutate] of [
  ["schema", data => { data.schema_version = "future"; }],
  ["live origin", data => { data.fixture.origin = "live"; }],
  ["nonzero model count", data => { data.request_counts.chat = 1; }],
  ["wrong returned scenario", data => { data.scenario = clone(SCENARIOS[1]); }],
  ["missing branch node", data => { data.graph.nodes = data.graph.nodes.filter(node => node.id !== "refuse"); }],
  ["missing conditional edge", data => { data.graph.edges = data.graph.edges.filter(edge => edge.target !== "refuse"); }],
  ["fabricated branch edge", data => { data.graph.edges[4].target = "validate"; }],
  ["wrong conditional marker", data => { data.graph.edges[4].conditional = false; }],
  ["duplicate graph node", data => { data.graph.nodes[1].id = "__start__"; }],
  ["executed both branches", data => { data.langgraph.events.splice(5, 0, clone(data.langgraph.events[4])); }],
  ["wrong route decision", data => { data.langgraph.decisions[0].target = "refuse"; }],
  ["forged evidence count", data => { data.plain.decisions[0].evidence_count = 0; }],
  ["false comparison", data => { data.comparison.same_decisions = false; }],
  ["forged final-state equality", data => { data.langgraph.final_state.answer.claims[0].text = "different"; }],
  ["broken merge", data => { data.plain.events[0].after.extra = "not returned by update"; }],
  ["broken continuity", data => { data.plain.events[1].before.query = "different"; }],
  ["wrong initial query", data => { data.fixture.query = "different"; }],
  ["unknown source path", data => { data.definitions.router.file = "/etc/passwd"; }],
  ["bad source line range", data => { data.definitions.router.end_line = 1; }],
]) {
  test(`rejects ${label} without rendering a successful comparison`, async () => {
    const data = fixture(); mutate(data); const h = await run({ postData: data });
    assert.equal(h.eval("state.data"), null); assert.equal(h.get("results").hidden, true);
    assert.equal(h.get("error").hidden, false); assert.equal(h.calls.length, 2);
  });
}

test("missing dependency and plan failure disable replay without retry or fallback", async () => {
  const data = plan(); data.available = false; data.langgraph_version = null; data.dependency_message = "缺少验证依赖";
  const h = harness({ planData: data }); await settle(); await h.eval("runReplay()");
  assert.equal(h.get("run-button").disabled, true); assert.equal(h.calls.length, 1);
  assert(h.get("error").textContent.includes(data.dependency_message));
  const failed = harness({ failPlan: true }); await settle();
  assert.equal(failed.calls.length, 1); assert.equal(failed.get("run-button").disabled, true);
});

test("the existing linear lesson stays separate from the branching page and API", () => {
  const linear = fs.readFileSync(path.join(__dirname, "../web/workflow_lab.html"), "utf8");
  assert(linear.includes('"/api/workflow/plan"')); assert(linear.includes('"/api/workflow/replay"'));
  assert(!script.includes('fetch("/api/workflow/replay"'));
});
