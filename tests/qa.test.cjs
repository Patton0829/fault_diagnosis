"use strict";

// Synthetic API contracts only: no browser, model service, or real network is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const page = fs.readFileSync(path.join(__dirname, "../web/qa.html"), "utf8");
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const clone = value => JSON.parse(JSON.stringify(value));
const QUERY = "Pod 中 DNS 如何排查？";
const MODEL = "qwen3.8-flash";
const SOURCE = "https://github.com/kubernetes/website/blob/" + "a".repeat(40) + "/content/zh-cn/dns.md";

function plan() {
  return { schema_version: "rag-live-plan/v1", default_model: MODEL,
    models: [{ id: MODEL, base_url: "https://example.invalid/v1", transport_notice: "HTTPS 传输" },
      { id: "glm-4.7", base_url: "http://10.0.0.1:8000/v1", transport_notice: "内网服务，非加密连接" }],
    reranker: { model: "qwen-reranker", url: "https://example.invalid/rerank" },
    pipeline: ["retrieve", "rerank", "answer"], notice: "点击后会请求模型，可能产生费用。" };
}
function result() {
  const quote = "学习示例：检查命名空间 <img src=x onerror=alert(1)>。";
  return { schema_version: "rag-live/v1", query: QUERY, model: MODEL,
    steps: ["retrieve", "rerank", "answer"], elapsed_seconds: 0.75,
    request_counts: { embedding: 1, http: 5, reranker: 1, chat: 1 },
    answer: { status: "partial", claims: [{ text: "合成回答 <script>throw 1</script>",
      evidence: [{ source_id: "S1", quote_id: "Q1" }] }], missing_info: ["缺少现场信息 <svg onload=alert(1)>"] },
    sources: { S1: { chunk_id: "synthetic-dns", text: quote, quotes: { Q1: quote },
      metadata: { title: "合成文档 <img onerror=alert(1)>", section_path: ["DNS"], source_url: SOURCE,
        review_status: "unreviewed_official_document" } } } };
}
function agentPlan() {
  return { ...plan(), agent: { enabled_models: [MODEL], max_tool_calls: 2, notice: "只读知识库工具，无自动重试。",
    max_request_counts: { embedding: 2, reranker: 2, chat: 3, http: 11 } } };
}
function agentResult(count = 1) {
  const data = result(); data.mode = "agent";
  data.agent = { max_tool_calls: 2, tool_calls: Array.from({ length: count }, (_, i) => ({ id: "call-" + i,
    name: "search_knowledge", query: i ? "跨命名空间查询" : QUERY, evidence_ids: ["synthetic-dns"] })),
    planner_calls: count ? 2 : 1, stop_reason: count === 2 ? "tool_budget_reached" : "model_finished" };
  data.steps = count === 0 ? ["plan", "answer"] : count === 1 ? ["plan", "search", "plan", "answer"] : ["plan", "search", "plan", "search", "answer"];
  data.request_counts = { embedding: count, reranker: count, chat: count ? 3 : 1, http: count ? count * 4 + 3 : 1 };
  if (!count) { data.answer = { status: "insufficient_evidence", claims: [], missing_info: ["没有工具证据，不能作答。"] }; data.sources = {}; }
  return data;
}
const CONVERSATION_ID = "a".repeat(32);
function conversationPlan() {
  return { ...agentPlan(), conversation: { max_history_turns: 3, max_sessions: 16, extra_chat_requests: 1, notice: "历史仅帮助改写，不是事实依据。" } };
}
function conversationResult(turn = 1, data = result()) {
  const value = clone(data);
  value.conversation = { id: CONVERSATION_ID, turn, history_turns: Math.min(turn - 1, 3), retained_turns: Math.min(turn, 3), effective_query: turn > 1 ? "完整独立的 DNS 检索问题" : value.query, rewritten: turn > 1 };
  if (turn > 1) { value.steps.unshift("contextualize"); value.request_counts.chat++; value.request_counts.http++; }
  return value;
}
function streamPlan(base = plan()) {
  return { ...base, stream: { url: "/api/qa/stream", protocol: "qa-progress/v1", heartbeat_seconds: 10 } };
}
const STORAGE_KEY = "qa.conversation.mysql";
function mysqlPlan() {
  return { ...conversationPlan(), conversation: { ...conversationPlan().conversation, max_sessions: null }, storage: { backend: "mysql", persistent: true, notice: "本机数据库保存已完成问答。" } };
}
function savedConversation(turn = 3) {
  return { schema_version: "qa-conversation/v1", id: CONVERSATION_ID, turn,
    turns: Array.from({ length: Math.min(turn, 3) }, (_, index) => conversationResult(turn - Math.min(turn, 3) + index + 1)) };
}
const frame = (event, data, newline = "\n") => "event: " + event + newline + JSON.stringify(data, null, 2).split("\n").map(line => "data: " + line).join(newline) + newline + newline;
const progress = (stage, status = "started", newline = "\n") => frame("progress", { schema_version: "qa-progress/v1", stage, status }, newline);
function syntheticStream(chunks, { open = false, failure = null } = {}) {
  const encoder = new TextEncoder();
  let controller;
  const stream = new ReadableStream({ start(value) {
    controller = value;
    for (const chunk of chunks) controller.enqueue(typeof chunk === "string" ? encoder.encode(chunk) : chunk);
    if (failure) controller.error(new Error(failure));
    else if (!open) controller.close();
  } });
  const stats = { cancelled: 0, released: 0 };
  return { response: { headers: { get: () => "text/event-stream; charset=utf-8" }, body: { getReader() {
    const reader = stream.getReader();
    return { read: () => reader.read(), cancel: async () => { stats.cancelled++; await reader.cancel(); }, releaseLock: () => { stats.released++; reader.releaseLock(); } };
  } } }, stats, push: value => controller.enqueue(typeof value === "string" ? encoder.encode(value) : value), close: () => controller.close() };
}

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this._text = ""; this.value = "";
    this.listeners = {}; this.hidden = false; this.disabled = false; this.focused = false; this.checked = false;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = children; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  focus() { this.focused = true; }
  querySelectorAll(tag) {
    return this.children.flatMap(child => [
      ...(child.tagName === tag.toUpperCase() ? [child] : []), ...child.querySelectorAll(tag),
    ]);
  }
}
function harness({ initial = plan(), response = result(), getFailure = false,
  postFailure = null, deferred = false, responseStatuses = [], resetFailure = false, streams = [], storedId,
  storageFailure = null, loadResponse = savedConversation(), loadStatus = 200, loadDeferred = false } = {}) {
  const elements = new Map([...page.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)].map(match => {
    const element = new Element(match[1]); element.hidden = /\bhidden\b/.test(match[2]);
    element.disabled = /\bdisabled\b/.test(match[2]); return [match[3], element];
  }));
  const calls = [], pending = [], storageCalls = [], storageValues = new Map(storedId === undefined ? [] : [[STORAGE_KEY, storedId]]); let postIndex = 0;
  const browserStorage = Object.fromEntries(["getItem", "setItem", "removeItem"].map(method => [method, (key, value) => {
    storageCalls.push({ method, key, value });
    if (storageFailure === method || storageFailure === "all") throw new Error("synthetic browser storage restriction");
    if (method === "getItem") return storageValues.has(key) ? storageValues.get(key) : null;
    if (method === "setItem") storageValues.set(key, value);
    else storageValues.delete(key);
  }]));
  const fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === "/api/qa/plan") return { ok: !getFailure, status: getFailure ? 503 : 200, json: async () => clone(initial) };
    if (url === "/api/qa/conversation/reset") return { ok: !resetFailure, status: resetFailure ? 409 : 200, json: async () => ({ cleared: true }) };
    if (url === "/api/qa/conversation/load") {
      if (loadDeferred) await new Promise(resolve => pending.push(resolve));
      return { ok: loadStatus === 200, status: loadStatus, json: async () => clone(loadResponse) };
    }
    assert.equal(url, initial.stream ? "/api/qa/stream" : "/api/qa");
    const currentIndex = postIndex++;
    if (deferred) await new Promise(resolve => pending.push(resolve));
    if (postFailure === "network") throw new Error("synthetic transport failure");
    const status = postFailure === "http" ? 503 : responseStatuses[currentIndex] || 200;
    return { ok: status >= 200 && status < 300, status, ...(streams[currentIndex]?.response || {}),
      json: async () => { if (postFailure === "json") throw new Error("synthetic invalid JSON");
        return clone(postFailure === "http" ? { detail: "本次生成失败，未得到校验通过的回答。" } : Array.isArray(response) ? response[currentIndex] : response); } };
  };
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: tag => new Element(tag) }, fetch, URL, TextDecoder, sessionStorage: browserStorage });
  vm.runInContext(script, context);
  return { calls, storageCalls, storageValues, get: id => elements.get(id), eval: code => vm.runInContext(code, context),
    reset: () => elements.get("new-conversation").listeners.click(), finish: () => pending.shift()(), submit: () => {
      let prevented = false;
      const promise = elements.get("question-form").listeners.submit({ preventDefault() { prevented = true; } });
      assert(prevented); return promise;
    } };
}
async function ready(options = {}) {
  const h = harness(options); await new Promise(resolve => setImmediate(resolve));
  h.get("query").value = QUERY; return h;
}

test("the compact page discloses cost, recipients and no retries without HTML execution sinks", () => {
  assert(!page.includes("innerHTML")); assert(!page.includes("insertAdjacentHTML"));
  assert(!page.match(/<script[^>]+src=/)); assert(!page.match(/<link[^>]+href="https?:/));
  assert(page.includes("可能产生费用")); assert(page.includes("不自动重试"));
  assert(page.includes("重排接收问题与候选正文")); assert(page.includes("HTTP 明文传输"));
  assert(page.includes('role="status"')); assert(page.includes('role="alert"'));
});

test("opening the page only gets the plan; switching models makes no model request", async () => {
  const h = await ready(); assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, "/api/qa/plan"); assert.equal(h.calls[0].options.method, "GET");
  assert.equal(h.calls[0].options.mode, "same-origin"); assert.equal(h.get("ask").disabled, false);
  assert(h.get("generation-destination").textContent.includes("https://example.invalid/v1"));
  assert(h.get("reranker-destination").textContent.includes("https://example.invalid/rerank"));
  h.get("model").value = "glm-4.7"; h.get("model").listeners.change();
  assert(h.get("generation-destination").textContent.includes("http://10.0.0.1:8000/v1"));
  assert(h.get("transport-notice").textContent.includes("HTTP 明文传输")); assert.equal(h.calls.length, 1);
});

test("one explicit form submit sends one same-origin guarded request and renders citations", async () => {
  const h = await ready(); h.get("query").value = "  " + QUERY + "  "; await h.submit();
  assert.equal(h.calls.length, 2); const request = h.calls[1];
  assert.equal(request.url, "/api/qa"); assert.equal(request.options.method, "POST");
  assert.equal(request.options.mode, "same-origin"); assert.equal(request.options.credentials, "same-origin");
  assert.equal(request.options.headers["X-Retrieval-Lab"], "1");
  assert.equal(request.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(request.options.body), { query: QUERY, model: MODEL, confirmed: true });
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("error").hidden, true);
  assert(h.get("claims").textContent.includes("[S1/Q1]"));
  assert(h.get("claims").textContent.includes(result().sources.S1.quotes.Q1));
  assert(h.get("run-summary").textContent.includes("Embedding 1 / Reranker 1 / 聊天 1"));
  const link = h.get("sources").querySelectorAll("a")[0];
  assert.equal(link.href, SOURCE); assert.equal(link.rel, "noopener noreferrer");
});

test("pending execution clears the old answer and prevents repeated submits", async () => {
  const h = await ready({ deferred: true }); h.eval(`renderAnswer(${JSON.stringify(result())})`);
  assert.equal(h.get("result").hidden, false); const pending = h.submit();
  assert.equal(h.get("result").hidden, true); assert.equal(h.get("claims").textContent, "");
  for (const id of ["query", "model", "ask"]) assert.equal(h.get(id).disabled, true);
  await h.submit(); await h.submit(); assert.equal(h.calls.length, 2);
  h.finish(); await pending; assert.equal(h.get("result").hidden, false);
  assert.equal(h.get("ask").disabled, false); assert.equal(h.eval("state.busy"), false);
});

test("transport, HTTP, malformed answers and invalid citations fail closed without retry or stale answers", async () => {
  const invalid = result(); invalid.answer.claims[0].evidence[0].quote_id = "Q999";
  const emptyQuote = result(); emptyQuote.sources.S1.quotes.Q1 = " ";
  const inventedQuote = result(); inventedQuote.sources.S1.quotes.Q1 = "不是原文中的摘录";
  const emptyAnswer = result(); emptyAnswer.answer.claims = [];
  const emptyRefusal = result(); emptyRefusal.answer = { status: "insufficient_evidence", claims: [], missing_info: [] };
  const vaguePartial = result(); vaguePartial.answer.missing_info = [];
  for (const options of [{ postFailure: "network" }, { postFailure: "http" }, { postFailure: "json" },
    ...[invalid, emptyQuote, inventedQuote, emptyAnswer, emptyRefusal, vaguePartial].map(response => ({ response }))]) {
    const h = await ready(options); h.eval(`renderAnswer(${JSON.stringify(result())})`); await h.submit();
    assert.equal(h.calls.length, 2); assert.equal(h.get("result").hidden, true);
    assert.equal(h.get("claims").textContent, ""); assert.equal(h.get("sources").textContent, "");
    assert.equal(h.get("error").hidden, false); assert(h.get("error").textContent.includes("不自动重试"));
    assert(h.get("status").textContent.includes("未完成")); assert.equal(h.get("refusal").hidden, true);
    assert.equal(h.get("ask").disabled, false); assert.equal(h.eval("state.busy"), false);
  }
});

test("failed or malformed configuration keeps submit disabled and does not start the pipeline", async () => {
  for (const options of [{ getFailure: true }, { initial: { ...plan(), schema_version: "wrong/v1" } }]) {
    const h = await ready(options); assert.equal(h.get("ask").disabled, true); await h.submit();
    assert.equal(h.calls.length, 1); assert.equal(h.get("error").hidden, false); assert.equal(h.get("result").hidden, true);
  }
});

test("insufficient evidence is a normal successful result, not a model execution failure", async () => {
  const response = result(); response.answer = { status: "insufficient_evidence", claims: [], missing_info: ["知识库未覆盖此问题。"] }; response.sources = {};
  const h = await ready({ response }); await h.submit();
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("refusal").hidden, false);
  assert.equal(h.get("answer-status").textContent, "证据不足"); assert.equal(h.get("error").hidden, true);
  assert(h.get("missing-list").textContent.includes("知识库未覆盖")); assert.equal(h.calls.length, 2);
});

test("claims, quotes and source metadata are literal text; unsafe source URLs never become links", async () => {
  const h = await ready(); await h.submit();
  assert(h.get("claims").textContent.includes("<script>throw 1</script>"));
  assert(h.get("claims").textContent.includes("<img src=x onerror=alert(1)>"));
  assert(h.get("sources").textContent.includes("<img onerror=alert(1)>"));
  for (const id of ["claims", "missing-list", "sources"]) {
    for (const tag of ["script", "img", "svg"]) assert.equal(h.get(id).querySelectorAll(tag).length, 0);
  }
  for (const url of ["javascript:alert(1)", "data:text/html,<script>1</script>", "http://github.com/kubernetes/website/blob/a/doc.md",
    "https://github.com.evil.invalid/kubernetes/website/blob/a/doc.md", "https://user@github.com/kubernetes/website/blob/a/doc.md"]) {
    const source = result().sources; source.S1.metadata.source_url = url;
    h.get("sources").replaceChildren(); h.eval(`showSources(${JSON.stringify(source)})`);
    assert.equal(h.get("sources").querySelectorAll("a").length, 0); assert(h.get("sources").textContent.includes(url));
  }
  assert.equal(h.calls.length, 2);
});

test("empty, overlong and unknown-model inputs are rejected before any model request", async () => {
  for (const [query, model] of [["   ", MODEL], ["x".repeat(2001), MODEL], [QUERY, "unknown-model"]]) {
    const h = await ready(); h.get("query").value = query; h.get("model").value = model; await h.submit();
    assert.equal(h.calls.length, 1); assert.equal(h.get("error").hidden, false); assert.equal(h.get("result").hidden, true);
  }
});

test("multi-step is opt-in and an older plan remains a usable single-run plan", async () => {
  const old = await ready(); assert.equal(old.get("agent-mode").checked, false); assert.equal(old.get("agent-mode").disabled, true);
  await old.submit(); assert.equal(JSON.parse(old.calls[1].options.body).mode, undefined);
  const h = await ready({ initial: agentPlan() }); assert.equal(h.get("agent-mode").checked, false);
  assert.equal(h.get("agent-mode").disabled, false); assert.equal(h.calls.length, 1);
  await h.submit(); assert.deepEqual(JSON.parse(h.calls[1].options.body), { query: QUERY, model: MODEL, confirmed: true });
});

test("selecting multi-step discloses its extra recipient and request budget without sending a request", async () => {
  const h = await ready({ initial: agentPlan() }); h.get("agent-mode").checked = true; h.get("agent-mode").listeners.change();
  const notice = h.get("agent-notice").textContent;
  for (const fragment of ["工具定义", "子查询", "前5块正文", "必要审核标记", "不发送本机路径或向量", "HTTP 最多11次", "不是多轮聊天", "额外费用"]) assert(notice.includes(fragment));
  assert.equal(h.calls.length, 1);
});

test("one opted-in submit sends agent mode and displays only actual search queries", async () => {
  for (const count of [0, 1, 2]) {
    const h = await ready({ initial: agentPlan(), response: agentResult(count) }); h.get("agent-mode").checked = true; await h.submit();
    assert.deepEqual(JSON.parse(h.calls[1].options.body), { query: QUERY, model: MODEL, confirmed: true, mode: "agent" });
    assert.equal(h.get("result").hidden, false); assert.equal(h.get("error").hidden, true);
    assert(h.get("run-summary").textContent.startsWith("多步检索流程已结束"));
    assert(h.get("run-summary").textContent.includes("工具检索 " + count + "/2，决策 " + (count ? 2 : 1) + "次"));
    assert.equal(h.get("agent-searches").hidden, count === 0); assert.equal(h.get("agent-query-list").children.length, count);
    if (count) assert(h.get("agent-query-list").textContent.includes(QUERY));
    assert.equal(h.calls.length, 2);
  }
});

test("switching away from Flash disables and clears the multi-step choice without requests", async () => {
  const h = await ready({ initial: agentPlan() }); h.get("agent-mode").checked = true;
  h.get("model").value = "glm-4.7"; h.get("model").listeners.change();
  assert.equal(h.get("agent-mode").checked, false); assert.equal(h.get("agent-mode").disabled, true);
  assert(h.get("agent-notice").textContent.includes("暂仅支持 qwen3.8-flash"));
  h.get("model").value = MODEL; h.get("model").listeners.change();
  assert.equal(h.get("agent-mode").checked, false); assert.equal(h.get("agent-mode").disabled, false); assert.equal(h.calls.length, 1);
});

test("an unsupported programmatic multi-step choice is rejected before POST", async () => {
  const h = await ready({ initial: agentPlan() }); h.get("model").value = "glm-4.7"; h.get("agent-mode").checked = true;
  await h.submit(); assert.equal(h.calls.length, 1); assert.equal(h.get("error").hidden, false);
});

test("pending multi-step disables its checkbox, prevents duplicates and clears old search details", async () => {
  const h = await ready({ initial: agentPlan(), response: agentResult(2), deferred: true });
  h.eval(`renderAnswer(${JSON.stringify(agentResult(1))})`); h.get("agent-mode").checked = true;
  const pending = h.submit(); assert.equal(h.get("agent-mode").disabled, true);
  assert.equal(h.get("agent-searches").hidden, true); assert.equal(h.get("agent-query-list").textContent, "");
  for (const id of ["query", "model", "ask"]) assert.equal(h.get(id).disabled, true);
  await h.submit(); assert.equal(h.calls.length, 2); h.finish(); await pending;
  assert.equal(h.get("agent-mode").disabled, false); assert.equal(h.get("result").hidden, false);
});

test("agent response contracts reject wrong modes, sequences, budgets and tool metadata without stale output", async () => {
  const cases = [result()];
  const edit = mutator => { const data = agentResult(1); mutator(data); cases.push(data); };
  edit(data => { delete data.mode; });
  edit(data => { data.steps = ["plan", "answer"]; });
  edit(data => { data.agent.max_tool_calls = 3; });
  edit(data => { data.agent.planner_calls = 1; });
  edit(data => { data.agent.stop_reason = "tool_budget_reached"; });
  edit(data => { data.agent.tool_calls[0].name = "run_shell"; });
  edit(data => { data.agent.tool_calls[0].query = " "; });
  edit(data => { data.agent.tool_calls[0].evidence_ids = [""]; });
  edit(data => { data.agent.tool_calls.push(clone(data.agent.tool_calls[0])); });
  for (const key of ["embedding", "reranker", "chat", "http"]) edit(data => { data.request_counts[key] = { embedding: 3, reranker: 3, chat: 4, http: 12 }[key]; });
  const unsupportedAnswer = agentResult(0); unsupportedAnswer.answer = result().answer; unsupportedAnswer.sources = result().sources; cases.push(unsupportedAnswer);
  for (const response of cases) {
    const h = await ready({ initial: agentPlan(), response }); h.eval(`renderAnswer(${JSON.stringify(agentResult(1))})`);
    h.get("agent-mode").checked = true; await h.submit();
    assert.equal(h.calls.length, 2); assert.equal(h.get("result").hidden, true); assert.equal(h.get("error").hidden, false);
    assert.equal(h.get("agent-query-list").textContent, ""); assert.equal(h.get("claims").textContent, "");
    assert(h.get("error").textContent.includes("不自动重试"));
  }
});

test("an agent result cannot impersonate an unselected single-run response", async () => {
  const h = await ready({ initial: agentPlan(), response: agentResult(1) }); await h.submit();
  assert.equal(h.get("result").hidden, true); assert.equal(h.get("error").hidden, false);
});

test("agent search query strings remain literal text", async () => {
  const response = agentResult(1); response.agent.tool_calls[0].query = "<img src=x onerror=alert(1)>";
  const h = await ready({ initial: agentPlan(), response }); h.get("agent-mode").checked = true; await h.submit();
  assert(h.get("agent-query-list").textContent.includes("<img src=x onerror=alert(1)>"));
  assert.equal(h.get("agent-query-list").querySelectorAll("img").length, 0);
});

test("conversation starts without a client ID and follow-ups send only the server ID, never history", async () => {
  const h = await ready({ initial: conversationPlan(), response: [conversationResult(1), conversationResult(2)] });
  assert.equal(h.calls.length, 1); assert.equal(h.get("new-conversation").hidden, false);
  assert(h.get("conversation-notice").textContent.includes("最多3轮先前问答摘要"));
  assert(h.get("conversation-notice").textContent.includes("切换模型后历史也会发送给新服务"));
  await h.submit();
  assert.deepEqual(JSON.parse(h.calls[1].options.body), { query: QUERY, model: MODEL, confirmed: true, conversation: true });
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("conversation-history").hidden, true);
  await h.submit();
  assert.deepEqual(JSON.parse(h.calls[2].options.body), { query: QUERY, model: MODEL, confirmed: true, conversation: true, conversation_id: CONVERSATION_ID });
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("history-list").children.length, 1);
  assert(h.get("history-list").textContent.includes("历史回答摘要"));
  assert(h.get("effective-query-text").textContent.includes("完整独立的 DNS 检索问题"));
  assert.equal(h.get("effective-query").hidden, false); assert.equal(h.calls.length, 3);
});

test("conversation and multi-step combine with exactly one extra context request budget", async () => {
  const h = await ready({ initial: conversationPlan(), response: [conversationResult(1, agentResult(2)), conversationResult(2, agentResult(2))] });
  h.get("agent-mode").checked = true; await h.submit();
  h.get("agent-mode").listeners.change(); assert(h.get("agent-notice").textContent.includes("HTTP 最多12次"));
  await h.submit(); assert.equal(h.get("result").hidden, false); assert.equal(h.get("error").hidden, true);
  assert(h.get("run-summary").textContent.includes("聊天 4")); assert(h.get("run-summary").textContent.includes("工具检索 2/2"));
  assert.equal(JSON.parse(h.calls[2].options.body).mode, "agent");
});

test("new conversation uses a guarded idempotent reset before clearing local history without asking", async () => {
  const h = await ready({ initial: conversationPlan(), response: [conversationResult(1), conversationResult(2), conversationResult(1)] });
  await h.submit(); await h.submit(); await h.reset();
  const reset = h.calls[3]; assert.equal(reset.url, "/api/qa/conversation/reset"); assert.equal(reset.options.method, "POST");
  assert.equal(reset.options.mode, "same-origin"); assert.equal(reset.options.credentials, "same-origin");
  assert.equal(reset.options.headers["X-Retrieval-Lab"], "1"); assert.equal(reset.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(reset.options.body), { conversation_id: CONVERSATION_ID });
  assert.equal(h.eval("state.conversationId"), null); assert.equal(h.get("history-list").textContent, ""); assert.equal(h.get("result").hidden, true);
  assert.equal(h.calls.length, 4); await h.submit(); assert.equal(JSON.parse(h.calls[4].options.body).conversation_id, undefined);
});

test("failed reset retains server ID and prior local history; busy requests disable reset", async () => {
  const h = await ready({ initial: conversationPlan(), response: [conversationResult(1), conversationResult(2)], resetFailure: true });
  await h.submit(); await h.submit(); const prior = h.get("history-list").textContent; await h.reset();
  assert.equal(h.eval("state.conversationId"), CONVERSATION_ID); assert.equal(h.get("history-list").textContent, prior);
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("error").hidden, false);
  const busy = await ready({ initial: conversationPlan(), response: conversationResult(1), deferred: true });
  const pending = busy.submit(); assert.equal(busy.get("new-conversation").disabled, true); await busy.reset();
  assert.equal(busy.calls.length, 2); busy.finish(); await pending;
});

test("expired conversation prompts explicit new conversation and never retries or appends failed history", async () => {
  const h = await ready({ initial: conversationPlan(), response: [conversationResult(1), {}], responseStatuses: [200, 410] });
  await h.submit(); await h.submit(); assert.equal(h.get("result").hidden, true); assert.equal(h.get("history-list").textContent, "");
  assert(h.get("error").textContent.includes("新对话")); assert.equal(h.get("ask").disabled, true);
  await h.submit(); assert.equal(h.calls.length, 3); await h.reset();
  assert.equal(h.get("ask").disabled, false); assert.equal(h.eval("state.conversationId"), null);
});

test("bad conversation identities, turn counts or context stages cannot replace a verified turn", async () => {
  const mutations = [data => { delete data.conversation; }, data => { data.conversation.id = "b".repeat(32); },
    data => { data.conversation.turn = 8; }, data => { data.conversation.history_turns = 0; },
    data => { data.conversation.retained_turns = 9; }, data => { data.conversation.effective_query = " "; },
    data => { data.steps.shift(); }, data => { data.conversation.rewritten = "yes"; }];
  for (const mutate of mutations) {
    const bad = conversationResult(2); mutate(bad);
    const h = await ready({ initial: conversationPlan(), response: [conversationResult(1), bad] });
    await h.submit(); await h.submit(); assert.equal(h.get("result").hidden, true); assert.equal(h.get("error").hidden, false);
    assert.equal(h.eval("state.turn"), 1); assert.equal(h.get("history-list").textContent, ""); assert.equal(h.calls.length, 3);
  }
});

test("recent history is bounded, literal text and absent from a freshly opened page", async () => {
  const responses = Array.from({ length: 5 }, (_, index) => conversationResult(index + 1));
  const h = await ready({ initial: conversationPlan(), response: responses });
  for (let i = 0; i < 5; i++) await h.submit();
  assert.equal(h.get("history-list").children.length, 3); assert.equal(h.eval("state.turn"), 5);
  assert(h.get("history-list").textContent.includes("<script>throw 1</script>")); assert.equal(h.get("history-list").querySelectorAll("script").length, 0);
  assert(!page.includes("localStorage")); assert.equal(h.storageCalls.length, 0);
  const refreshed = await ready({ initial: conversationPlan() }); assert.equal(refreshed.eval("state.conversationId"), null);
  assert.equal(refreshed.get("history-list").textContent, ""); assert.equal(refreshed.calls.length, 1);
});

test("streaming POST updates real stage status before completion and keeps old answers hidden until EOF", async () => {
  const stream = syntheticStream([progress("retrieve"), progress("retrieve", "completed"), progress("answer")], { open: true });
  const h = await ready({ initial: streamPlan(conversationPlan()), streams: [stream] });
  const pending = h.submit(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.calls[1].url, "/api/qa/stream"); assert.equal(h.calls[1].options.method, "POST");
  assert.equal(h.calls[1].options.headers["X-Retrieval-Lab"], "1"); assert.equal(h.calls[1].options.credentials, "same-origin");
  assert(h.get("status").textContent.includes("生成与校验回答中…")); assert(h.get("status").textContent.includes("最近完成：混合检索"));
  assert.equal(h.get("result").hidden, true); assert.equal(h.get("ask").disabled, true); assert.equal(h.get("new-conversation").disabled, true);
  await h.submit(); await h.reset(); assert.equal(h.calls.length, 2);
  stream.push(frame("result", conversationResult(1))); await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.get("result").hidden, true); assert.equal(h.eval("state.turn"), 0);
  stream.close(); await pending;
  assert.equal(h.get("result").hidden, false); assert.equal(h.eval("state.turn"), 1); assert.equal(h.get("ask").disabled, false);
  assert.equal(stream.stats.cancelled, 0); assert.equal(stream.stats.released, 1);
});

test("SSE handles fragmented UTF-8, CRLF, comments and multiline data without treating heartbeats as progress", async () => {
  const content = ": keep-alive\r\n\r\n" + progress("rerank", "started", "\r\n") + frame("result", result(), "\r\n");
  const bytes = new TextEncoder().encode(content);
  const stream = syntheticStream(Array.from(bytes, byte => Uint8Array.of(byte)));
  const h = await ready({ initial: streamPlan(), streams: [stream] }); await h.submit();
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("error").hidden, true);
  assert(h.get("claims").textContent.includes("合成回答")); assert.equal(stream.stats.released, 1);
  assert(!h.get("status").textContent.includes("keep-alive")); assert.equal(h.calls.length, 2);
});

test("a heartbeat alone changes no stage and a stream without a terminal never produces an answer", async () => {
  const stream = syntheticStream([": keep-alive\n\n"], { open: true });
  const h = await ready({ initial: streamPlan(), streams: [stream] }); const pending = h.submit();
  await new Promise(resolve => setImmediate(resolve));
  assert(h.get("status").textContent.includes("正在自动完成")); stream.close(); await pending;
  assert.equal(h.get("result").hidden, true); assert.equal(h.get("error").hidden, false);
  assert.equal(stream.stats.cancelled, 1); assert.equal(stream.stats.released, 1); assert.equal(h.calls.length, 2);
});

test("SSE rejects duplicate terminals, events after terminal, unknown events, bad JSON and truncated frames without retry", async () => {
  const cases = [
    frame("result", result()) + frame("result", result()),
    frame("result", result()) + progress("answer", "completed"),
    frame("unknown", { detail: "not accepted" }),
    "event: progress\ndata: not-json-secret-marker\n\n",
    frame("progress", { schema_version: "wrong/v1", stage: "answer", status: "started" }),
    progress("execute_shell"),
    "event: result\ndata: " + JSON.stringify(result()),
    "x".repeat(256 * 1024 + 1),
    frame("error", { detail: "回答或引用校验未完成，不自动重试。" }),
  ];
  for (const content of cases) {
    const stream = syntheticStream([content]); const h = await ready({ initial: streamPlan(conversationPlan()), streams: [stream] });
    h.eval(`renderAnswer(${JSON.stringify(result())})`); await h.submit();
    assert.equal(h.get("result").hidden, true); assert.equal(h.get("claims").textContent, "");
    assert.equal(h.eval("state.turn"), 0); assert.equal(h.get("error").hidden, false);
    assert(!h.get("error").textContent.includes("not-json-secret-marker"));
    assert(h.get("error").textContent.includes("断线不代表服务端已停止"));
    assert.equal(h.calls.length, 2); assert.equal(stream.stats.cancelled, 1); assert.equal(stream.stats.released, 1);
  }
});

test("SSE invalid final answers and transport failures cannot update conversation history", async () => {
  const invalid = conversationResult(1); invalid.answer.claims[0].evidence[0].quote_id = "Q999";
  const failedStreams = [syntheticStream([frame("result", invalid)]), syntheticStream([], { failure: "synthetic disconnection" })];
  for (const stream of failedStreams) {
    const h = await ready({ initial: streamPlan(conversationPlan()), streams: [stream] }); await h.submit();
    assert.equal(h.get("result").hidden, true); assert.equal(h.eval("state.turn"), 0); assert.equal(h.eval("state.conversationId"), null);
    assert.equal(h.get("error").hidden, false); assert.equal(h.calls.length, 2); assert.equal(stream.stats.released, 1);
    assert.equal(stream.stats.cancelled, 1);
  }
});

test("SSE follow-up preserves context, agent budgets and explicit new-conversation reset", async () => {
  const streams = [syntheticStream([progress("plan"), frame("result", conversationResult(1, agentResult(2)))]),
    syntheticStream([progress("contextualize"), progress("contextualize", "completed"), frame("result", conversationResult(2, agentResult(2)))])];
  const h = await ready({ initial: streamPlan(conversationPlan()), streams }); h.get("agent-mode").checked = true;
  await h.submit(); await h.submit(); assert.equal(h.get("result").hidden, false); assert.equal(h.eval("state.turn"), 2);
  assert.equal(h.get("history-list").children.length, 1); assert(h.get("run-summary").textContent.includes("聊天 4"));
  assert.equal(JSON.parse(h.calls[2].options.body).conversation_id, CONVERSATION_ID); await h.reset();
  assert.equal(h.calls[3].url, "/api/qa/conversation/reset"); assert.equal(h.eval("state.turn"), 0);
});

test("stream guard failures stay JSON HTTP errors and never open a stream or retry", async () => {
  const h = await ready({ initial: streamPlan(conversationPlan()), responseStatuses: [410] });
  await h.submit(); assert.equal(h.calls.length, 2); assert.equal(h.get("result").hidden, true);
  assert(h.get("error").textContent.includes("新对话")); assert.equal(h.get("ask").disabled, true);
});

test("memory storage never touches browser storage, even when it is unavailable", async () => {
  const memory = { ...conversationPlan(), storage: { backend: "memory", persistent: false, notice: "仅服务内存。" } };
  for (const initial of [conversationPlan(), memory]) {
    const h = await ready({ initial, storedId: CONVERSATION_ID, storageFailure: "all", response: conversationResult(1) });
    await h.submit(); await h.reset(); assert.equal(h.storageCalls.length, 0);
    assert(!h.calls.some(call => call.url.endsWith("/load"))); assert.equal(h.get("error").hidden, true);
  }
});

test("mysql mode stores only the successful ID and reset retains database-history disclosure", async () => {
  const h = await ready({ initial: mysqlPlan(), response: conversationResult(1) });
  assert.equal(h.calls.length, 1); assert(h.get("storage-notice").textContent.includes("本机 MySQL"));
  await h.submit(); assert.equal(h.get("result").hidden, false);
  assert.equal(h.storageValues.get(STORAGE_KEY), CONVERSATION_ID);
  const writes = h.storageCalls.filter(call => call.method === "setItem");
  assert.deepEqual(writes, [{ method: "setItem", key: STORAGE_KEY, value: CONVERSATION_ID }]);
  assert(h.get("conversation-notice").textContent.includes("不删除已保存问答"));
  await h.reset(); assert.equal(h.storageValues.has(STORAGE_KEY), false); assert.equal(h.calls.length, 3);
});

test("mysql refresh restores a fully validated tail without model requests or changing the selected model", async () => {
  const loaded = savedConversation(5); loaded.turns[2].model = "glm-4.7";
  const h = await ready({ initial: mysqlPlan(), storedId: CONVERSATION_ID, loadResponse: loaded, response: conversationResult(6) });
  assert.equal(h.calls.length, 2); const request = h.calls[1]; assert.equal(request.url, "/api/qa/conversation/load");
  assert.equal(request.options.method, "POST"); assert.equal(request.options.mode, "same-origin");
  assert.equal(request.options.headers["X-Retrieval-Lab"], "1"); assert.equal(request.options.credentials, "same-origin");
  assert.deepEqual(JSON.parse(request.options.body), { conversation_id: CONVERSATION_ID });
  assert.equal(h.eval("state.turn"), 5); assert.equal(h.eval("state.conversationId"), CONVERSATION_ID);
  assert.equal(h.get("model").value, MODEL); assert.equal(h.get("result").hidden, false); assert.equal(h.get("history-list").children.length, 2);
  assert(h.get("status").textContent.includes("未重新检索或调用模型"));
  await h.submit(); assert.equal(JSON.parse(h.calls[2].options.body).conversation_id, CONVERSATION_ID);
  assert.equal(h.get("result").hidden, false); assert.equal(h.eval("state.turn"), 6);
});

test("restoring mysql history disables controls until all rows validate, with no partial answer", async () => {
  const h = await ready({ initial: mysqlPlan(), storedId: CONVERSATION_ID, loadDeferred: true });
  for (const id of ["query", "model", "ask", "new-conversation", "agent-mode"]) assert.equal(h.get(id).disabled, true);
  await h.submit(); await h.reset(); assert.equal(h.calls.length, 2); assert.equal(h.get("result").hidden, true);
  h.finish(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.get("result").hidden, false); assert.equal(h.get("ask").disabled, false);
});

test("malformed restored records fail closed and keep the ID until explicit reset", async () => {
  const mutations = [data => { data.id = "b".repeat(32); }, data => { data.turn++; },
    data => { data.turns[0].conversation.turn = 9; }, data => { data.turns[0].conversation.id = "b".repeat(32); },
    data => { data.turns[1].model = "unlisted-model"; }, data => { data.turns[1].answer.claims[0].evidence[0].quote_id = "Q999"; },
    data => { data.turns[1].conversation.effective_query = " "; }];
  for (const mutate of mutations) {
    const loaded = savedConversation(); mutate(loaded);
    const h = await ready({ initial: mysqlPlan(), storedId: CONVERSATION_ID, loadResponse: loaded });
    assert.equal(h.get("result").hidden, true); assert.equal(h.get("history-list").textContent, "");
    assert.equal(h.get("ask").disabled, true); assert.equal(h.get("new-conversation").disabled, false);
    assert.equal(h.storageValues.get(STORAGE_KEY), CONVERSATION_ID); assert.equal(h.calls.length, 2);
    await h.submit(); assert.equal(h.calls.length, 2); await h.reset(); assert.equal(h.storageValues.has(STORAGE_KEY), false);
  }
});

test("mysql 410 clears the old marker but 503 preserves it without silently starting a new session", async () => {
  const expired = await ready({ initial: mysqlPlan(), storedId: CONVERSATION_ID, loadStatus: 410 });
  assert.equal(expired.storageValues.has(STORAGE_KEY), false); assert.equal(expired.eval("state.conversationId"), null);
  assert.equal(expired.get("ask").disabled, false); assert(expired.get("status").textContent.includes("已失效")); assert.equal(expired.calls.length, 2);
  const unavailable = await ready({ initial: mysqlPlan(), storedId: CONVERSATION_ID, loadStatus: 503 });
  assert.equal(unavailable.storageValues.get(STORAGE_KEY), CONVERSATION_ID); assert.equal(unavailable.get("ask").disabled, true);
  assert(unavailable.get("error").textContent.includes("未静默开启新会话")); await unavailable.submit(); assert.equal(unavailable.calls.length, 2);
});

test("invalid stored IDs never reach the server and browser-storage failures never hide successful answers", async () => {
  const malformed = await ready({ initial: mysqlPlan(), storedId: "<script>untrusted</script>" });
  assert.equal(malformed.calls.length, 1); assert.equal(malformed.storageValues.has(STORAGE_KEY), false);
  const unreadable = await ready({ initial: mysqlPlan(), storageFailure: "getItem" });
  assert.equal(unreadable.calls.length, 1); assert(unreadable.get("storage-notice").textContent.includes("无法自动恢复"));
  const unwritable = await ready({ initial: mysqlPlan(), response: conversationResult(1), storageFailure: "setItem" });
  await unwritable.submit(); assert.equal(unwritable.get("result").hidden, false); assert.equal(unwritable.get("error").hidden, true);
  assert(unwritable.get("storage-notice").textContent.includes("当前回答不受影响")); assert.equal(unwritable.calls.length, 2);
});

test("only an explicitly consistent persistent plan permits unbounded database-session metadata", async () => {
  const cases = [
    { ...conversationPlan(), conversation: { ...conversationPlan().conversation, max_sessions: null } },
    { ...mysqlPlan(), storage: { backend: "mysql", persistent: false, notice: "invalid" } },
    { ...mysqlPlan(), storage: { backend: "memory", persistent: true, notice: "invalid" } },
    { ...mysqlPlan(), conversation: { ...conversationPlan().conversation, max_sessions: 16 } },
  ];
  for (const initial of cases) {
    const h = await ready({ initial, storedId: CONVERSATION_ID });
    assert.equal(h.get("ask").disabled, true); assert.equal(h.calls.length, 1); assert.equal(h.storageCalls.length, 0);
  }
});
