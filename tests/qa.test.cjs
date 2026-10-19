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

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this._text = ""; this.value = "";
    this.listeners = {}; this.hidden = false; this.disabled = false; this.focused = false;
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
  postFailure = null, deferred = false } = {}) {
  const elements = new Map([...page.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/gi)].map(match => {
    const element = new Element(match[1]); element.hidden = /\bhidden\b/.test(match[2]);
    element.disabled = /\bdisabled\b/.test(match[2]); return [match[3], element];
  }));
  const calls = [], pending = [];
  const fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === "/api/qa/plan") return { ok: !getFailure, status: getFailure ? 503 : 200, json: async () => clone(initial) };
    assert.equal(url, "/api/qa");
    if (deferred) await new Promise(resolve => pending.push(resolve));
    if (postFailure === "network") throw new Error("synthetic transport failure");
    return { ok: postFailure !== "http", status: postFailure === "http" ? 503 : 200,
      json: async () => { if (postFailure === "json") throw new Error("synthetic invalid JSON");
        return clone(postFailure === "http" ? { detail: "本次生成失败，未得到校验通过的回答。" } : response); } };
  };
  const context = vm.createContext({ document: { getElementById: id => elements.get(id), createElement: tag => new Element(tag) }, fetch, URL });
  vm.runInContext(script, context);
  return { calls, get: id => elements.get(id), eval: code => vm.runInContext(code, context),
    finish: () => pending.shift()(), submit: () => {
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
