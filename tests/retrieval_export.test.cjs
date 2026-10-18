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

function rerankFixture(candidateCount = 10) {
  const data = fixture();
  for (let number = 11; number <= candidateCount; number++) {
    const id = `chunk-${number}`;
    data.records[id] = { chunk_id: id, text: `完整合成正文 ${number}，不可执行。`, metadata: { title: `额外候选${number}`, chunk_id: id } };
  }
  if (candidateCount !== 10) data.routes.dense = Array.from({ length: 8 }, (_, index) => ({
    chunk_id: `chunk-${candidateCount - 7 + index}`, rank: index + 1, score: 0.91 - index / 100,
  }));
  data.experiment_id = "opaque-server-experiment-01";
  const union = [...new Set(["bm25", "dense"].flatMap(key => data.routes[key].map(hit => hit.chunk_id)))];
  data.rrf_pool = union.map(id => {
    const ranks = {}, contributions = {};
    for (const key of ["bm25", "dense"]) {
      const hit = data.routes[key].find(item => item.chunk_id === id);
      ranks[key] = hit ? hit.rank : null;
      contributions[key] = hit ? 1 / (data.rrf_constant + hit.rank) : 0;
    }
    return { chunk_id: id, ranks, contributions, score: contributions.bm25 + contributions.dense };
  }).sort((a, b) => b.score - a.score).map((hit, index) => ({ ...hit, rank: index + 1 }));
  data.routes.rrf = data.rrf_pool.slice(0, data.top_k);
  const poolMap = new Map(data.rrf_pool.map((hit, index) => [hit.chunk_id, { hit, index }]));
  const order = candidateCount === 10 ? [9, 3, 1, 8, 4, 5, 6, 7, 2, 10]
    : [candidateCount, ...data.rrf_pool.map(hit => Number(hit.chunk_id.split("-")[1])).filter(number => number !== candidateCount)];
  data.routes.reranker = order.map((number, index) => {
    const id = `chunk-${number}`, old = poolMap.get(id), rank = index + 1;
    return { chunk_id: id, rank, score: 0.9876543210123456 - index / 25,
      input_index: old.index, rrf_rank: old.hit.rank, rank_change: old.hit.rank - rank };
  });
  data.rerank_comparison = {
    candidate_count: data.rrf_pool.length, final_k: 5, model: "synthetic-reranker-only",
    input_order: data.rrf_pool.map(hit => hit.chunk_id),
    request_counts: { embedding: 0, http: 1, reranker: 1, chat: 0 },
    notice: "模型排序需要人审；同分不是相关性标注。",
  };
  data.request_counts = { embedding: 1, http: 4, reranker: 1, chat: 0 };
  return data;
}

function beforeRerank(completed = rerankFixture(13)) {
  const data = JSON.parse(JSON.stringify(completed));
  delete data.routes.reranker;
  delete data.rerank_comparison;
  data.request_counts = { embedding: 1, http: 3, reranker: 0, chat: 0 };
  return data;
}

function answerFixture({ status = "answered", model = "synthetic-chat-only" } = {}) {
  const data = rerankFixture(13), sources = {};
  const ids = data.routes.reranker.slice(0, Math.min(5, data.top_k)).map(hit => hit.chunk_id);
  for (const [index, id] of ids.entries()) {
    const record = data.records[id];
    sources[`S${index + 1}`] = { chunk_id: id, text: record.text,
      quotes: { Q1: record.text.split("\n")[0] }, metadata: JSON.parse(JSON.stringify(record.metadata)) };
  }
  data.answer_generation = {
    model, max_tokens: 4096, evidence_ids: ids, sources,
    answer: {
      status,
      claims: status === "insufficient_evidence" ? [] : [
        { text: '合成回答：这只是待核验的示例，不执行命令 <img src=x onerror=alert(1)>。', evidence: [{ source_id: "S1", quote_id: "Q1" }] },
        { text: "合成回答：需要区分示例值与现场实际值。", evidence: [{ source_id: "S2", quote_id: "Q1" }] },
      ],
      missing_info: status === "answered" ? [] : ["还需补充集群版本和现场解析结果。"],
    },
    request_counts: { embedding: 0, http: 1, reranker: 0, chat: 1 },
    notice: "合成回答，仅验证结构与引用定位；不能证明推断被原文支持。",
  };
  data.request_counts = { embedding: 1, http: 5, reranker: 1, chat: 1 };
  return data;
}

function beforeAnswer(completed = answerFixture()) {
  const data = JSON.parse(JSON.stringify(completed));
  delete data.answer_generation;
  data.request_counts = { embedding: 1, http: 4, reranker: 1, chat: 0 };
  return data;
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

test("current BM25 server response permits an explicit empty root RRF pool but rejects a populated one", () => {
  const data = fixture("bm25");
  data.rrf_pool = [];
  const report = helper.buildReport(deepFreeze(data), TIMES);
  assert.deepEqual(report.result.rrf_pool, []);
  const summary = helper.summaryText(report);
  assert.match(summary, /仅BM25（Dense\/RRF未运行）/);
  assert.match(summary, /BM25 8；Dense 0；RRF 0；去重并集8块/);
  assert.match(summary, /Embedding 0；HTTP 0；Reranker 0；聊天 0/);
  assert.ok(summary.split("\n").filter(line => line.startsWith("| chunk-")).every(line => line.endsWith("| 未运行 | 未运行 |")));
  const invalid = fixture("bm25");
  invalid.rrf_pool = [{ chunk_id: "chunk-1", rank: 1, score: 1 / 61 }];
  assert.throws(() => helper.buildReport(invalid, TIMES), /仅BM25模式不能有融合候选/);
});

test("reranker JSON preserves the full input pool, original precision, rank mappings, identity, and cumulative counts", () => {
  const result = deepFreeze(rerankFixture());
  const report = helper.buildReport(result, { ...TIMES, selectedChunkId: "chunk-9" });
  assert.deepEqual(report.result, result);
  const saved = JSON.parse(JSON.stringify(report)).result;
  assert.equal(saved.experiment_id, "opaque-server-experiment-01");
  assert.equal(saved.rrf_pool.length, 10);
  assert.equal(saved.routes.rrf.length, 5);
  assert.equal(saved.routes.reranker.length, 10);
  assert.equal(saved.routes.reranker[0].score, result.routes.reranker[0].score);
  assert.deepEqual(saved.rerank_comparison.input_order, result.rrf_pool.map(hit => hit.chunk_id));
  assert.deepEqual(saved.request_counts, { embedding: 1, http: 4, reranker: 1, chat: 0 });
  assert.deepEqual(saved.rerank_comparison.request_counts, { embedding: 0, http: 1, reranker: 1, chat: 0 });
  assert.ok(saved.records["chunk-9"].text.endsWith("END-9"));
});

test("reranker summary adds a separate complete comparison without changing original three-route table semantics", () => {
  const data = deepFreeze(rerankFixture());
  const before = JSON.stringify(data);
  const summary = helper.summaryText(helper.buildReport(data, TIMES));
  // Both tables have five columns; inspect each section independently instead of counting only visible cards.
  const originalTable = summary.split("## 完整候选排名")[1].split("## RRF贡献")[0];
  const originalRows = originalTable.split("\n").filter(line => line.startsWith("| chunk-"));
  assert.equal(originalRows.length, 10);
  assert.ok(originalRows.find(line => line.startsWith("| chunk-9 | ")).endsWith("| 未入融合前5 |"));
  const comparisonTable = summary.split("## 完整候选重排对照")[1].split("## 来源")[0];
  const comparisonRows = comparisonTable.split("\n").filter(line => line.startsWith("| chunk-"));
  assert.equal(comparisonRows.length, 10);
  assert.deepEqual(comparisonRows.map(line => line.split(" | ")[0]), data.routes.reranker.map(hit => `| ${hit.chunk_id}`));
  assert.match(comparisonRows[0], /chunk-9.*#9 \/.*#1 \/.*上升8.*新进Top5/);
  assert.match(comparisonRows[5], /chunk-5.*下降3.*退出Top5/);
  assert.match(comparisonRows[1], /保留Top5/);
  assert.match(comparisonRows[9], /不变.*仍在Top5外/);
  assert.match(summary, /完整两路去重并集10块，不是先取RRF前5块/);
  assert.match(summary, /实际请求：Embedding 1；HTTP 4；Reranker 1；聊天 0/);
  assert.match(summary, /本次成功重排新增请求：Embedding 0；HTTP 1；Reranker 1；聊天 0/);
  assert.match(summary, /完成阶段累计，不重复相加/);
  assert.match(summary, /失败请求另计，可能未包含/);
  assert.match(summary, /只接收问题和候选正文，不发送候选元数据/);
  assert.match(summary, /无人工相关性标注，不能据此宣称更好/);
  assert.match(summary, /原始分数不可混加/);
  assert.match(summary, /重排分数不是概率或正确率/);
  assert.match(summary, /synthetic-reranker-only/);
  assert.match(summary, /opaque-server-experiment-01/);
  assert.equal(JSON.stringify(data), before);
});

test("complete RRF pool can be exported before running reranker", () => {
  const data = rerankFixture();
  delete data.routes.reranker;
  delete data.rerank_comparison;
  data.request_counts = { embedding: 1, http: 3, reranker: 0, chat: 0 };
  const report = helper.buildReport(data, TIMES);
  assert.equal(report.result.rrf_pool.length, 10);
  assert.ok(!helper.summaryText(report).includes("## 完整候选重排对照"));
});

test("answer report preserves complete evidence, reference precision, original metadata, and known-stage counts", () => {
  const original = deepFreeze(answerFixture());
  const report = helper.buildReport(original, { ...TIMES, selectedChunkId: original.answer_generation.evidence_ids[0] });
  const saved = JSON.parse(JSON.stringify(report)).result;
  assert.deepEqual(saved, original);
  assert.deepEqual(saved.answer_generation.evidence_ids, saved.routes.reranker.slice(0, 5).map(hit => hit.chunk_id));
  assert.equal(Object.keys(saved.answer_generation.sources).length, 5);
  assert.equal(saved.answer_generation.sources.S2.metadata.source, "/Users/synthetic/private-source.md");
  assert.equal(saved.answer_generation.sources.S2.text, original.records[saved.answer_generation.evidence_ids[1]].text);
  assert.deepEqual(saved.answer_generation.request_counts, { embedding: 0, http: 1, reranker: 0, chat: 1 });
  assert.deepEqual(saved.request_counts, { embedding: 1, http: 5, reranker: 1, chat: 1 });
  const summary = helper.summaryText(report);
  assert.match(summary, /模式：.*带引用回答/);
  assert.match(summary, /回答模型：synthetic-chat-only；最大输出Token=4096/);
  assert.match(summary, /回答状态：answered/);
  assert.match(summary, /实际请求：Embedding 1；HTTP 5；Reranker 1；聊天 1/);
  assert.match(summary, /成功生成新增请求：Embedding 0；HTTP 1；Reranker 0；聊天 1/);
  assert.match(summary, /\[S1\] chunk-13/);
  assert.match(summary, /原文\[S1\/Q1\]：完整合成正文 13/);
  assert.match(summary, /needs_review/);
  assert.match(summary, /来源URL未记录/);
  assert.match(summary, /不是正确率评测/);
  assert.match(summary, /示例名称、IP、输出不能当作现场值/);
  assert.match(summary, /命令只作为文本展示，不执行/);
  assert.match(summary, /不等于已验证推断被原文充分支持/);
  assert.ok(!summary.includes("/Users/synthetic"));
  assert.ok(!summary.includes("<img"));
  assert.ok(!summary.includes("END-3")); // Candidate body stays complete in JSON, not the summary.
});

test("partial and insufficient answers expose missing information without implying correctness", () => {
  for (const status of ["partial", "insufficient_evidence"]) {
    const summary = helper.summaryText(helper.buildReport(deepFreeze(answerFixture({ status })), TIMES));
    assert.match(summary, new RegExp(`回答状态：${status}`));
    assert.match(summary, /还需补充或核实：/);
    assert.match(summary, /还需补充集群版本和现场解析结果/);
    assert.match(summary, /仅检查结构、引用身份及摘录能否定位到原文/);
    if (status === "insufficient_evidence") assert.match(summary, /本轮没有可展示的有引用要点/);
  }
});

test("answer export rejects changed evidence, incomplete rerank, malformed schema, unsupported references, and invented counts", () => {
  const mutations = [
    data => { delete data.routes.reranker; delete data.rerank_comparison; },
    data => { data.routes.reranker.pop(); },
    data => { data.answer_generation = null; },
    data => { data.answer_generation.extra = "not allowed"; },
    data => { data.answer_generation.model = ""; },
    data => { data.answer_generation.max_tokens = 1024; },
    data => { delete data.answer_generation.notice; },
    data => { data.answer_generation.evidence_ids.reverse(); },
    data => { data.answer_generation.evidence_ids.pop(); },
    data => { data.answer_generation.evidence_ids[0] = "chunk-12"; },
    data => { data.answer_generation.sources.S6 = data.answer_generation.sources.S1; },
    data => { delete data.answer_generation.sources.S2; },
    data => { data.answer_generation.sources.S01 = data.answer_generation.sources.S1; delete data.answer_generation.sources.S1; },
    data => { data.answer_generation.sources.S1.chunk_id = "chunk-1"; },
    data => { data.answer_generation.sources.S1.text = "changed source"; },
    data => { data.answer_generation.sources.S1.metadata.title = "changed metadata"; },
    data => { delete data.answer_generation.sources.S1.metadata.chunk_id; },
    data => { data.answer_generation.sources.S1.quotes = {}; },
    data => { data.answer_generation.sources.S1.quotes.Q1 = "this quotation does not occur in the original"; },
    data => { data.answer_generation.sources.S1.quotes.Q1 = ""; },
    data => { data.answer_generation.sources.S1.quotes.Q01 = data.answer_generation.sources.S1.quotes.Q1; delete data.answer_generation.sources.S1.quotes.Q1; },
    data => { data.answer_generation.sources.S1.quotes.Q3 = data.answer_generation.sources.S1.quotes.Q1; },
    data => { data.answer_generation.answer.status = "reliable"; },
    data => { data.answer_generation.answer.extra = true; },
    data => { data.answer_generation.answer.claims = []; },
    data => { data.answer_generation.answer.status = "partial"; },
    data => { data.answer_generation.answer.status = "insufficient_evidence"; },
    data => { data.answer_generation.answer.claims.push(...data.answer_generation.answer.claims); },
    data => { data.answer_generation.answer.claims[0].extra = true; },
    data => { data.answer_generation.answer.claims[0].text = ""; },
    data => { data.answer_generation.answer.claims[0].text = " "; },
    data => { data.answer_generation.answer.claims[0].text = " 文本 "; },
    data => { data.answer_generation.answer.claims[0].text = "字".repeat(301); },
    data => { data.answer_generation.answer.claims[0].text = "自由文本不能包含[S1]引用"; },
    data => { data.answer_generation.answer.claims[0].text = "不能有\n换行"; },
    data => { data.answer_generation.answer.claims[0].text = "不能有\u007f控制符"; },
    data => { data.answer_generation.answer.claims[0].evidence = []; },
    data => { data.answer_generation.answer.claims[0].evidence = Array(4).fill({ source_id: "S1", quote_id: "Q1" }); },
    data => { data.answer_generation.answer.claims[0].evidence[0].source_id = "S6"; },
    data => { data.answer_generation.answer.claims[0].evidence[0].source_id = "__proto__"; },
    data => { data.answer_generation.answer.claims[0].evidence[0].quote_id = "Q2"; },
    data => { data.answer_generation.answer.claims[0].evidence[0].quote_id = "Q01"; },
    data => { data.answer_generation.answer.claims[0].evidence[0].extra = true; },
    data => { data.answer_generation.answer.missing_info = ["一", "二", "三", "四"]; },
    data => { data.answer_generation.answer.missing_info = ["字".repeat(241)]; },
    data => { data.answer_generation.answer.missing_info = ["\t不能有制表符"]; },
    data => { data.answer_generation.answer.missing_info = ["不能在这里引用[S5]"]; },
    data => { data.answer_generation.request_counts.http = 2; },
    data => { data.answer_generation.request_counts.chat = 0; },
    data => { data.answer_generation.request_counts.embedding = 1; },
    data => { data.answer_generation.request_counts.reranker = 1; },
    data => { data.request_counts.http = 4; },
    data => { data.request_counts.http = 6; },
    data => { data.request_counts.embedding = 2; },
  ];
  for (const mutate of mutations) {
    const data = answerFixture(); mutate(data);
    assert.throws(() => helper.buildReport(data, TIMES), undefined, mutate.toString());
  }
  const insufficient = answerFixture({ status: "insufficient_evidence" });
  insufficient.answer_generation.answer.missing_info = [];
  assert.throws(() => helper.buildReport(insufficient, TIMES));
});

test("answer text uses Unicode codepoints and complete JSON metadata equality independent of object key order", () => {
  const data = answerFixture();
  data.answer_generation.answer.claims[0].text = "🧪".repeat(300);
  const source = data.answer_generation.sources.S2;
  source.metadata = Object.fromEntries(Object.entries(source.metadata).reverse());
  assert.doesNotThrow(() => helper.buildReport(data, TIMES));
  data.answer_generation.answer.claims[0].text += "🧪";
  assert.throws(() => helper.buildReport(data, TIMES));
});

test("answer evidence honors a smaller final Top-K and long citation previews never truncate the JSON", () => {
  const data = answerFixture();
  data.top_k = 3;
  data.routes.rrf = data.rrf_pool.slice(0, 3);
  data.rerank_comparison.final_k = 3;
  data.answer_generation.evidence_ids = data.answer_generation.evidence_ids.slice(0, 3);
  delete data.answer_generation.sources.S4;
  delete data.answer_generation.sources.S5;
  const source = data.answer_generation.sources.S2;
  const offset = source.text.indexOf("尾段不可截断。");
  source.quotes.Q1 = source.text.slice(offset, offset + 300);
  const report = helper.buildReport(data, TIMES), summary = helper.summaryText(report);
  assert.equal(Object.keys(report.result.answer_generation.sources).length, 3);
  assert.equal(report.result.answer_generation.sources.S2.quotes.Q1.length, 300);
  assert.match(summary, /摘要截短；完整原文见JSON/);
  assert.ok(!summary.includes(source.quotes.Q1));
  data.answer_generation.evidence_ids.push(data.routes.reranker[3].chunk_id);
  assert.throws(() => helper.buildReport(data, TIMES));
});

test("answer failure history remains exportable before and after a successful manual retry", () => {
  for (const completed of [false, true]) {
    const data = completed ? answerFixture() : beforeAnswer();
    data.answer_failure = { message: "合成回答失败 <script>bad</script>", request_may_have_occurred: true, request_counts_scope: "completed_stages_only" };
    data.rerank_failure = { message: "此前重排失败", request_may_have_occurred: true, request_counts_scope: "completed_stages_only" };
    const saved = helper.buildReport(deepFreeze(data), TIMES), summary = helper.summaryText(saved);
    assert.deepEqual(saved.result.answer_failure, data.answer_failure);
    assert.match(summary, /回答生成失败经历：合成回答失败 &lt;script&gt;/);
    assert.match(summary, /重排失败经历：此前重排失败/);
    assert.match(summary, /仅覆盖完成阶段，不包含未知的失败请求/);
    assert.match(summary, /手动重试成功也不会抹去这次失败经历/);
    assert.match(summary, completed ? /HTTP 5；Reranker 1；聊天 1/ : /HTTP 4；Reranker 1；聊天 0/);
  }
  for (const failure of [null, {}, { message: "error", request_may_have_occurred: false, request_counts_scope: "completed_stages_only" },
    { message: "error", request_may_have_occurred: true, request_counts_scope: "all_attempts" }]) {
    const data = beforeAnswer(); data.answer_failure = failure;
    assert.throws(() => helper.buildReport(data, TIMES));
  }
  const invalidStage = beforeRerank();
  invalidStage.answer_failure = { message: "error", request_may_have_occurred: true, request_counts_scope: "completed_stages_only" };
  assert.throws(() => helper.buildReport(invalidStage, TIMES));
});

test("failed rerank history survives export before and after a successful manual retry without inventing counts", () => {
  for (const completed of [false, true]) {
    const data = rerankFixture();
    data.rerank_failure = {
      message: "合成失败 <img src=x onerror=alert(1)> | 服务未给出有效重排结果",
      request_may_have_occurred: true,
      request_counts_scope: "completed_stages_only",
    };
    if (!completed) {
      delete data.routes.reranker;
      delete data.rerank_comparison;
      data.request_counts = { embedding: 1, http: 3, reranker: 0, chat: 0 };
    }
    const report = helper.buildReport(deepFreeze(data), TIMES);
    assert.deepEqual(report.result.rerank_failure, data.rerank_failure);
    const summary = helper.summaryText(report);
    assert.match(summary, /重排失败经历：合成失败 &lt;img/);
    assert.match(summary, /失败请求可能已发送，处理量与费用未知/);
    assert.match(summary, /仅覆盖完成阶段，不包含未知的失败请求/);
    assert.match(summary, /手动重试成功也不会抹去这次失败经历/);
    assert.match(summary, completed ? /Embedding 1；HTTP 4；Reranker 1；聊天 0/ : /Embedding 1；HTTP 3；Reranker 0；聊天 0/);
    assert.ok(!summary.includes("<img"));
  }
  for (const failure of [null, {}, { message: "error", request_may_have_occurred: false, request_counts_scope: "completed_stages_only" },
    { message: "error", request_may_have_occurred: true, request_counts_scope: "all_attempts" }]) {
    const data = rerankFixture(); data.rerank_failure = failure;
    assert.throws(() => helper.buildReport(data, TIMES));
  }
});

test("malformed full pools, rerank coverage, identity, mapping, parameters, and request counts are rejected", () => {
  const invalid = [
    data => { delete data.rrf_pool; },
    data => { delete data.rerank_comparison; },
    data => { delete data.routes.reranker; },
    data => { delete data.experiment_id; },
    data => { data.experiment_id = ""; },
    data => { data.mode = "bm25"; },
    data => { data.rrf_pool.pop(); },
    data => { data.rrf_pool[9].chunk_id = data.rrf_pool[8].chunk_id; },
    data => { data.rrf_pool[9].chunk_id = "unknown"; },
    data => { data.rrf_pool[9].rank = 9; },
    data => { data.rrf_pool[9].score = Infinity; },
    data => { data.routes.rrf[0] = { ...data.routes.rrf[0], score: 0 }; },
    data => { data.routes.rrf.pop(); },
    data => { data.routes.reranker.pop(); },
    data => { data.routes.reranker[1].chunk_id = data.routes.reranker[0].chunk_id; },
    data => { data.routes.reranker[0].chunk_id = "unseen"; },
    data => { data.routes.reranker[0].rank = 2; },
    data => { data.routes.reranker[0].score = NaN; },
    data => { data.routes.reranker[0].input_index = 0; },
    data => { data.routes.reranker[0].rrf_rank = 1; },
    data => { data.routes.reranker[0].rank_change = -8; },
    data => { data.rerank_comparison.input_order.reverse(); },
    data => { data.rerank_comparison.candidate_count = 5; },
    data => { data.rerank_comparison.final_k = 3; },
    data => { data.rerank_comparison.model = ""; },
    data => { delete data.rerank_comparison.notice; },
    data => { data.rerank_comparison.request_counts.embedding = 1; },
    data => { data.rerank_comparison.request_counts.http = 2; },
    data => { data.rerank_comparison.request_counts.reranker = 0; },
    data => { data.rerank_comparison.request_counts.chat = 1; },
    data => { data.request_counts.reranker = 0; },
  ];
  for (const mutate of invalid) {
    const data = rerankFixture();
    mutate(data);
    assert.throws(() => helper.buildReport(data, TIMES), undefined, mutate.toString());
  }
  const report = helper.buildReport(rerankFixture(), TIMES);
  report.result.routes.reranker[0].rank_change = 0;
  assert.throws(() => helper.summaryText(report));
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
  vm.runInContext(inline.replace(/\ninit\(\);\s*$/, "") + "\nglobalThis.ui={state,renderResults,clearResults,setBusy,hasExportResult,updateExportButtons,copySummary,downloadJson,compare,rerank,updateRerankControls,validateRerankData,generateAnswer,updateAnswerControls,validateAnswerData,showAnswerSource,configureAnswerModels,renderAnswerEvidence};", sandbox, { filename: pagePath });
  const element = id => document.getElementById(id);
  function ready(data = fixture()) {
    sandbox.ui.state.plan = { document_count: 42, reranker: { model: "synthetic-reranker-only", url: "https://example.invalid/synthetic-rerank", max_candidates: 16 }, answer: {
      models: [
        { id: "synthetic-chat-only", base_url: "https://example.invalid/synthetic-chat", transport_notice: "HTTPS合成服务，仍发送证据。" },
        { id: "synthetic-glm-http", base_url: "http://example.invalid/synthetic-glm", transport_notice: "HTTP明文传输，不发送敏感资料。" },
      ], default_model: "synthetic-chat-only", max_sources: 5, max_tokens: 4096, notice: "只发送问题、证据和必要标记；无自动重试。",
    } };
    element("query").value = data.query;
    sandbox.ui.renderResults(data);
    return data;
  }
  return { sandbox, ui: sandbox.ui, element, events, created, ready };
}

test("cloud rerank UI requires a matching completed hybrid experiment, consent, capacity, and idle state", async () => {
  const { ui, element, events, ready } = uiHarness();
  ready(fixture("bm25"));
  element("rerank-consent").checked = true;
  ui.updateRerankControls();
  assert.equal(element("rerank-consent").disabled, true);
  assert.equal(element("rerank-button").disabled, true);
  await ui.rerank();
  assert.equal(events.fetch.length, 0);
  ready(beforeRerank());
  assert.equal(element("rerank-consent").checked, false);
  assert.equal(element("rerank-consent").disabled, false);
  assert.equal(element("rerank-button").disabled, true);
  await ui.rerank();
  assert.equal(events.fetch.length, 0);
  assert.match(element("rerank-plan").textContent, /共 13 块，全部参与云端重排/);
  assert.match(element("rerank-disclosure").textContent, /https:\/\/example\.invalid\/synthetic-rerank/);
  assert.match(element("rerank-disclosure").textContent, /不发送来源路径等元数据/);
  element("rerank-consent").checked = true;
  element("rerank-consent").listeners.change();
  assert.equal(element("rerank-button").disabled, false);
  ui.setBusy(true);
  assert.equal(element("rerank-button").disabled, true);
  assert.equal(element("rerank-consent").disabled, true);
  await ui.rerank();
  assert.equal(events.fetch.length, 0);
  ui.setBusy(false);
  ui.state.plan.reranker.max_candidates = 12;
  ui.updateRerankControls();
  assert.equal(element("rerank-button").disabled, true);
  assert.match(element("rerank-button-help").textContent, /超过本轮上限 12/);
  ui.state.plan.reranker.max_candidates = 16;
  element("query").value = "different question";
  ui.updateRerankControls();
  await ui.rerank();
  assert.equal(events.fetch.length, 0);
});

test("UI posts only experiment token once and preserves all 13 candidates while displaying Top-5", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  const completed = rerankFixture(13), original = ready(beforeRerank(completed));
  const originalJson = JSON.stringify(original);
  let resolveResponse;
  sandbox.fetch = (...args) => {
    events.fetch.push(args);
    return new Promise(resolve => { resolveResponse = resolve; });
  };
  element("rerank-consent").checked = true;
  ui.updateRerankControls();
  const pending = ui.rerank();
  assert.equal(ui.state.busy, true);
  assert.equal(element("rerank-button").disabled, true);
  assert.equal(element("copy-result").disabled, true);
  await ui.rerank();
  assert.equal(events.fetch.length, 1);
  const [url, options] = events.fetch[0];
  assert.equal(url, "/api/rerank");
  assert.equal(options.method, "POST");
  assert.equal(options.credentials, "same-origin");
  assert.equal(options.cache, "no-store");
  assert.equal(options.headers["X-Retrieval-Lab"], "1");
  assert.deepEqual(JSON.parse(options.body), { experiment_id: original.experiment_id });
  for (const privateText of [original.query, "metadata", "/Users/synthetic", "完整原文"])
    assert.ok(!options.body.includes(privateText));
  resolveResponse({ ok: true, json: async () => completed });
  await pending;
  assert.equal(JSON.stringify(original), originalJson);
  assert.equal(ui.state.data.routes.reranker.length, 13);
  assert.equal(ui.state.data.rrf_pool.length, 13);
  assert.equal(element("rrf-before-list").children.length, 5);
  assert.equal(element("reranker-list").children.length, 5);
  assert.equal(element("rank-change-body").children.length, 13);
  assert.match(element("ranking-detail-label").textContent, /全部 13 块/);
  assert.match(element("rerank-summary").textContent, /同一 13 块/);
  assert.equal(element("rerank-results").hidden, false);
  assert.equal(element("rerank-button").disabled, true);
  assert.equal(element("rerank-consent").disabled, true);
  assert.equal(element("copy-result").disabled, false);
  await ui.rerank();
  await ui.copySummary();
  ui.downloadJson();
  assert.equal(events.fetch.length, 1); // Neither repeat click nor sharing re-runs any stage.
  assert.match(events.clipboard[0], /完整两路去重并集13块/);
  const report = JSON.parse(await events.blobs[0].text());
  assert.equal(report.result.routes.reranker.length, 13);
  assert.equal(report.result.rrf_pool.length, 13);
});

test("UI failure keeps original routes exportable and successful manual retry retains failed-request history", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  const completed = rerankFixture(13), original = ready(beforeRerank(completed));
  const originalRoutes = JSON.stringify(original.routes), originalRecords = JSON.stringify(original.records);
  sandbox.fetch = async (...args) => {
    events.fetch.push(args);
    return { ok: false, json: async () => ({ detail: "synthetic unavailable" }) };
  };
  element("rerank-consent").checked = true;
  await ui.rerank();
  assert.equal(events.fetch.length, 1); // No automatic retry.
  assert.equal(JSON.stringify(ui.state.data.routes), originalRoutes);
  assert.equal(JSON.stringify(ui.state.data.records), originalRecords);
  assert.equal(ui.state.data.request_counts.http, 3);
  assert.equal(ui.state.data.rerank_failure.request_may_have_occurred, true);
  assert.equal(element("rerank-results").hidden, true);
  assert.equal(element("rank-change-body").children.length, 0);
  assert.equal(element("rerank-error").hidden, false);
  assert.match(element("rerank-error").textContent, /不会自动重试/);
  assert.equal(element("copy-result").disabled, false);
  assert.equal(element("export-json").disabled, false);
  await ui.copySummary();
  assert.match(events.clipboard[0], /重排失败经历/);
  assert.match(events.clipboard[0], /失败请求可能已发送/);
  const failure = JSON.stringify(ui.state.data.rerank_failure);
  assert.equal(element("rerank-button").disabled, false); // Human can explicitly choose to retry.
  sandbox.fetch = async (...args) => { events.fetch.push(args); return { ok: true, json: async () => completed }; };
  await ui.rerank();
  assert.equal(events.fetch.length, 2);
  assert.equal(ui.state.data.routes.reranker.length, 13);
  assert.equal(JSON.stringify(ui.state.data.rerank_failure), failure);
  assert.match(element("request-summary").textContent, /不含失败尝试/);
  assert.match(element("rerank-status").textContent, /失败尝试可能已产生云端费用/);
  await ui.copySummary();
  ui.downloadJson();
  assert.match(events.clipboard[1], /重排失败经历/);
  assert.match(events.clipboard[1], /完成阶段累计/);
  const report = JSON.parse(await events.blobs[0].text());
  assert.equal(JSON.stringify(report.result.rerank_failure), failure);
  assert.equal(report.result.request_counts.http, 4);
  assert.equal(events.fetch.length, 2);
  element("query").value = "新问题，不沿用旧实验";
  element("query").listeners.input();
  assert.equal(ui.state.data, null);
  assert.equal(element("rerank-consent").checked, false);
  assert.equal(element("rerank-button").disabled, true);
  assert.equal(element("rerank-results").hidden, true);
  assert.equal(element("copy-result").disabled, true);
});

test("UI validates fixed original candidates and rejects malformed rerank responses without partial display", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  const mutations = [
    data => { data.query = "changed"; },
    data => { data.experiment_id = "another-experiment"; },
    data => { data.rrf_pool.reverse(); },
    data => { data.records["chunk-1"].text = "changed original"; },
    data => { data.routes.bm25[0].score = 999; },
    data => { data.routes.reranker.pop(); },
    data => { data.routes.reranker[0].chunk_id = data.routes.reranker[1].chunk_id; },
    data => { data.routes.reranker[0].score = Infinity; },
    data => { data.routes.reranker[0].input_index = 0; },
    data => { data.routes.reranker[0].rrf_rank = 1; },
    data => { data.routes.reranker[0].rank_change = -100; },
    data => { data.rerank_comparison.input_order.reverse(); },
    data => { data.rerank_comparison.request_counts.http = 2; },
  ];
  for (const mutate of mutations) {
    const valid = rerankFixture(13), original = beforeRerank(valid), invalid = JSON.parse(JSON.stringify(valid));
    mutate(invalid);
    assert.throws(() => ui.validateRerankData(invalid, original), undefined, mutate.toString());
  }
  const original = ready(beforeRerank());
  const invalid = rerankFixture(13); invalid.routes.reranker.pop();
  sandbox.fetch = async (...args) => { events.fetch.push(args); return { ok: true, json: async () => invalid }; };
  element("rerank-consent").checked = true;
  await ui.rerank();
  assert.equal(events.fetch.length, 1);
  assert.equal(ui.state.data.routes, original.routes);
  assert.equal(element("rerank-results").hidden, true);
  assert.equal(element("rank-change-body").children.length, 0);
  assert.equal(element("copy-result").disabled, false);
});

test("answer UI requires completed rerank, current question, explicit consent and idle state", async () => {
  const { ui, element, events, ready } = uiHarness();
  for (const data of [fixture("bm25"), beforeRerank()]) {
    ready(data);
    element("answer-consent").checked = true;
    ui.updateAnswerControls();
    assert.equal(element("answer-consent").disabled, true);
    assert.equal(element("answer-button").disabled, true);
    await ui.generateAnswer();
  }
  ready(beforeAnswer());
  assert.equal(element("answer-consent").checked, false);
  assert.equal(element("answer-consent").disabled, false);
  assert.equal(element("answer-button").disabled, true);
  assert.equal(ui.state.answerModel, "synthetic-chat-only");
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 0);
  assert.match(element("answer-plan").textContent, /重排前 5 块/);
  assert.match(element("answer-disclosure").textContent, /https:\/\/example\.invalid\/synthetic-chat/);
  assert.match(element("answer-disclosure").textContent, /来源路径和完整元数据不交给模型/);
  element("answer-consent").checked = true;
  element("answer-consent").listeners.change();
  assert.equal(element("answer-button").disabled, false);
  ui.setBusy(true);
  assert.equal(element("answer-button").disabled, true);
  assert.equal(element("answer-model").disabled, true);
  assert.equal(element("answer-consent").disabled, true);
  await ui.generateAnswer();
  ui.setBusy(false);
  element("query").value = "another question";
  ui.updateAnswerControls();
  assert.equal(element("answer-button").disabled, true);
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 0);
});

test("changing answer destination revokes consent and exposes HTTP transport risk without sending anything", async () => {
  const { ui, element, events, ready } = uiHarness();
  ready(beforeAnswer());
  element("answer-consent").checked = true;
  element("answer-consent").listeners.change();
  assert.equal(element("answer-button").disabled, false);
  element("answer-model").value = "synthetic-glm-http";
  element("answer-model").listeners.change();
  assert.equal(ui.state.answerModel, "synthetic-glm-http");
  assert.equal(element("answer-consent").checked, false);
  assert.equal(element("answer-button").disabled, true);
  assert.equal(element("answer-transport").hidden, false);
  assert.match(element("answer-transport").textContent, /HTTP 明文连接/);
  assert.match(element("answer-disclosure").textContent, /http:\/\/example\.invalid\/synthetic-glm/);
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 0);
  element("answer-model").value = "unlisted-model";
  element("answer-model").listeners.change();
  assert.equal(ui.state.answerModel, "synthetic-glm-http");
  assert.equal(element("answer-model").value, "synthetic-glm-http");
  ui.state.answerModel = "unlisted-model";
  element("answer-consent").checked = true;
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 0);
});

test("answer UI posts only experiment identity and model once, preserves candidates, and renders escaped citation text", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  const completed = answerFixture(), original = ready(beforeAnswer(completed));
  const originalJson = JSON.stringify(original), routes = JSON.stringify(original.routes), records = JSON.stringify(original.records);
  let resolveResponse;
  sandbox.fetch = (...args) => { events.fetch.push(args); return new Promise(resolve => { resolveResponse = resolve; }); };
  element("answer-consent").checked = true;
  ui.updateAnswerControls();
  const pending = ui.generateAnswer();
  assert.equal(ui.state.busy, true);
  assert.equal(element("answer-button").disabled, true);
  assert.equal(element("copy-result").disabled, true);
  assert.equal(element("answer-result").hidden, true);
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 1);
  const [url, options] = events.fetch[0];
  assert.equal(url, "/api/answer");
  assert.equal(options.method, "POST");
  assert.equal(options.credentials, "same-origin");
  assert.equal(options.cache, "no-store");
  assert.equal(options.headers["X-Retrieval-Lab"], "1");
  assert.deepEqual(JSON.parse(options.body), { experiment_id: original.experiment_id, model: "synthetic-chat-only" });
  for (const privateText of [original.query, "metadata", "/Users/synthetic", "完整原文", "api_key", "evidence"])
    assert.ok(!options.body.includes(privateText));
  resolveResponse({ ok: true, json: async () => completed });
  await pending;
  assert.equal(JSON.stringify(original), originalJson);
  assert.equal(JSON.stringify(ui.state.data.routes), routes);
  assert.equal(JSON.stringify(ui.state.data.records), records);
  assert.equal(element("answer-result").hidden, false);
  assert.equal(element("answer-claims").children.length, 2);
  assert.equal(element("answer-claims").children[0].children[0].textContent, completed.answer_generation.answer.claims[0].text);
  assert.equal(element("answer-button").disabled, true);
  assert.equal(element("answer-model").disabled, true);
  assert.equal(element("answer-consent").disabled, true);
  assert.equal(element("copy-result").disabled, false);
  ui.showAnswerSource("S1", "Q1");
  assert.equal(element("answer-source-view").hidden, false);
  assert.equal(element("answer-quote").textContent, completed.answer_generation.sources.S1.quotes.Q1);
  assert.equal(element("answer-source-text").textContent, completed.answer_generation.sources.S1.text);
  assert.equal(ui.state.selected, completed.answer_generation.sources.S1.chunk_id);
  assert.match(element("answer-quote-label").textContent, /不等于证明结论成立/);
  await ui.generateAnswer();
  await ui.copySummary();
  ui.downloadJson();
  assert.equal(events.fetch.length, 1);
  assert.match(events.clipboard[0], /回答状态：answered/);
  assert.match(events.clipboard[0], /原文\[S1\/Q1\]/);
  const saved = JSON.parse(await events.blobs[0].text()).result;
  assert.deepEqual(saved.answer_generation, completed.answer_generation);
  assert.equal(saved.routes.reranker.length, 13);
});

test("failed answer is not displayed, original retrieval remains exportable, and manual retry retains failure history", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  const completed = answerFixture(), original = ready(beforeAnswer(completed));
  const originalRoutes = JSON.stringify(original.routes), originalRecords = JSON.stringify(original.records);
  sandbox.fetch = async (...args) => { events.fetch.push(args); return { ok: false, json: async () => ({ detail: "no raw model output" }) }; };
  element("answer-consent").checked = true;
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 1);
  assert.equal(JSON.stringify(ui.state.data.routes), originalRoutes);
  assert.equal(JSON.stringify(ui.state.data.records), originalRecords);
  assert.equal(ui.state.data.request_counts.http, 4);
  assert.equal(ui.state.data.request_counts.chat, 0);
  assert.equal(ui.state.data.answer_failure.request_may_have_occurred, true);
  assert.equal(element("answer-result").hidden, true);
  assert.equal(element("answer-claims").children.length, 0);
  assert.equal(element("answer-error").hidden, false);
  assert.match(element("answer-error").textContent, /不会自动重试/);
  assert.equal(element("copy-result").disabled, false);
  await ui.copySummary();
  assert.match(events.clipboard[0], /回答生成失败经历/);
  const failure = JSON.stringify(ui.state.data.answer_failure);
  assert.equal(element("answer-button").disabled, false);
  sandbox.fetch = async (...args) => { events.fetch.push(args); return { ok: true, json: async () => completed }; };
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 2);
  assert.equal(JSON.stringify(ui.state.data.answer_failure), failure);
  assert.equal(element("answer-result").hidden, false);
  assert.match(element("request-summary").textContent, /不含失败尝试/);
  await ui.copySummary();
  assert.match(events.clipboard[1], /回答生成失败经历/);
  assert.match(events.clipboard[1], /HTTP 5；Reranker 1；聊天 1/);
  element("query").value = "new question";
  element("query").listeners.input();
  assert.equal(ui.state.data, null);
  assert.equal(element("answer-consent").checked, false);
  assert.equal(element("answer-button").disabled, true);
  assert.equal(element("answer-result").hidden, true);
});

test("answer UI validates the response model, immutable stages, citation contract and strict schema before displaying any body", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  const original = ready(beforeAnswer());
  const mutations = [
    data => { data.answer_generation.model = "another-model"; },
    data => { data.query = "different question"; },
    data => { data.experiment_id = "different-experiment"; },
    data => { data.records["chunk-1"].text = "changed"; },
    data => { data.routes.reranker[0].score = 0.999; },
    data => { data.answer_generation.sources.S1.text = "changed"; },
    data => { data.answer_generation.answer.claims[0].text = "字".repeat(301); },
    data => { data.answer_generation.answer.claims[0].text = "不能有[S1]"; },
    data => { data.answer_generation.answer.claims[0].text = "不能\n换行"; },
    data => { data.answer_generation.answer.claims[0].evidence = Array(4).fill({ source_id: "S1", quote_id: "Q1" }); },
    data => { data.answer_generation.answer.status = "partial"; },
    data => { data.answer_generation.answer.missing_info = ["字".repeat(241)]; },
    data => { data.answer_generation.answer.claims[0].evidence[0].quote_id = "Q999"; },
    data => { data.request_counts.chat = 2; },
  ];
  for (const mutate of mutations) {
    const invalid = answerFixture(); mutate(invalid);
    assert.throws(() => ui.validateAnswerData(invalid, original, "synthetic-chat-only"), undefined, mutate.toString());
  }
  const invalid = answerFixture();
  invalid.answer_generation.answer.claims[0].text = "字".repeat(301);
  sandbox.fetch = async (...args) => { events.fetch.push(args); return { ok: true, json: async () => invalid }; };
  element("answer-consent").checked = true;
  await ui.generateAnswer();
  assert.equal(events.fetch.length, 1);
  assert.equal(ui.state.data.routes, original.routes);
  assert.equal(element("answer-result").hidden, true);
  assert.equal(element("answer-claims").children.length, 0);
  assert.equal(element("copy-result").disabled, false);
  assert.ok(!JSON.stringify(ui.state.data).includes("字".repeat(301)));
});

test("late answer response cannot replace another experiment or expose a stale answer", async () => {
  const { ui, sandbox, element, events, ready } = uiHarness();
  ready(beforeAnswer());
  let resolveResponse;
  sandbox.fetch = (...args) => { events.fetch.push(args); return new Promise(resolve => { resolveResponse = resolve; }); };
  element("answer-consent").checked = true;
  const pending = ui.generateAnswer();
  ui.state.sequence++;
  ui.setBusy(false);
  ui.clearResults();
  const next = fixture("bm25"); next.query = "a new experiment";
  ready(next);
  resolveResponse({ ok: true, json: async () => answerFixture() });
  await pending;
  assert.equal(ui.state.data, next);
  assert.equal(element("answer-result").hidden, true);
  assert.equal(element("answer-claims").children.length, 0);
  assert.equal(element("answer-consent").checked, false);
  assert.equal(events.fetch.length, 1);
});

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
