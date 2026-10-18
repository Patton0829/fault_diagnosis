/* 已完成结果 -> 可复制摘要/完整JSON。纯数据函数，不访问网络、文件或剪贴板。 */
(function (root) {
  "use strict";
  const SCHEMA = "retrieval-lab-report/v1";
  const ROUTES = ["bm25", "dense", "rrf"];
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const requireValue = (condition, message) => { if (!condition) throw new Error(message); };

  function validateCounts(counts) {
    requireValue(counts && ["embedding", "http", "reranker", "chat"].every(
      key => Number.isInteger(counts[key]) && counts[key] >= 0
    ), "缺少实际请求计数");
  }

  function validateHits(hits, records) {
    requireValue(Array.isArray(hits), "缺少完整路线结果");
    const seen = new Set();
    for (const [position, hit] of hits.entries()) {
      requireValue(hit && typeof hit.chunk_id === "string" && !seen.has(hit.chunk_id), "候选身份重复或缺失");
      requireValue(hit.rank === position + 1 && Number.isFinite(hit.score), "候选名次或分数非法");
      requireValue(own(records, hit.chunk_id) && typeof records[hit.chunk_id].text === "string", "缺少候选完整正文");
      seen.add(hit.chunk_id);
    }
    return seen;
  }

  const isObject = value => value !== null && typeof value === "object" && !Array.isArray(value);
  function exactKeys(value, keys) {
    return isObject(value) && Object.keys(value).length === keys.length && keys.every(key => own(value, key));
  }
  function sameJson(left, right) {
    if (left === right) return true;
    if (Array.isArray(left) || Array.isArray(right))
      return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((item, index) => sameJson(item, right[index]));
    return isObject(left) && isObject(right) && Object.keys(left).length === Object.keys(right).length &&
      Object.keys(left).every(key => own(right, key) && sameJson(left[key], right[key]));
  }
  function validateFreeText(text, maximum) {
    // Array.from按Unicode码点计数，与Python字符串长度一致；不把emoji的代理对算两次。
    requireValue(typeof text === "string" && text.trim() && text === text.trim() &&
      Array.from(text).length <= maximum && !/[\u0000-\u001f\u007f]/.test(text) && !/\[S[0-9]+\]/.test(text),
    "回答自由文本超长、为空、含控制字符或混入引用编号");
  }
  function validateAnswer(result) {
    requireValue(result.mode === "hybrid" && own(result.routes, "reranker") && own(result, "rerank_comparison"),
      "回答必须以完整重排结果为基础");
    const generation = result.answer_generation;
    requireValue(exactKeys(generation, ["model", "max_tokens", "evidence_ids", "sources", "answer", "request_counts", "notice"]) &&
      typeof generation.model === "string" && generation.model.trim() && generation.max_tokens === 4096 && typeof generation.notice === "string",
    "回答生成合同、模型或输出额度非法");
    const ids = result.routes.reranker.slice(0, Math.min(5, result.top_k)).map(hit => hit.chunk_id);
    requireValue(ids.length > 0 && sameJson(generation.evidence_ids, ids), "回答证据必须精确对应本轮重排前5块");
    const sourceIds = ids.map((id, index) => `S${index + 1}`);
    requireValue(exactKeys(generation.sources, sourceIds), "回答证据编号不完整或非法");
    for (const [index, sourceId] of sourceIds.entries()) {
      const source = generation.sources[sourceId], record = result.records[ids[index]];
      requireValue(exactKeys(source, ["chunk_id", "text", "quotes", "metadata"]) && source.chunk_id === ids[index] &&
        source.text === record.text && isObject(source.metadata) && sameJson(source.metadata, record.metadata),
      "回答来源正文或完整元数据与本轮候选不一致");
      requireValue(isObject(source.quotes) && Object.keys(source.quotes).length > 0, "回答来源缺少原文片段");
      const quoteIds = Object.keys(source.quotes);
      requireValue(quoteIds.every((id, position) => own(source.quotes, `Q${position + 1}`)) &&
        quoteIds.every(id => /^Q[1-9][0-9]*$/.test(id) && typeof source.quotes[id] === "string" &&
          source.quotes[id].length > 0 && source.text.includes(source.quotes[id])), "回答原文片段编号非法或无法定位到连续原文");
    }
    const answer = generation.answer;
    requireValue(exactKeys(answer, ["status", "claims", "missing_info"]) &&
      ["answered", "partial", "insufficient_evidence"].includes(answer.status) &&
      Array.isArray(answer.claims) && answer.claims.length <= 3 &&
      Array.isArray(answer.missing_info) && answer.missing_info.length <= 3, "回答JSON结构或条目数非法");
    requireValue(answer.status === "insufficient_evidence" ? answer.claims.length === 0 && answer.missing_info.length > 0
      : answer.claims.length > 0, "回答状态与要点或资料缺口不一致");
    requireValue(answer.status !== "partial" || answer.missing_info.length > 0, "部分回答必须说明资料缺口");
    for (const claim of answer.claims) {
      requireValue(exactKeys(claim, ["text", "evidence"]) && Array.isArray(claim.evidence) &&
        claim.evidence.length >= 1 && claim.evidence.length <= 3, "回答要点或引用条目数非法");
      validateFreeText(claim.text, 300);
      for (const reference of claim.evidence) {
        requireValue(exactKeys(reference, ["source_id", "quote_id"]) &&
          typeof reference.source_id === "string" && /^S[1-9][0-9]*$/.test(reference.source_id) &&
          typeof reference.quote_id === "string" && /^Q[1-9][0-9]*$/.test(reference.quote_id) &&
          own(generation.sources, reference.source_id) && own(generation.sources[reference.source_id].quotes, reference.quote_id),
        "回答引用不存在或编号非法");
      }
    }
    for (const missing of answer.missing_info) validateFreeText(missing, 240);
    validateCounts(generation.request_counts);
    requireValue(sameJson(generation.request_counts, { embedding: 0, http: 1, reranker: 0, chat: 1 }),
      "回答新增请求计数不符合单次生成合同");
    requireValue(sameJson(result.request_counts, { embedding: 1, http: 5, reranker: 1, chat: 1 }),
      "回答完成阶段累计请求计数不一致");
  }

  function validateResult(result) {
    requireValue(result && typeof result.query === "string" && result.query.trim(), "缺少已完成的问题");
    requireValue(["bm25", "hybrid"].includes(result.mode), "未知检索模式");
    for (const key of ["scope_count", "recall_k", "top_k", "rrf_constant"])
      requireValue(Number.isInteger(result[key]) && result[key] > 0, "缺少有效的检索参数");
    requireValue(result.routes && result.records && typeof result.records === "object" && !Array.isArray(result.records), "缺少候选原文");
    for (const route of ROUTES) validateHits(result.routes[route], result.records);
    validateCounts(result.request_counts);
    if (own(result, "experiment_id"))
      requireValue(typeof result.experiment_id === "string" && result.experiment_id.trim(), "实验身份非法");
    if (own(result, "rerank_failure")) {
      const failure = result.rerank_failure;
      requireValue(result.mode === "hybrid" && failure && typeof failure.message === "string" && failure.message.trim() &&
        failure.request_may_have_occurred === true && failure.request_counts_scope === "completed_stages_only",
      "重排失败经历或请求计数口径非法");
    }

    // 新响应可额外保存截取前的完整RRF并集；旧三路报告不要求有该字段。
    if (own(result, "rrf_pool")) {
      const pool = result.rrf_pool;
      const poolIds = validateHits(pool, result.records);
      if (result.mode === "bm25") {
        requireValue(pool.length === 0, "仅BM25模式不能有融合候选");
      } else {
        const union = new Set(["bm25", "dense"].flatMap(key => result.routes[key].map(hit => hit.chunk_id)));
        requireValue(poolIds.size === union.size && [...union].every(id => poolIds.has(id)), "完整融合池未覆盖两路候选并集");
        requireValue(result.routes.rrf.length === Math.min(result.top_k, pool.length) && result.routes.rrf.every(
          (hit, index) => hit.chunk_id === pool[index].chunk_id && hit.rank === pool[index].rank && hit.score === pool[index].score
        ), "展示融合结果与完整池不一致");
      }
    }

    const hasReranker = own(result.routes, "reranker");
    requireValue(hasReranker === own(result, "rerank_comparison"), "重排结果或对照契约缺失");
    if (hasReranker) {
      requireValue(own(result, "rrf_pool") && result.rrf_pool.length > 0, "重排缺少完整输入候选池");
      requireValue(typeof result.experiment_id === "string" && result.experiment_id.trim(), "重排缺少实验身份");
      const pool = result.rrf_pool, comparison = result.rerank_comparison;
      const rerankedIds = validateHits(result.routes.reranker, result.records);
      requireValue(comparison && comparison.candidate_count === pool.length && rerankedIds.size === pool.length &&
        pool.every(hit => rerankedIds.has(hit.chunk_id)), "重排未完整覆盖输入候选");
      requireValue(Number.isInteger(comparison.final_k) && comparison.final_k > 0 && comparison.final_k === result.top_k,
        "重排最终截取参数不一致");
      requireValue(typeof comparison.model === "string" && comparison.model.trim() && typeof comparison.notice === "string",
        "重排模型或边界说明缺失");
      requireValue(Array.isArray(comparison.input_order) && comparison.input_order.length === pool.length &&
        pool.every((hit, index) => comparison.input_order[index] === hit.chunk_id), "重排输入顺序与完整融合池不一致");
      const poolMap = new Map(pool.map((hit, index) => [hit.chunk_id, { hit, index }]));
      for (const hit of result.routes.reranker) {
        const original = poolMap.get(hit.chunk_id);
        requireValue(hit.input_index === original.index && hit.rrf_rank === original.hit.rank &&
          hit.rank_change === hit.rrf_rank - hit.rank, "重排名次映射或升降不一致");
      }
      validateCounts(comparison.request_counts);
      requireValue(comparison.request_counts.embedding === 0 && comparison.request_counts.http === 1 &&
        comparison.request_counts.reranker === 1 && comparison.request_counts.chat === 0, "重排新增请求计数不符合单次重排契约");
      requireValue(["embedding", "http", "reranker", "chat"].every(
        key => result.request_counts[key] >= comparison.request_counts[key]
      ), "累计请求计数小于重排新增计数");
    }
    if (own(result, "answer_failure")) {
      const failure = result.answer_failure;
      requireValue(hasReranker && failure && typeof failure.message === "string" && failure.message.trim() &&
        failure.request_may_have_occurred === true && failure.request_counts_scope === "completed_stages_only",
      "回答失败经历或请求计数口径非法");
    }
    if (own(result, "answer_generation")) validateAnswer(result);
  }

  function isoTime(value) {
    requireValue(typeof value === "string" && Number.isFinite(Date.parse(value)), "缺少有效时间");
    return new Date(value).toISOString();
  }

  function buildReport(result, { receivedAt, exportedAt, selectedChunkId = null } = {}) {
    validateResult(result);
    requireValue(selectedChunkId === null || (typeof selectedChunkId === "string" && own(result.records, selectedChunkId)), "选中块不在本轮结果中");
    // 完整保留服务已返回的原文、精度和元数据；不从可见卡片截取，不混入plan或其他轮次。
    const snapshot = JSON.parse(JSON.stringify(result, (key, value) => {
      requireValue(typeof value !== "number" || Number.isFinite(value), "结果包含非有限数");
      requireValue(!["undefined", "function", "symbol", "bigint"].includes(typeof value), "结果不是完整JSON数据");
      return value;
    }));
    return {
      schema_version: SCHEMA,
      browser_received_at: isoTime(receivedAt),
      exported_at: isoTime(exportedAt),
      selected_chunk_id: selectedChunkId,
      result: snapshot,
    };
  }

  function cell(value) {
    return String(value ?? "未记录").replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/\\/g, "\\\\").replace(/\|/g, "\\|").replace(/[\r\n]+/g, " ");
  }

  function summaryText(report) {
    requireValue(report && report.schema_version === SCHEMA, "未知实验报告版本");
    const data = report.result;
    validateResult(data);
    const maps = Object.fromEntries(ROUTES.map(key => [key, new Map(data.routes[key].map(hit => [hit.chunk_id, hit]))]));
    const ids = [...new Set(["rrf", "bm25", "dense"].flatMap(key => data.routes[key].map(hit => hit.chunk_id)))];
    const rankCell = (key, id) => {
      if (data.mode === "bm25" && key !== "bm25") return "未运行";
      const hit = maps[key].get(id);
      return hit ? `#${hit.rank} / ${hit.score.toFixed(key === "rrf" ? 8 : 6)}`
        : key === "rrf" ? `未入融合前${data.top_k}` : `未入召回前${data.recall_k}`;
    };
    const lines = [
      "# Kubernetes 检索实验摘要", "",
      `问题：${cell(data.query)}`,
      `模式：${data.mode === "hybrid" ? "BM25 + Dense + RRF" + (data.routes.reranker ? " + Reranker（完整候选池重排）" : "") + (data.answer_generation ? " + 带引用回答" : "") : "仅BM25（Dense/RRF未运行）"}`,
      `浏览器收到结果时间：${report.browser_received_at}（不是服务端推理时间）`,
      `导出时间：${report.exported_at}`,
      `参数：范围${data.scope_count}块；每路最多${data.recall_k}块；融合最多${data.top_k}块；RRF常数${data.rrf_constant}`,
      `实际候选：BM25 ${data.routes.bm25.length}；Dense ${data.routes.dense.length}；RRF ${data.routes.rrf.length}；去重并集${ids.length}块`,
      `问题分词：${Array.isArray(data.tokens) ? data.tokens.map(cell).join(" / ") : "未记录"}`,
      `实际请求：Embedding ${data.request_counts.embedding}；HTTP ${data.request_counts.http}；Reranker ${data.request_counts.reranker}；聊天 ${data.request_counts.chat}`,
      `当前选中块：${cell(report.selected_chunk_id || "无")}`, "",
      ...(data.rerank_failure ? [
        `重排失败经历：${cell(data.rerank_failure.message)}`,
        "失败请求可能已发送，处理量与费用未知；以上请求计数仅覆盖完成阶段，不包含未知的失败请求。手动重试成功也不会抹去这次失败经历。", "",
      ] : []),
      ...(data.answer_failure ? [
        `回答生成失败经历：${cell(data.answer_failure.message)}`,
        "失败请求可能已发送，处理量与费用未知；以上请求计数仅覆盖完成阶段，不包含未知的失败请求。手动重试成功也不会抹去这次失败经历。", "",
      ] : []),
      "## 完整候选排名（含折叠候选；名次 / 原路线分数）", "",
      "| chunk_id | 标题 / 章节 | BM25 | Dense | RRF |",
      "|---|---|---|---|---|",
    ];
    for (const id of ids) {
      const metadata = data.records[id].metadata || {};
      const paths = Array.isArray(metadata.sections) ? metadata.sections.filter(Array.isArray).map(path => path.join(" / ")).join("；") : "";
      lines.push(`| ${cell(id)} | ${cell(paths || metadata.title || "未记录")} | ${rankCell("bm25", id)} | ${rankCell("dense", id)} | ${rankCell("rrf", id)} |`);
    }
    if (!ids.length) lines.push("| 无候选 | — | — | — | — |");
    if (data.routes.rrf.length) {
      lines.push("", "## RRF贡献（只融合排名，不混加原始分数）", "");
      for (const hit of data.routes.rrf) {
        const parts = ["bm25", "dense"].map(key => {
          const rank = hit.ranks?.[key], contribution = hit.contributions?.[key];
          return Number.isInteger(rank) && Number.isFinite(contribution)
            ? `${key} #${rank}：1/(${data.rrf_constant}+${rank})=${contribution.toFixed(8)}`
            : `${key} 未入召回列表，贡献0`;
        });
        lines.push(`- ${cell(hit.chunk_id)}：${parts.join("；")}；合计=${hit.score.toFixed(8)}`);
      }
    }
    if (data.routes.reranker) {
      const comparison = data.rerank_comparison;
      const counts = comparison.request_counts;
      const cutoff = comparison.final_k;
      lines.push("", "## 完整候选重排对照（截取前RRF全池 → Reranker）", "",
        `重排模型：${cell(comparison.model)}`,
        `实验身份：${cell(data.experiment_id)}`,
        `输入范围：完整两路去重并集${comparison.candidate_count}块，不是先取RRF前${cutoff}块；重排后最终取前${cutoff}块。`,
        `本次成功重排新增请求：Embedding ${counts.embedding}；HTTP ${counts.http}；Reranker ${counts.reranker}；聊天 ${counts.chat}（上方实际请求为完成阶段累计，不重复相加；失败请求另计，可能未包含）`,
        "发送边界：云端重排只接收问题和候选正文，不发送候选元数据；复制/导出不再发送这些内容。",
        "名次升降以完整RRF池为基准；新进/退出仅表示Top-K成员变化，无人工相关性标注，不能据此宣称更好。", "",
        "| 重排候选 chunk_id | 全池RRF名次 / 分数 | Reranker名次 / 分数 | 名次变化 | 最终Top-K变化 |",
        "|---|---|---|---|---|");
      const poolMap = new Map(data.rrf_pool.map(hit => [hit.chunk_id, hit]));
      for (const hit of data.routes.reranker) {
        const old = poolMap.get(hit.chunk_id);
        const change = hit.rank_change > 0 ? `上升${hit.rank_change}` : hit.rank_change < 0 ? `下降${-hit.rank_change}` : "不变";
        const topChange = old.rank > cutoff && hit.rank <= cutoff ? `新进Top${cutoff}`
          : old.rank <= cutoff && hit.rank > cutoff ? `退出Top${cutoff}`
            : hit.rank <= cutoff ? `保留Top${cutoff}` : `仍在Top${cutoff}外`;
        lines.push(`| ${cell(hit.chunk_id)} | #${old.rank} / ${old.score.toFixed(8)} | #${hit.rank} / ${hit.score.toFixed(6)} | ${change} | ${topChange} |`);
      }
      lines.push("", `重排说明：${cell(comparison.notice)}`,
        "重排边界：Reranker与BM25、余弦、RRF原始分数不可混加；重排分数不是概率或正确率；重排不能找回两路都未召回的文本。");
    }
    if (data.answer_generation) {
      const generation = data.answer_generation, answer = generation.answer, counts = generation.request_counts;
      const labels = { answered: "依据候选作答，仍需人工核验", partial: "部分回答，仍有资料缺口", insufficient_evidence: "证据不足，无法据此回答" };
      lines.push("", "## 带引用回答（重排前5块 → 生成 → 结构与引用定位校验）", "",
        `回答模型：${cell(generation.model)}；最大输出Token=${generation.max_tokens}`,
        `回答状态：${answer.status}（${labels[answer.status]}）`,
        `本次成功生成新增请求：Embedding ${counts.embedding}；HTTP ${counts.http}；Reranker ${counts.reranker}；聊天 ${counts.chat}（上方为完成阶段累计，不重复相加；未知失败请求不包含在内）`,
        "选中证据：");
      for (const [sourceId, source] of Object.entries(generation.sources)) {
        const metadata = source.metadata;
        lines.push(`- [${sourceId}] ${cell(source.chunk_id)}；${cell(metadata.title || "标题未记录")}；${cell(metadata.source_url || "来源URL未记录")}；版本=${cell(metadata.kubernetes_release || "未确认")}；审核=${cell(metadata.review_status || "未记录")}`);
      }
      lines.push("", "回答要点：");
      if (!answer.claims.length) lines.push("现有证据不足，本轮没有可展示的有引用要点。");
      for (const [index, claim] of answer.claims.entries()) {
        const references = claim.evidence.map(reference => `[${reference.source_id}/${reference.quote_id}]`);
        lines.push(`${index + 1}. ${cell(claim.text)} ${references.join(" ")}`);
        for (const reference of claim.evidence) {
          const quote = generation.sources[reference.source_id].quotes[reference.quote_id];
          const preview = Array.from(quote).slice(0, 240).join("");
          lines.push(`   原文[${reference.source_id}/${reference.quote_id}]：${cell(preview)}${Array.from(quote).length > 240 ? "…（摘要截短；完整原文见JSON）" : ""}`);
        }
      }
      if (answer.missing_info.length) {
        lines.push("", "还需补充或核实：");
        for (const item of answer.missing_info) lines.push(`- ${cell(item)}`);
      }
      lines.push("", `回答说明：${cell(generation.notice)}`,
        "回答校验边界：仅检查结构、引用身份及摘录能否定位到原文，不等于已验证推断被原文充分支持，也不是正确率评测。文档未审核、版本适用性仍需核实；示例名称、IP、输出不能当作现场值；命令只作为文本展示，不执行。复制、原文核对与导出不再请求模型。");
    }
    lines.push("", "## 来源 / 快照 / 审核", "");
    const sources = new Set();
    for (const id of ids) {
      const metadata = data.records[id].metadata || {};
      // 摘要不主动附带本机source绝对路径；完整JSON仍保留原始元数据。
      const source = [metadata.source_url || metadata.corpus_relative_path || "来源未记录", metadata.snapshot_commit || "快照未记录", metadata.review_status || "审核未记录"];
      const key = JSON.stringify(source);
      if (!sources.has(key)) lines.push(`- ${source.map(cell).join("；")}`);
      sources.add(key);
    }
    lines.push("", "边界：分数不是概率或准确率；不同路线的原始分数不可横向比较；同分次序不代表内容优劣。摘要不包含候选全文，逐字核对请使用完整JSON。复制/导出不重新检索或请求模型。", "");
    return lines.join("\n");
  }

  function downloadName(report) {
    requireValue(report && report.schema_version === SCHEMA, "未知实验报告版本");
    return `k8s-retrieval-${isoTime(report.exported_at).replace(/[-:.]/g, "")}.json`;
  }

  const api = Object.freeze({ buildReport, summaryText, downloadName });
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.RetrievalExport = api;
})(typeof globalThis === "object" ? globalThis : this);
