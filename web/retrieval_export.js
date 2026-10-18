/* 已完成结果 -> 可复制摘要/完整JSON。纯数据函数，不访问网络、文件或剪贴板。 */
(function (root) {
  "use strict";
  const SCHEMA = "retrieval-lab-report/v1";
  const ROUTES = ["bm25", "dense", "rrf"];
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  const requireValue = (condition, message) => { if (!condition) throw new Error(message); };

  function validateResult(result) {
    requireValue(result && typeof result.query === "string" && result.query.trim(), "缺少已完成的问题");
    requireValue(["bm25", "hybrid"].includes(result.mode), "未知检索模式");
    for (const key of ["scope_count", "recall_k", "top_k", "rrf_constant"])
      requireValue(Number.isInteger(result[key]) && result[key] > 0, "缺少有效的检索参数");
    requireValue(result.routes && result.records && typeof result.records === "object" && !Array.isArray(result.records), "缺少候选原文");
    for (const route of ROUTES) {
      const hits = result.routes[route];
      requireValue(Array.isArray(hits), "缺少完整路线结果");
      const seen = new Set();
      for (const [position, hit] of hits.entries()) {
        requireValue(hit && typeof hit.chunk_id === "string" && !seen.has(hit.chunk_id), "候选身份重复或缺失");
        requireValue(hit.rank === position + 1 && Number.isFinite(hit.score), "候选名次或分数非法");
        requireValue(own(result.records, hit.chunk_id) && typeof result.records[hit.chunk_id].text === "string", "缺少候选完整正文");
        seen.add(hit.chunk_id);
      }
    }
    requireValue(result.request_counts && ["embedding", "http", "reranker", "chat"].every(
      key => Number.isInteger(result.request_counts[key]) && result.request_counts[key] >= 0
    ), "缺少实际请求计数");
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
      `模式：${data.mode === "hybrid" ? "BM25 + Dense + RRF" : "仅BM25（Dense/RRF未运行）"}`,
      `浏览器收到结果时间：${report.browser_received_at}（不是服务端推理时间）`,
      `导出时间：${report.exported_at}`,
      `参数：范围${data.scope_count}块；每路最多${data.recall_k}块；融合最多${data.top_k}块；RRF常数${data.rrf_constant}`,
      `实际候选：BM25 ${data.routes.bm25.length}；Dense ${data.routes.dense.length}；RRF ${data.routes.rrf.length}；去重并集${ids.length}块`,
      `问题分词：${Array.isArray(data.tokens) ? data.tokens.map(cell).join(" / ") : "未记录"}`,
      `实际请求：Embedding ${data.request_counts.embedding}；HTTP ${data.request_counts.http}；Reranker ${data.request_counts.reranker}；聊天 ${data.request_counts.chat}`,
      `当前选中块：${cell(report.selected_chunk_id || "无")}`, "",
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
