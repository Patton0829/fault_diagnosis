"""相邻块上下文补全：保留命中身份，限制来源/章节边界，预算内整块加入。

默认只显示计划；显式--seed-chunk可零HTTP复现指定起点，--run才展示正文。
未指定起点时，--run复用真实问题检索一次；补全不额外请求模型或数据库。
"""

import argparse
from copy import deepcopy
from math import isfinite
from pathlib import Path
import sys

from qa_platform.ingestion.embedding_plan import canonical, require


SOURCE_FIELDS = (
    "corpus_relative_path",
    "strategy",
    "target_chars",
    "source",
    "source_url",
    "cleaned_sha256",
    "raw_sha256",
    "kubernetes_release",
    "license",
    "cleaner_version",
    "cleanup_version",
    "builder_version",
    "splitter_version",
    "offset_scope",
)
OFFSET_SCOPE = "cleaned_document_python_characters_half_open"


def source_key(row: dict) -> str:
    metadata = row["metadata"]
    return canonical(
        {
            **{
                field: row[field]
                for field in ("parent_id", "kb_id", "snapshot_commit", "review_status")
            },
            **{field: metadata.get(field) for field in SOURCE_FIELDS},
        }
    )


def group_key(row: dict) -> str:
    return canonical([source_key(row), row["metadata"]["sections"]])


def validate_records(records: list) -> tuple[dict, dict]:
    """本步只处理正文与清洗后字符区间一一对应、单章节的块。"""
    require(isinstance(records, list) and bool(records), "上下文候选为空")
    by_id, by_position = {}, {}
    for row in records:
        require(
            isinstance(row, dict)
            and {
                "metadata",
                "chunk_id",
                "text",
                "parent_id",
                "kb_id",
                "snapshot_commit",
                "review_status",
            }
            <= set(row),
            "块缺少正文、身份或metadata字段",
        )
        metadata = row["metadata"]
        require(isinstance(metadata, dict), "块metadata必须是对象")
        chunk_id = row["chunk_id"]
        require(
            isinstance(chunk_id, str) and chunk_id and chunk_id not in by_id,
            "块ID重复或非法",
        )
        require(isinstance(row["text"], str) and row["text"].strip(), "块正文为空")
        for field in (
            "chunk_id",
            "parent_id",
            "kb_id",
            "snapshot_commit",
            "review_status",
        ):
            require(
                isinstance(row.get(field), str)
                and row[field]
                and metadata.get(field) == row[field],
                f"块显式身份与元数据不一致：{field}",
            )
        index, count = metadata.get("chunk_index"), metadata.get("chunk_count")
        start, end = (
            metadata.get("cleaned_char_start"),
            metadata.get("cleaned_char_end"),
        )
        require(
            type(index) is int and type(count) is int and 0 <= index < count,
            "块序号/总数非法",
        )
        require(
            type(start) is int
            and type(end) is int
            and 0 <= start < end
            and end - start == len(row["text"]),
            "正文长度与清洗后字符区间不匹配",
        )
        require(
            metadata.get("offset_scope") == OFFSET_SCOPE
            and metadata.get("heading_context_ranges") == [],
            "本步不支持额外标题前缀或未知坐标系，不猜测正文位置",
        )
        sections = metadata.get("sections")
        require(
            isinstance(sections, list)
            and len(sections) == 1
            and isinstance(sections[0], list)
            and sections[0]
            and all(isinstance(part, str) and part for part in sections[0]),
            "本步只接受明确的单章节路径",
        )
        require(
            all(
                isinstance(metadata.get(field), str) and metadata[field]
                for field in ("source", "source_url", "cleaned_sha256")
            ),
            "缺少来源或清洗版本指纹",
        )
        position = (row["parent_id"], index)
        require(position not in by_position, "同一父文档出现重复块序号，不猜测相邻块")
        by_id[chunk_id], by_position[position] = row, row
    return by_id, by_position


def merge_passages(selected: set, by_id: dict, ranks: dict) -> list:
    """只合并同来源/章节且字符区间连续或重叠的正文；重叠须逐字相同。"""
    rows = sorted(
        (by_id[key] for key in selected),
        key=lambda row: (
            group_key(row),
            row["metadata"]["cleaned_char_start"],
            row["metadata"]["chunk_index"],
        ),
    )
    passages = []
    for row in rows:
        metadata = row["metadata"]
        start, end, identity = (
            metadata["cleaned_char_start"],
            metadata["cleaned_char_end"],
            group_key(row),
        )
        previous = passages[-1] if passages else None
        if (
            previous is not None
            and previous["identity"] == identity
            and start <= previous["end"]
        ):
            overlap = min(previous["end"] - start, len(row["text"]))
            offset = start - previous["start"]
            require(
                previous["text"][offset : offset + overlap] == row["text"][:overlap],
                "重叠区文字不一致，拒绝拼接",
            )
            if end > previous["end"]:
                previous["text"] += row["text"][overlap:]
                previous["end"] = end
            previous["chunk_ids"].append(row["chunk_id"])
        else:
            passages.append(
                {
                    "identity": identity,
                    "parent_id": row["parent_id"],
                    # 扩展段不是首块：不把首块的ID、序号或旧坐标伪装成合并后身份。
                    "metadata": {
                        field: deepcopy(metadata[field])
                        for field in (
                            *SOURCE_FIELDS,
                            "title",
                            "sections",
                            "parent_id",
                            "kb_id",
                            "snapshot_commit",
                            "review_status",
                        )
                        if field in metadata
                    },
                    "metadata_scope": "shared_source_and_section_not_original_chunk",
                    "start": start,
                    "end": end,
                    "text": row["text"],
                    "chunk_ids": [row["chunk_id"]],
                }
            )
    for passage in passages:
        passage["seed_ranks"] = sorted(
            ranks[key] for key in passage["chunk_ids"] if key in ranks
        )
        require(bool(passage["seed_ranks"]), "出现无法连续连接原命中的孤立补充块")
        require(
            len(passage["text"]) == passage["end"] - passage["start"],
            "合并后字符区间不一致",
        )
    return sorted(
        passages, key=lambda passage: (min(passage["seed_ranks"]), passage["start"])
    )


def expand_context(
    hits: list, records: list, window: int = 1, max_context_chars: int = 6000
) -> dict:
    require(type(window) is int and 0 <= window <= 3, "window须为0～3")
    require(
        type(max_context_chars) is int and 1 <= max_context_chars <= 50000,
        "max-context-chars须为1～50000",
    )
    by_id, by_position = validate_records(records)
    require(isinstance(hits, list) and hits, "没有初始命中，不能补全")
    ranks, seed_ids = {}, []
    for rank, hit in enumerate(hits, 1):
        require(
            isinstance(hit, dict) and {"chunk_id", "record", "score"} <= set(hit),
            "初始命中缺少身份、正文快照或分数字段",
        )
        key = hit["chunk_id"]
        require(
            key in by_id and key not in ranks and hit["record"] == by_id[key],
            "初始命中未知、重复或与快照不符",
        )
        score = hit["score"]
        require(
            score is None
            or (
                type(score) in (int, float)
                and isfinite(score)
                and -1.00001 <= score <= 1.00001
            ),
            "初始分数非法",
        )
        ranks[key] = rank
        seed_ids.append(key)
    selected = set(seed_ids)
    baseline = merge_passages(selected, by_id, ranks)
    baseline_unique_chars = sum(len(p["text"]) for p in baseline)
    require(
        baseline_unique_chars <= max_context_chars,
        f"原始命中去重正文需{baseline_unique_chars}字符，超过预算{max_context_chars}；不丢弃原命中、不截断，请增大预算",
    )
    events, blocked = [], set()
    frontiers = {
        (key, direction): by_id[key] for key in seed_ids for direction in (-1, 1)
    }
    # 先近后远；同一距离按原命中顺序分配预算，不宣称这是最优分配。
    for distance in range(1, window + 1):
        for key in seed_ids:
            anchor = by_id[key]
            for direction in (-1, 1):
                route = (key, direction)
                if route in blocked:
                    continue
                current = frontiers[route]
                index = anchor["metadata"]["chunk_index"] + direction * distance
                neighbor = by_position.get((anchor["parent_id"], index))
                event = {
                    "seed_id": key,
                    "seed_rank": ranks[key],
                    "direction": direction,
                    "distance": distance,
                    "neighbor_id": neighbor["chunk_id"] if neighbor else None,
                }
                reason = None
                if neighbor is None:
                    reason = "missing_or_outside_filter"
                elif source_key(neighbor) != source_key(anchor):
                    reason = "different_source_or_version"
                elif neighbor["metadata"]["sections"] != anchor["metadata"]["sections"]:
                    reason = "different_section"
                elif (
                    neighbor["metadata"]["chunk_count"]
                    != anchor["metadata"]["chunk_count"]
                ):
                    reason = "different_chunk_count"
                elif (
                    neighbor["metadata"]["cleaned_char_start"]
                    > current["metadata"]["cleaned_char_end"]
                    or current["metadata"]["cleaned_char_start"]
                    > neighbor["metadata"]["cleaned_char_end"]
                ):
                    reason = "non_contiguous_range"
                if reason:
                    events.append({**event, "status": "skipped", "reason": reason})
                    blocked.add(route)
                    continue
                candidate = neighbor["chunk_id"]
                if candidate in selected:
                    events.append(
                        {**event, "status": "reused", "reason": "already_selected"}
                    )
                    frontiers[route] = neighbor
                    continue
                trial = selected | {candidate}
                passages = merge_passages(trial, by_id, ranks)
                chars = sum(len(p["text"]) for p in passages)
                if chars > max_context_chars:
                    events.append(
                        {**event, "status": "skipped", "reason": "budget_exceeded"}
                    )
                    blocked.add(route)
                    continue
                selected = trial
                frontiers[route] = neighbor
                events.append(
                    {**event, "status": "added", "reason": "adjacent_same_section"}
                )
    passages = merge_passages(selected, by_id, ranks)
    selected_ids = [key for p in passages for key in p["chunk_ids"]]
    return {
        "hits": deepcopy(hits),
        "seed_ids": seed_ids,
        "selected_ids": selected_ids,
        "added_ids": [key for key in selected_ids if key not in ranks],
        "baseline_chars": sum(len(by_id[key]["text"]) for key in seed_ids),
        "baseline_unique_chars": baseline_unique_chars,
        "expanded_chars": sum(len(p["text"]) for p in passages),
        "max_context_chars": max_context_chars,
        "events": events,
        "passages": passages,
    }


REASONS = {
    "missing_or_outside_filter": "无相邻块或不在本次过滤范围",
    "different_source_or_version": "来源/版本/审核身份不同",
    "different_section": "跨章节，保守跳过",
    "different_chunk_count": "父文档块总数不一致",
    "non_contiguous_range": "原文区间不连续，不跨缺口拼接",
    "already_selected": "已经是初始命中或已补入，不重复加入",
    "budget_exceeded": "正文预算不足，整块跳过且不越过它继续延伸",
    "adjacent_same_section": "同来源同章节连续相邻块",
}


def print_comparison(
    result: dict, records: list, preview_chars: int, manual: bool
) -> None:
    by_id = {row["chunk_id"]: row for row in records}
    print(
        "\n[A：不补上下文，保持原始输入顺序]"
        if manual
        else "\n[A：不补上下文，保持原始检索排名]"
    )
    for rank, hit in enumerate(result["hits"], 1):
        row = hit["record"]
        score = (
            "未评分（手工指定，不是本次检索结果）"
            if hit["score"] is None
            else f"COSINE={hit['score']:.7f}"
        )
        print(
            f"\n[原块 {rank}] {row['chunk_id']} | index={row['metadata']['chunk_index']} | {score}"
        )
        print(row["text"][:preview_chars])
        if len(row["text"]) > preview_chars:
            print(f"[仅展示截断] {preview_chars}/{len(row['text'])}字符；原块未截断。")
    print("\n[补全账本] 邻块没有单独相关性分数；补全不改变原始命中排名/分数。")
    for event in result["events"]:
        side = "前" if event["direction"] < 0 else "后"
        print(
            f"起点{event['seed_rank']} {side}{event['distance']}块：{event['neighbor_id'] or '无'} | {event['status']} | {REASONS[event['reason']]}"
        )
    print("\n[B：补全后按来源位置组织，不是重新相关性排序]")
    ranks = {key: rank for rank, key in enumerate(result["seed_ids"], 1)}
    for number, passage in enumerate(result["passages"], 1):
        metadata = passage["metadata"]
        print(
            f"\n[上下文 {number}] {metadata.get('title', '无标题')} | parent_id={passage['parent_id']} | 初始命中={passage['seed_ranks']}"
        )
        sequence = [
            f"{by_id[key]['metadata']['chunk_index']}:{key}({'原块' + str(ranks[key]) if key in ranks else '补充邻块'})"
            for key in passage["chunk_ids"]
        ]
        print("[顺序] " + " → ".join(sequence))
        print("[章节] " + " / ".join(metadata["sections"][0]))
        print(f"[来源] {metadata['source_url']}")
        print(
            f"[版本/审核] {metadata['snapshot_commit']}；{metadata['review_status']}；release={metadata.get('kubernetes_release') or '未确认'}"
        )
        print(
            f"[位置] 清洗后字符[{passage['start']},{passage['end']})；不是原文件行号；正文{len(passage['text'])}字符。"
        )
        print(passage["text"][:preview_chars])
        if len(passage["text"]) > preview_chars:
            print(
                f"[仅展示截断] {preview_chars}/{len(passage['text'])}字符；完整上下文保留在本轮内存中。"
            )
    print(
        f"\n[数量对照] 初始{len(result['seed_ids'])}块 → 去重后共{len(result['selected_ids'])}块（新增{len(result['added_ids'])}块）→ {len(result['passages'])}段连续上下文。"
    )
    print(
        f"[邻块尝试] {len(result['events'])}个方向/距离事件；涉及{len({event['neighbor_id'] for event in result['events'] if event['neighbor_id'] is not None})}个不同邻块ID（可能含原命中）；新增{len(result['added_ids'])}块；跳过{sum(event['status'] == 'skipped' for event in result['events'])}次。"
    )
    print(
        f"[正文字符] 原始直接拼接={result['baseline_chars']}；原命中区间去重={result['baseline_unique_chars']}；补全后={result['expanded_chars']}；预算={result['max_context_chars']}。"
    )
    print(
        "[预算边界] 只计正文字符，不含标题/来源/提示词等包装；字符不是Token。本步不声明满足模型上下文上限。"
    )
    print(
        "[结论边界] 补全不改变初始检索命中与排名，但增加最终阅读上下文；邻块未必相关，可能引入噪声；固定窗口不保证语义完整，也可能错过其他章节的必要前提。未重排、未生成回答、未执行文档命令、未改原文或数据库记录。"
    )


def retrieve_once(
    check: dict, timeout: int = 180, session_factory=None, client_factory=None
) -> list:
    """沿用已经测试的校验/编码/搜索函数，但本步每路只搜索一次。"""
    import numpy as np
    from qa_platform.storage import kubernetes_search as search

    database = check["index_dir"] / search.build.DATABASE
    require(
        database.is_dir() and not database.is_symlink(),
        "已有数据库目录缺失，不自动新建",
    )
    if client_factory is None:
        from pymilvus import MilvusClient

        client_factory = MilvusClient
    client = client_factory(uri=str(database), timeout=30)
    try:
        matrix = search.read_index(client, check["plan"])
        require(
            np.array_equal(matrix, np.asarray(check["bundle"]["vectors"], dtype="<f4")),
            "已有数据库与SQLite缓存向量不一致",
        )
        query, stats = search.encode_query(check, timeout, session_factory)
        memory = search.rank_numpy(
            check["records"], check["vectors"], query, check["top_k"]
        )
        hits = search.search_milvus(
            client, check["records"], query, check["top_k"], check["filter"]
        )
        compared = search.validate_top_k(
            hits, check["records"], check["vectors"], query, check["top_k"]
        )
        shared = len(
            {item["chunk_id"] for item in memory} & {item["chunk_id"] for item in hits}
        )
        print(
            f"[检索核验] 每路各搜索1次；Top-K交集={shared}/{len(hits)}；最大分差={compared['max_score_difference']:.3g}；允许边界近同分。"
        )
        print(
            f"[实际请求] 问题Embedding={stats['embedding_requests']}；模型HTTP={stats['http_requests']}；文档编码/重排/聊天=0；补全额外HTTP=0。"
        )
        return hits
    finally:
        client.close()


def main() -> int:
    # 延迟导入便于纯扩展规则测试；现有读取器负责冻结语料/缓存/ready核验。
    from qa_platform.storage import kubernetes_search as search

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--index-dir", type=Path, default=search.build.DEFAULT_INDEX_DIR
    )
    parser.add_argument(
        "--seed-chunk",
        action="append",
        help="手工指定起点，可重复；零HTTP示例，不是本次检索",
    )
    parser.add_argument("--query", default=search.DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, help="真实检索的Top-K；不能与手工起点混用")
    parser.add_argument("--parent-id")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--preview-chars", type=int, default=2400)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if (
        not 0 <= args.window <= 3
        or not 1 <= args.max_context_chars <= 50000
        or not 100 <= args.preview_chars <= 8000
        or not 1 <= args.timeout <= 600
    ):
        parser.error(
            "window须为0～3；max-context-chars为1～50000；preview-chars为100～8000；timeout为1～600"
        )
    if args.seed_chunk and (args.top_k is not None or len(args.seed_chunk) > 50):
        parser.error("手工起点最多50块，不能同时指定top-k")
    check = search.prepare_search(
        args.index_dir.expanduser().absolute(),
        args.query,
        args.top_k if args.top_k is not None else 3,
        args.parent_id,
    )
    by_id, _ = validate_records(check["records"])
    print(
        f"[范围] 已核验ready索引及本地缓存；可用候选{len(by_id)}块；window={args.window}；正文预算={args.max_context_chars}字符。"
    )
    print(
        "[规则] 保留全部原命中；先近后远、同距离按初始顺序补前后邻块；同父/来源版本/章节且位置连续才加入。"
    )
    if args.seed_chunk:
        require(
            all(key in by_id for key in args.seed_chunk), "指定起点不在本次候选范围内"
        )
        hits = [
            {"chunk_id": key, "record": by_id[key], "score": None}
            for key in args.seed_chunk
        ]
        result = expand_context(
            hits, check["records"], args.window, args.max_context_chars
        )
        print(
            f"[手工示例] 按给定顺序复现{len(hits)}个起点；不新检索、不伪造相似度分数；HTTP=0，未打开Milvus。"
        )
        if args.run:
            print_comparison(result, check["records"], args.preview_chars, manual=True)
        else:
            print(
                f"[计划] 可补入{len(result['added_ids'])}块；正文{result['baseline_unique_chars']}→{result['expanded_chars']}字符；加--run才展示正文。"
            )
    else:
        print(
            f"[问题] {check['query']}；Top-K={check['top_k']}；过滤={check['filter']}"
        )
        print(
            "[请求计划] 问题Embedding1次+前后digest各1次=模型HTTP3次；NumPy/Milvus各搜索1次；A/B共用同一轮命中，扩展仅查本地已核验候选，额外HTTP/数据库取块请求均0。原命中若超过正文预算则停止补全，不截断。"
        )
        if args.run:
            hits = retrieve_once(check, timeout=args.timeout)
            result = expand_context(
                hits,
                check["records"],
                args.window,
                args.max_context_chars,
            )
            print_comparison(result, check["records"], args.preview_chars, manual=False)
    if not args.run:
        print(
            "[计划完成] 本轮实际HTTP=0；未导入Milvus SDK、未创建HTTP/数据库客户端、未写文件。"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[中断] 不自动重试，未改数据库记录。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        message = (
            str(error)
            if isinstance(error, (ValueError, OSError))
            else "检查索引、块元数据或本地模型；不显示HTTP响应正文"
        )
        print(
            f"[停止] {type(error).__name__}：{message}；不猜测、不覆盖或自动重建。",
            file=sys.stderr,
        )
        raise SystemExit(1)
