"""第五步：对真实知识库分块进行 BM25 中文关键词检索，不调用模型。"""

import argparse
import logging
from math import isfinite
import re
import sys

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from qa_platform.ingestion.railway import load_records, split_records


DEFAULT_QUERY = "电机还在转，但丝杆一点也不动，可能是什么原因？"
# 英文缩写、数字和带连接符的编号整体保留；中文片段交给 jieba。
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
STOPWORDS = frozenset(
    {
        "的",
        "了",
        "吗",
        "呢",
        "是",
        "有",
        "和",
        "与",
        "或",
        "在",
        "还",
        "也",
        "都",
        "我",
        "你",
        "它",
        "请",
        "一下",
        "什么",
        "为什么",
        "怎么",
        "如何",
        "哪些",
        "可能",
    }
)
DOMAIN_WORDS = ("转辙机", "丝杆", "滑床板", "联轴器", "表示杆", "动作杆")
jieba.setLogLevel(logging.WARNING)
TOKENIZER = jieba.Tokenizer()
for domain_word in DOMAIN_WORDS:
    # 少量领域词避免“丝杆”被拆成“丝 / 杆”，不依赖新词猜测。
    TOKENIZER.add_word(domain_word)


def tokenize(text: str) -> list[str]:
    """查询与文档共用的分词规则：去标点、英文小写、少量停用词过滤。"""
    tokens = []
    for match in TOKEN_PATTERN.finditer(text):
        span = match.group()
        words = (
            [span.lower()]
            if span.isascii()
            else TOKENIZER.lcut_for_search(span, HMM=False)
        )
        tokens.extend(word for word in words if word not in STOPWORDS)
    # 不删除“无、不、正常”等条件词，也不对文档词去重：BM25需要词频。
    return tokens


class BM25Index:
    """把分词结果、评分器和原始 Document 放在一起，保持位置对应。"""

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("没有可建立索引的文本块。")
        self.documents = list(documents)
        corpus = [tokenize(document.page_content) for document in self.documents]
        if any(not tokens for tokens in corpus):
            raise ValueError("存在分词后为空的文本块，请检查知识库内容。")
        self.document_terms = [set(tokens) for tokens in corpus]
        # 使用 BM25Okapi 的常用默认参数，当前先不进行参数调优。
        self.ranker = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)

    def search(
        self, query_tokens: list[str], top_k: int = 3
    ) -> list[tuple[Document, float, list[str]]]:
        """仅保留有词汇交集的块，再按原始 BM25 分数稳定排序。"""
        if not query_tokens:
            raise ValueError("问题分词后为空，请输入有实际内容的问题。")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数。")

        # 重复提问词不额外加权；文档中的重复词仍参与词频计算。
        query_terms = list(dict.fromkeys(query_tokens))
        scores = self.ranker.get_scores(query_terms)
        if len(scores) != len(self.documents) or not all(
            isfinite(float(score)) for score in scores
        ):
            raise ValueError("BM25 返回的分数数量错误或包含非有限值。")

        hits = []
        for document, terms, score in zip(self.documents, self.document_terms, scores):
            matched = [term for term in query_terms if term in terms]
            if matched:
                hits.append((document, float(score), matched))
        # 零分、负分仍可能有关键词命中；不用 score > 0 作为筛选条件。
        # Python稳定排序让同分结果沿用原始文档顺序。
        hits.sort(key=lambda hit: -hit[1])
        return hits[:top_k]


def main() -> int:
    """读取、分块、建立内存索引、检索并打印候选，不生成诊断答案。"""
    parser = argparse.ArgumentParser(description="验证真实知识库的 BM25 关键词检索")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="检索问题")
    parser.add_argument(
        "--kb",
        choices=["test", "experience", "all"],
        default="test",
        help="选择知识库；默认仅测试库",
    )
    parser.add_argument("--top-k", type=int, default=3, help="最多返回几个检索子块")
    parser.add_argument(
        "--chunk-size", type=int, default=800, help="子块最大字符数，包含标题"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="同一父条目内子块的目标重叠字符数",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k 必须大于 0")
    if args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("必须满足 chunk_size > 0 且 0 <= chunk_overlap < chunk_size")

    query_tokens = tokenize(args.query)
    if not query_tokens:
        parser.error("问题分词后为空，请输入有实际内容的问题。")
    try:
        records = load_records(args.kb)
        chunks = split_records(records, args.chunk_size, args.chunk_overlap)
        index = BM25Index(chunks)
        hits = index.search(query_tokens, args.top_k)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[错误] BM25 检索失败：{exc}", file=sys.stderr)
        return 1

    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] 知识库：{args.kb}；Top-K：{args.top_k}")
    print(
        f"[配置] 最大块大小：{args.chunk_size} 字符；目标重叠：{args.chunk_overlap} 字符"
    )
    print(
        f"[索引] 父记录 {len(records)} 条，检索子块 {len(chunks)} 个；仅保存在本次进程内存中"
    )
    if args.kb in {"test", "all"}:
        print("[提醒] 本次范围包含测试资料，候选内容不应直接作为设备维护规程。")
    if args.kb in {"experience", "all"}:
        print("[提醒] 经验库内容尚未审核，关键词匹配不代表内容已经获得专业确认。")
    print(f"\n[问题] {args.query}")
    print(f"[分词] {' / '.join(query_tokens)}")

    if not hits:
        print("\n[结果] 没有与问题分词存在词汇交集的文本块。")
        print("[说明] 这不等于知识库没有相关知识；换一种说法可能需要向量检索。")
    for rank, (document, score, matched) in enumerate(hits, start=1):
        metadata = document.metadata
        label = metadata["entry_id"] or metadata["title"]
        print(f"\n[候选 {rank}] {label} | BM25={score:.4f}")
        print(f"[命中词] {' / '.join(matched)}")
        print(
            f"[来源] {metadata['source']}；父记录第 {metadata['line_start']}–{metadata['line_end']} 行"
        )
        print(
            f"[块] {metadata['chunk_id']}；父记录 {metadata['parent_id']}；第 {metadata['part_index'] + 1}/{metadata['part_count']} 块"
        )
        print(document.page_content)

    print(
        "\n[说明] BM25 分数是当前索引中的关键词排序分数，不是概率，也不能直接与向量或重排分数比较。"
    )
    print("[成功] BM25 检索完成，未调用模型、未写入数据库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
