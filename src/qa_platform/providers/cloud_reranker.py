"""第三步：让 Reranker 根据问题重新排列候选知识片段。"""

import argparse
import math
import os
import sys
from time import perf_counter
from typing import SupportsFloat, SupportsIndex

import httpx
from dotenv import load_dotenv

from qa_platform.settings import ENV_PATH, PROJECT_ROOT as PROJECT_ROOT


API_BASE_URL = "https://maas.qianwenaiapi.com/api/v1"
API_URL = f"{API_BASE_URL}/services/rerank/text-rerank/text-rerank"
MODEL_NAME = "qwen3.7-text-rerank"
DEFAULT_QUERY = "只有A轴振动出现连续尖峰，另外两轴和三相电流都正常，优先检查什么？"

# 与 cloud_embedding.py 使用相同的三条手工摘录，便于对照学习。
# 这些是测试材料；本步骤尚未执行知识库解析、入库或真实候选召回。
DOCUMENTS = [
    {
        "id": "TEST-SW-002",
        "text": "转辙机三相电流不平衡，伴随异常振动，可能与缺相、接线端子接触不良、"
        "电机绕组异常或电流传感器配置有关。",
    },
    {
        "id": "TEST-SW-003",
        "text": "只有A轴加速度出现连续尖峰，其他轴变化较小且三相电流正常时，"
        "可能与传感器固定松动、安装面间隙或线缆干扰有关，优先检查传感器安装和信号线路。",
    },
    {
        "id": "TEST-SW-006",
        "text": "电机转速正常但丝杆转速为零，可能与联轴器松脱、传动部件损坏或"
        "丝杆转速传感器故障有关，需要区分丝杆实际停转与采集异常。",
    },
]


def rerank_documents(
    client: httpx.Client, query: str, documents: list[str], top_n: int
) -> list[tuple[int, float]]:
    """发送重排请求，返回按分数降序排列的 (原始索引, 相关性分数)。"""
    response = client.post(
        API_URL,
        json={
            "model": MODEL_NAME,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": top_n},
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code"):
        raise ValueError("服务未返回成功的重排响应。")
    output = payload.get("output")
    if not isinstance(output, dict):
        raise ValueError("响应缺少 output 对象。")
    items = output.get("results")
    if not isinstance(items, list) or len(items) != top_n:
        raise ValueError("返回结果数量与请求的 top_n 不一致。")

    results = []
    seen_indices = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("重排结果条目格式不正确。")
        index = item.get("index")
        score: SupportsFloat | SupportsIndex = item.get("relevance_score")  # pyright: ignore[reportAssignmentType]
        # index 是输入列表中的原始位置，不是重排之后的名次。
        if type(index) is not int or not 0 <= index < len(documents):
            raise ValueError("文档索引必须是输入列表范围内的整数。")
        if index in seen_indices:
            raise ValueError("重排结果含有重复文档索引。")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("相关性分数必须是有限数值。")
        seen_indices.add(index)
        results.append((index, float(score)))

    # 使用分数排序，同时保留原始索引，以便找回原始文档及来源 ID。
    return sorted(results, key=lambda result: result[1], reverse=True)


def main() -> int:
    """检查配置、调用 Reranker，并展示候选片段的重排结果。"""
    parser = argparse.ArgumentParser(description="验证 Reranker 接口和候选片段重排序")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="要检索的问题")
    parser.add_argument(
        "--top-n",
        type=int,
        default=len(DOCUMENTS),
        help="保留得分最高的前 N 条，默认保留全部三条",
    )
    parser.add_argument(
        "--check-config", action="store_true", help="只检查本地配置，不发送请求"
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query 不能为空或只有空白字符")
    if not 1 <= args.top_n <= len(DOCUMENTS):
        parser.error(f"--top-n 必须是 1 到 {len(DOCUMENTS)} 之间的整数")

    if not ENV_PATH.is_file():
        print(f"[错误] 配置文件不存在：{ENV_PATH}", file=sys.stderr)
        return 1
    load_dotenv(dotenv_path=ENV_PATH, override=False, encoding="utf-8")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        print("[错误] 请配置 DASHSCOPE_API_KEY。", file=sys.stderr)
        return 1

    print(f"[配置] Python：{sys.executable}")
    print(f"[配置] 模型：{MODEL_NAME}")
    print(f"[配置] 服务地址：{API_URL}")
    print(f"[配置] 保留结果数：{args.top_n}")
    print("[配置] DASHSCOPE_API_KEY 已配置（不显示密钥）", flush=True)
    if args.check_config:
        print("[成功] 本地配置检查通过，未发送模型请求。")
        return 0

    print("[演示] 候选文本是测试知识库的手工摘录，不是真实设备维护规程。")
    print(f"[问题] {args.query}")
    print("[候选] 原始输入顺序（索引从 0 开始）：")
    for index, document in enumerate(DOCUMENTS):
        print(f"  index={index} -> {document['id']}")

    started = perf_counter()
    try:
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        ) as client:
            print("[请求] 将问题和候选文本一起发送给 Reranker。", flush=True)
            results = rerank_documents(
                client,
                args.query,
                [document["text"] for document in DOCUMENTS],
                args.top_n,
            )
    except httpx.TimeoutException:
        print("[错误] 请求超时，请检查服务状态或网络。", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"[错误] 服务返回 HTTP {exc.response.status_code}。", file=sys.stderr)
        return 1
    except httpx.RequestError:
        print("[错误] 连接失败，请检查服务地址和网络。", file=sys.stderr)
        return 1
    except (ValueError, TypeError, OverflowError) as exc:
        print(f"[错误] 响应解析或结果校验失败：{exc}", file=sys.stderr)
        return 1

    print(f"\n[重排] 按相关性分数从高到低展示前 {args.top_n} 条：")
    for rank, (index, score) in enumerate(results, start=1):
        # 用原始 index 找回条目，而不是用 rank 找条目。
        document = DOCUMENTS[index]
        print(f"{rank}. {document['id']} | 相关性分数={score:.4f} | 原始 index={index}")
        print(f"   {document['text']}")
    print("\n[说明] 重排分数不是故障概率，也不能与 Embedding 余弦分数直接比较。")
    print(
        f"[成功] Reranker 调用与结果校验完成，耗时 {perf_counter() - started:.2f} 秒。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
