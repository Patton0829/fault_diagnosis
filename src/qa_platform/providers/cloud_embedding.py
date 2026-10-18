"""第二步：调用 Embedding，把问题和演示文本转换为向量并比较相似度。"""

import argparse
import os
import sys
from time import perf_counter

import httpx
import numpy as np
from dotenv import load_dotenv

from qa_platform.settings import ENV_PATH, PROJECT_ROOT as PROJECT_ROOT


API_BASE_URL = "https://maas.qianwenaiapi.com/api/v1"
API_URL = f"{API_BASE_URL}/services/embeddings/text-embedding/text-embedding"
MODEL_NAME = "qwen3.7-text-embedding-flash"
DEFAULT_QUERY = "只有A轴振动出现连续尖峰，另外两轴和三相电流都正常，优先检查什么？"

# 手工摘录自测试知识库，仅作为本次演示样本，尚未解析或入库知识库文件。
# 故意不按相关性排列，让后面的相似度计算决定结果顺序。
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


def embed_texts(client: httpx.Client, texts: list[str], text_type: str) -> np.ndarray:
    """把一批文本转成二维数组：每行对应一段文本的向量。"""
    response = client.post(
        API_URL,
        json={
            "model": MODEL_NAME,
            "input": {"texts": texts},
            "parameters": {"text_type": text_type},
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("code"):
        raise ValueError("服务未返回成功的向量响应。")
    output = payload.get("output")
    if not isinstance(output, dict):
        raise ValueError("响应缺少 output 对象。")
    items = output.get("embeddings")
    if not isinstance(items, list) or len(items) != len(texts):
        raise ValueError("返回的向量数量与输入文本数量不一致。")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("向量条目的格式不正确。")

    # 文档使用 text_index；当前服务实测使用 index，两者都表示输入位置。
    # 如果两种字段同时出现却互相矛盾，就拒绝使用这些向量。
    if any(
        "text_index" in item and "index" in item and item["text_index"] != item["index"]
        for item in items
    ):
        raise ValueError("text_index 与 index 不一致，无法确定文本对应关系。")
    indices = [item.get("text_index", item.get("index")) for item in items]
    if any(type(index) is not int for index in indices):
        raise ValueError("文本索引 text_index/index 必须是整数。")
    if sorted(indices) != list(range(len(texts))):
        raise ValueError("文本索引存在重复、缺失或越界。")
    ordered = sorted(items, key=lambda item: item.get("text_index", item.get("index")))
    vectors = [item.get("embedding") for item in ordered]
    if any(not isinstance(vector, list) or not vector for vector in vectors):
        raise ValueError("返回的向量必须是非空列表。")

    matrix = np.asarray(vectors)
    if matrix.ndim != 2 or matrix.dtype.kind not in "fiu":
        raise ValueError("向量必须是一维、等维的数值列表。")
    matrix = matrix.astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("向量含有 NaN 或无穷大，不能计算相似度。")
    return matrix


def cosine_scores(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """余弦相似度 = 点积 / 两个向量长度的乘积。"""
    if query.shape[0] != documents.shape[1]:
        raise ValueError("问题向量和文档向量的维度不一致。")
    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)
    if not np.isfinite(query_norm) or not np.isfinite(document_norms).all():
        raise ValueError("向量长度不是有限值。")
    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("零向量不能计算余弦相似度。")
    # @ 是矩阵乘法；这里会一次计算各篇文档与问题的点积。
    return (documents @ query) / (document_norms * query_norm)


def main() -> int:
    """检查配置、分别编码问题和文档，再展示语义相似度排序。"""
    parser = argparse.ArgumentParser(description="验证 Embedding 接口和语义相似度")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="要检索的问题")
    parser.add_argument(
        "--check-config", action="store_true", help="只检查本地配置，不发送请求"
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query 不能为空或只有空白字符")

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
    print("[配置] DASHSCOPE_API_KEY 已配置（不显示密钥）", flush=True)
    if args.check_config:
        print("[成功] 本地配置检查通过，未发送模型请求。")
        return 0

    print("[演示] 候选文本是测试知识库的手工摘录，不是真实设备维护规程。")
    print(f"[问题] {args.query}")
    started = perf_counter()
    try:
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        ) as client:
            print("[请求] 编码问题：text_type=query", flush=True)
            query_vector = embed_texts(client, [args.query], "query")[0]
            print("[请求] 批量编码三条候选文本：text_type=document", flush=True)
            document_vectors = embed_texts(
                client, [document["text"] for document in DOCUMENTS], "document"
            )
        scores = cosine_scores(query_vector, document_vectors)
    except httpx.TimeoutException:
        print("[错误] 请求超时，请检查服务状态或网络。", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(f"[错误] 服务返回 HTTP {exc.response.status_code}。", file=sys.stderr)
        return 1
    except httpx.RequestError:
        print("[错误] 连接失败，请检查服务地址和网络。", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as exc:
        print(f"[错误] 响应解析或向量校验失败：{exc}", file=sys.stderr)
        return 1

    print(f"\n[向量] 问题形状：{query_vector.shape}")
    print(f"[向量] 候选文本矩阵形状：{document_vectors.shape}")
    print(f"[向量] 问题向量前 5 个数：{query_vector[:5].round(6).tolist()}")
    print("\n[排序] 按余弦相似度从高到低排列：")
    # argsort 默认从小到大；[::-1] 把顺序反过来。
    for rank, index in enumerate(np.argsort(scores)[::-1], start=1):
        document = DOCUMENTS[index]
        print(f"{rank}. {document['id']} | 相似度={scores[index]:.4f}")
        print(f"   {document['text']}")
    print("\n[说明] 分数表示文本的语义相似程度，不是故障发生概率。")
    print(
        f"[成功] Embedding 调用与相似度计算完成，耗时 {perf_counter() - started:.2f} 秒。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
