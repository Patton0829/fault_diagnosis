"""本地Embedding小规模验证：默认离线计划，--run才请求已配置的Ollama。

只抽样8块（含最长正文），比较批量/单条编码，再编码一个带指令的问题。
不读取.env、不回退云端、不调用Reranker/GLM、不写语料、报告或Milvus。
"""

import argparse
from ipaddress import ip_address, ip_network
from pathlib import Path
import re
import sys
from time import perf_counter
from urllib.parse import urlsplit

from qa_platform.ingestion.embedding_plan import (
    CORPUS,
    load_plan,
    request_body,
    require,
)


QUERY = "如何排查Pod的DNS解析失败？"
SAMPLE_SIZE = 8
ABS_TOLERANCE = 1e-4
REL_TOLERANCE = 1e-3


def local_endpoint_base(profile: dict) -> str:
    """共享地址边界：只允许指南中的内网/回环IP，不允许云端回退。"""
    endpoint = urlsplit(profile["endpoint"])
    address = ip_address(endpoint.hostname)
    require(
        address.is_loopback
        or any(
            address in ip_network(network)
            for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
        ),
        "本脚本只允许指南中的内网/回环IP，不允许云端域名或公网IP",
    )
    return f"{endpoint.scheme}://{endpoint.netloc}"


def prepare_check(corpus: Path) -> dict:
    """复用冻结语料校验；选最长正文和其余前7块，不修改任何输入。"""
    plan = load_plan(corpus, SAMPLE_SIZE, "body-only")
    profile = plan["profile"]
    base_url = local_endpoint_base(profile)
    rows = plan["inputs"]["body-only"]
    require(len(rows) >= SAMPLE_SIZE, "本验证需要至少8个候选块")
    longest = max(rows, key=lambda row: row["input_chars"])
    others = [row for row in rows if row["chunk_id"] != longest["chunk_id"]]
    selected_ids = {row["chunk_id"] for row in others[: SAMPLE_SIZE - 1]}
    selected_ids.add(longest["chunk_id"])
    sample = [row for row in rows if row["chunk_id"] in selected_ids]
    return {
        "plan": plan,
        "base_url": base_url,
        "sample": sample,
        "longest": longest,
        "anchor_index": next(
            index
            for index, row in enumerate(sample)
            if row["chunk_id"] == longest["chunk_id"]
        ),
        "query_text": profile["query_format"].format(
            instruction=profile["query_instruction"], query=QUERY
        ),
    }


def show_check(check: dict):
    plan = check["plan"]
    profile = plan["profile"]
    print(f"[配置] Python：{sys.executable}")
    print(
        f"[模型] {profile['model']}；{profile['endpoint']}；指南维度={profile['documented_dimension']}"
    )
    print(
        f"[范围] 已准备子集{len(plan['inputs']['body-only'])}块，仅抽样{SAMPLE_SIZE}块；文档只编码原正文。"
    )
    for index, row in enumerate(check["sample"]):
        mark = (
            "；全子集最长正文，另做单条对照" if index == check["anchor_index"] else ""
        )
        print(
            f"  index={index} | {row['chunk_id']} | {row['input_chars']}字符 | {row['title']}{mark}"
        )
    print(
        "[模型请求计划] 1次批量8块 + 1次单独编码最长块 + 1次问题编码 = 3次；共处理10条输入，其中最长块重复一次。"
    )
    print(
        "[元数据计划] /api/tags前后各1次，/api/ps最后1次，共3次；正常完成总计6次HTTP请求，无自动重试。"
    )
    print(
        f"[对照] 同一最长块，抽查批量位置与单条输出：allclose(atol={ABS_TOLERANCE}, rtol={REL_TOLERANCE})。"
    )
    print(
        f"[边界] 只验证抽样可运行、维度/数值及一条对应关系；不证明全部{len(plan['inputs']['body-only'])}块可用、批量上限、检索质量或所有输入顺序。"
    )
    print(
        "[Token] 最长按字符选，不代表Token最多；字符不换算成Token；只展示服务返回的处理量，批次总量不冒充单块Token数。"
    )
    print(
        "[目标] 仅使用指南中的内网Ollama，禁用环境代理/重定向；不读.env、不回退云端、不请求Reranker/GLM、不写数据库。",
        flush=True,
    )


def model_entry(payload: dict, model: str, required: bool = True) -> dict | None:
    require(
        isinstance(payload, dict) and isinstance(payload.get("models"), list),
        "模型列表响应结构不正确",
    )
    require(
        all(isinstance(item, dict) for item in payload["models"]), "模型列表包含非对象"
    )
    matches = [
        item
        for item in payload["models"]
        if model in (item.get("name"), item.get("model"))
    ]
    require(len(matches) <= 1, "模型别名匹配不唯一")
    require(
        not required or bool(matches),
        "服务列表中没有指南指定的模型；不会自动下载或换模型",
    )
    return matches[0] if matches else None


def model_digest(entry: dict) -> str:
    value = entry.get("digest")
    require(isinstance(value, str), "服务未提供模型digest")
    value = value.removeprefix("sha256:")
    require(
        re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None, "模型digest不是SHA256格式"
    )
    return value.lower()


def validate_embeddings(payload: dict, expected_count: int, profile: dict):
    """拒绝数量/维度错误、字符串、bool、NaN、Inf、零向量和float32溢出。"""
    import numpy as np

    require(
        isinstance(payload, dict) and not payload.get("error"),
        "Embedding返回服务错误或非对象",
    )
    require(payload.get("model") == profile["model"], "返回模型名与指定模型不一致")
    vectors = payload.get("embeddings")
    require(
        isinstance(vectors, list) and len(vectors) == expected_count,
        "返回向量条数与输入不一致",
    )
    dimension = profile["documented_dimension"]
    require(
        all(
            isinstance(vector, list) and len(vector) == dimension for vector in vectors
        ),
        "实际向量维度与指南不一致",
    )
    require(
        all(type(number) in (int, float) for vector in vectors for number in vector),
        "向量分量必须为数字，不能是bool或字符串",
    )
    matrix = np.asarray(vectors, dtype=np.float64)
    require(np.isfinite(matrix).all(), "向量含NaN/Inf")
    require(
        (np.abs(matrix) <= np.finfo(np.float32).max).all(), "向量不能安全存为float32"
    )
    stored = matrix.astype(np.float32)
    norms = np.linalg.norm(stored.astype(np.float64), axis=1)
    require(np.isfinite(norms).all() and (norms > 0).all(), "向量范数非有限或为零")
    return matrix, norms


def compare_anchor(batch, single, anchor_index: int) -> dict:
    import numpy as np

    left, right = batch[anchor_index], single[0]
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return {
        "cosine": cosine,
        "max_abs_difference": float(np.max(np.abs(left - right))),
        "within_tolerance": bool(
            np.allclose(left, right, atol=ABS_TOLERANCE, rtol=REL_TOLERANCE)
        ),
    }


def http_json(session, method: str, url: str, *, payload=None, timeout=180) -> dict:
    """只允许调用方指定的地址；不跟随重定向，不输出原始服务错误正文。"""
    response = session.request(
        method, url, json=payload, timeout=(5, timeout), allow_redirects=False
    )
    try:
        require(
            response.status_code == 200,
            f"HTTP {response.status_code}；停止，不自动重试或切换接口",
        )
        data = response.json()
        require(
            isinstance(data, dict) and not data.get("error"), "服务返回错误或非JSON对象"
        )
        return data
    finally:
        response.close()


def run_check(check: dict, timeout: int, session) -> dict:
    """3次推理，3次只读元数据请求；注入session便于零网络测试。"""
    profile = check["plan"]["profile"]
    base, model = check["base_url"], profile["model"]
    print("\n[请求] 元数据1/3：读取模型列表和digest；不会拉取/创建模型。", flush=True)
    before = model_digest(
        model_entry(http_json(session, "GET", base + "/api/tags", timeout=15), model)
    )
    print(f"[身份] 调用前模型digest={before}")
    query_row = {"text": check["query_text"]}
    experiments = (
        ("文档批量8块（含最长块）", check["sample"]),
        ("同一最长块单条编码", [check["longest"]]),
        ("问题编码（含Instruct/Query指令）", [query_row]),
    )
    matrices, records = [], []
    for index, (name, rows) in enumerate(experiments, 1):
        print(
            f"[请求] 模型{index}/3：{name}；{len(rows)}条输入；truncate=false。",
            flush=True,
        )
        started = perf_counter()
        data = http_json(
            session,
            "POST",
            profile["endpoint"],
            payload=request_body(profile, rows),
            timeout=timeout,
        )
        wall_seconds = perf_counter() - started
        matrix, norms = validate_embeddings(data, len(rows), profile)
        matrices.append(matrix)
        tokens = data.get("prompt_eval_count")
        require(
            tokens is None or (type(tokens) is int and tokens >= 0),
            "服务Token统计不合法",
        )
        record = {
            "name": name,
            "shape": matrix.shape,
            "wall_seconds": wall_seconds,
            "processed_tokens": tokens,
        }
        records.append(record)
        print(
            f"[返回] 形状={matrix.shape}；float32可保存；L2范数范围={norms.min():.6f}～{norms.max():.6f}；客户端耗时={wall_seconds:.3f}s"
        )
        print(
            f"[服务统计] 本次处理Token={tokens if tokens is not None else '未返回'}；不是云端账单或逐文本Tokenizer核验。",
            flush=True,
        )
    print("[请求] 元数据2/3：再次读取模型列表；核对别名digest没有变化。", flush=True)
    after = model_digest(
        model_entry(http_json(session, "GET", base + "/api/tags", timeout=15), model)
    )
    require(before == after, "调用前后模型digest不同，不能把本轮视为同一模型验证")
    print(
        "[身份边界] 前后digest相同只表示两个观测时刻一致，不能证明中途每次推理或未来建库均使用相同权重；全量建库仍须重新记录身份。"
    )
    print("[请求] 元数据3/3：读取当前已加载模型的上下文信息。", flush=True)
    running = model_entry(
        http_json(session, "GET", base + "/api/ps", timeout=15), model, required=False
    )
    context = None
    if running:
        require(model_digest(running) == before, "已加载模型digest与列表不一致")
        context = running.get("context_length")
        require(
            context is None or (type(context) is int and context > 0),
            "运行中模型的上下文统计不合法",
        )
    comparison = compare_anchor(matrices[0], matrices[1], check["anchor_index"])
    print(
        f"\n[批量/单条对照] 同一块{check['longest']['chunk_id']}；原批量index={check['anchor_index']}"
    )
    print(
        f"[数值] 余弦={comparison['cosine']:.9f}；最大分量差={comparison['max_abs_difference']:.9g}；在显式容差内={'是' if comparison['within_tolerance'] else '否'}"
    )
    print(
        "[解释] 比较的是同一文本两次编码的数值一致性，不是问题与文档的相关性，也不是检索准确率。"
    )
    require(
        comparison["within_tolerance"],
        "抽查向量不在容差内；需核实批处理/推理差异，不能直接归因于排序错误",
    )
    print(
        f"[运行中上下文] 服务此刻报告={context if context is not None else '未知/未返回'}；不是所有请求或其他模型的永久上限。"
    )
    print(
        "[计时边界] 首次调用可能含模型加载；8条与1条处理量不同，不能直接凭耗时比较批量性能。"
    )
    print(
        "[成功] 本地Embedding小样本验证通过；3次推理、3次元数据HTTP请求；无重试、无云端回退。"
    )
    print(
        "[尚未完成] 未检查剩余文本的Token/可编码性，未全量编码、未保存向量、未连接Milvus；原语料ready标记不变。"
    )
    return {
        "digest": before,
        "running_context_tokens": context,
        "records": records,
        "comparison": comparison,
        "http_requests": 6,
        "model_requests": 3,
    }


def self_test():
    """合成响应和假Session覆盖完整请求流；不读真实语料、不联网。"""
    import numpy as np

    profile = {
        "model": "demo",
        "documented_dimension": 2,
        "endpoint": "http://127.0.0.1/api/embed",
        "truncate": False,
        "keep_alive": "10m",
    }
    valid = {"model": "demo", "embeddings": [[1.0, 0.0]]}
    matrix, norms = validate_embeddings(valid, 1, profile)
    assert matrix.shape == (1, 2) and norms[0] == 1
    checks = 1
    for bad in (
        {**valid, "model": "wrong"},
        {**valid, "embeddings": []},
        {**valid, "embeddings": [[1.0]]},
        {**valid, "embeddings": [[True, 0]]},
        {**valid, "embeddings": [["1", 0]]},
        {**valid, "embeddings": [[float("nan"), 0]]},
        {**valid, "embeddings": [[float("inf"), 0]]},
        {**valid, "embeddings": [[0, 0]]},
        {**valid, "embeddings": [[1e50, 0]]},
        {**valid, "embeddings": [[1e-60, 0]]},
    ):
        try:
            validate_embeddings(bad, 1, profile)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("不合法向量应拒绝")
    assert compare_anchor(matrix, matrix, 0)["within_tolerance"]
    assert not compare_anchor(matrix, np.array([[0.0, 1.0]]), 0)["within_tolerance"]
    checks += 1
    digest = "a" * 64
    entry = {"name": "demo", "digest": digest, "context_length": 8192}
    assert model_digest(model_entry({"models": [entry]}, "demo")) == digest
    checks += 1

    class Response:
        status_code = 200

        def __init__(self, data):
            self.data = data

        def json(self):
            return self.data

        def close(self):
            pass

    class Session:
        def __init__(self, status=200, drift=False):
            self.calls = []
            self.status = status
            self.drift = drift

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            assert kwargs["allow_redirects"] is False
            if method == "GET":
                item = dict(entry)
                if self.drift and len(self.calls) > 1:
                    item["digest"] = "b" * 64
                response = Response({"models": [item]})
            else:
                body = kwargs["json"]
                assert body["truncate"] is False and body["model"] == "demo"
                response = Response(
                    {
                        "model": "demo",
                        "embeddings": [[1.0, 0.0] for _ in body["input"]],
                        "prompt_eval_count": 12,
                    }
                )
            response.status_code = self.status
            return response

    import contextlib
    import io

    rows = [{"text": f"正文{i}", "chunk_id": str(i)} for i in range(SAMPLE_SIZE)]
    check = {
        "plan": {"profile": profile},
        "base_url": "http://127.0.0.1",
        "sample": rows,
        "longest": rows[-1],
        "anchor_index": 7,
        "query_text": "Instruct: test\nQuery: 问题",
    }
    session = Session()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_check(check, 15, session)
    assert len(session.calls) == 6 and result["model_requests"] == 3
    assert [
        len(call[2]["json"]["input"]) for call in session.calls if call[0] == "POST"
    ] == [8, 1, 1]
    checks += 1
    for session, expected in ((Session(status=302), 1), (Session(drift=True), 5)):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_check(check, 15, session)
        except ValueError:
            assert len(session.calls) == expected
            checks += 1
        else:
            raise AssertionError("重定向或模型漂移必须停止")
    print(
        f"[自测通过] {checks}组：异常向量、float32、单条/批量、6次假HTTP请求、重定向停止、digest漂移停止；零真实请求。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="每次推理的读取超时秒数；连接超时固定5秒",
    )
    parser.add_argument(
        "--run", action="store_true", help="执行3次本地推理及3次元数据请求"
    )
    parser.add_argument(
        "--self-test", action="store_true", help="只用内存假响应，不联网"
    )
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout须为1～600秒")
    if args.self_test and args.run:
        parser.error("self-test不能与run同时使用")
    if args.self_test:
        self_test()
        return 0
    check = prepare_check(args.corpus.resolve())
    show_check(check)
    if not args.run:
        print(
            "[计划完成] 实际HTTP请求=0；未导入requests、创建客户端、读取密钥或写入任何产物；加--run才执行。"
        )
        return 0
    import requests

    try:
        with requests.Session() as session:
            session.trust_env = False
            run_check(check, args.timeout, session)
    except requests.Timeout:
        print(
            "[失败] 本地服务连接或读取超时；不重试、不回退云端。服务端可能仍在处理已发出的请求。",
            file=sys.stderr,
        )
        return 1
    except requests.RequestException:
        print(
            "[失败] 无法连接本地服务或解析其响应；请核对校园网与服务状态。不回退云端。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print(
            f"[校验失败] {error}；不会继续请求、回退云端或写入Milvus。", file=sys.stderr
        )
        raise SystemExit(2)
