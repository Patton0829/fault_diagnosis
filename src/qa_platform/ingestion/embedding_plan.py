"""Kubernetes Embedding 入库前预检：只读、离线、不读取.env、不创建模型客户端。

校验v2语料；按当前本地接口文档提出计划；比较正文/章节前缀两种输入。
未知Token数、部署上下文、模型digest和批量容量始终标为待核验，不能以字符数冒充。
本文件没有--run：不调用模型，不写报告，不建目录或Milvus索引。
"""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from qa_platform.ingestion.kubernetes_corpus import verify_directory
from qa_platform.settings import PROJECT_ROOT, PACKAGE_ROOT, MODEL_DOCS_DIR


ROOT = PROJECT_ROOT
CORPUS = ROOT / "data/k8s_corpora/troubleshooting_v2"
API_DOCUMENT = MODEL_DOCS_DIR / "本地Embedding模型调用.md"
LEGACY_SCRIPT = PACKAGE_ROOT / "providers/cloud_embedding.py"
MODES = ("body-only", "section-prefix")
POLICY_VERSION = "k8s-embedding-input-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def literal_constants(code: str, names: set[str]) -> dict:
    """只解析指定常量；不导入、不exec文档或旧脚本中的任何代码。"""
    values = {}
    for statement in ast.parse(code).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    require(target.id not in values, f"常量重复：{target.id}")
                    values[target.id] = ast.literal_eval(statement.value)
    require(set(values) == names, "文档缺少预期的显式配置常量")
    return values


def profile_from_document(text: str) -> dict:
    """读取当前指南的显式Python常量与请求选项，不从自然语言猜限制。"""
    blocks = re.findall(
        r"^~~~python[ \t]*\n(.*?)^~~~[ \t]*$", text, re.MULTILINE | re.DOTALL
    )
    names = {"EMBEDDING_URL", "EMBEDDING_MODEL", "QUERY_INSTRUCTION"}
    profiles = []
    for code in blocks:
        tree = ast.parse(code)
        assignments = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        if not names <= assignments:
            continue
        constants = literal_constants(code, names)
        options = {"truncate": [], "keep_alive": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and key.value in options
                    ):
                        options[key.value].append(ast.literal_eval(value))
        require(
            len(options["truncate"]) == 1 and options["truncate"][0] is False,
            "指南中的请求必须明确且唯一地设置truncate=False",
        )
        require(
            len(options["keep_alive"]) == 1
            and isinstance(options["keep_alive"][0], str),
            "指南缺少明确keep_alive",
        )
        profiles.append(
            {**constants, "truncate": False, "keep_alive": options["keep_alive"][0]}
        )
    require(
        len(profiles) == 1,
        "需要唯一的本地Embedding配置Python示例；指南结构已变更时须先核对",
    )
    values = profiles[0]
    require(
        all(isinstance(values[name], str) and values[name].strip() for name in names),
        "模型/地址/查询指令必须为非空字符串",
    )
    endpoint = urlsplit(values["EMBEDDING_URL"])
    require(
        endpoint.scheme in ("http", "https")
        and endpoint.hostname
        and endpoint.path == "/api/embed",
        "本步仅规划Ollama原生/api/embed接口",
    )
    require(
        not endpoint.username
        and not endpoint.password
        and not endpoint.query
        and not endpoint.fragment,
        "服务地址不能夹带凭据或查询参数",
    )
    require(
        "\n" not in values["QUERY_INSTRUCTION"]
        and "\r" not in values["QUERY_INSTRUCTION"],
        "查询任务描述须为单行",
    )
    dimension = re.findall(
        r"^\|\s*实测向量维度\s*\|\s*`?([0-9]+)`?\s*\|", text, re.MULTILINE
    )
    require(len(dimension) == 1 and int(dimension[0]) > 0, "指南缺少唯一的实测维度记录")
    return {
        "provider": "ollama",
        "model": values["EMBEDDING_MODEL"],
        "endpoint": values["EMBEDDING_URL"],
        "documented_dimension": int(dimension[0]),
        "actual_dimension": None,
        "truncate": False,
        "keep_alive": values["keep_alive"],
        "query_instruction": values["QUERY_INSTRUCTION"],
        "query_format": "Instruct: {instruction}\nQuery: {query}",
        "document_query_prefix": False,
        "server_model_digest": None,
        "server_context_tokens": None,
        "server_batch_limit": None,
        "tokenizer_match_verified": False,
        "status": "proposed_from_local_document_not_connected",
    }


def make_inputs(chunks: list[dict], mode: str, contract_hash: str) -> list[dict]:
    require(mode in MODES and bool(chunks), "编码模式不合法或没有候选块")
    rows = []
    for chunk in chunks:
        metadata, body = chunk["metadata"], chunk["page_content"]
        require(
            chunk["id"] == metadata["chunk_id"]
            and isinstance(body, str)
            and bool(body.strip()),
            "候选身份或正文不合法",
        )
        sections, title = metadata["sections"], metadata["title"]
        require(isinstance(title, str) and bool(title.strip()), "缺少文档标题")
        require(
            isinstance(sections, list)
            and len(sections) == 1
            and isinstance(sections[0], list),
            "本策略只接受单章节候选，不能将多章节混成一个版本前提",
        )
        require(
            all(
                isinstance(heading, str) and heading.strip() for heading in sections[0]
            ),
            "章节路径不合法",
        )
        path = list(sections[0])
        if not path or path[0] != title:
            path.insert(0, title)
        prefix = (
            "文档与章节：" + " / ".join(path) + "\n\n"
            if mode == "section-prefix"
            else ""
        )
        text = (
            prefix + body
        )  # 不strip、不截断、不重切正文；不加入整篇所有feature_states。
        input_hash = fingerprint(text)
        rows.append(
            {
                "chunk_id": chunk["id"],
                "parent_id": metadata["parent_id"],
                "title": title,
                "sections": path,
                "mode": mode,
                "prefix": prefix,
                "text": text,
                "body_chars": len(body),
                "input_chars": len(text),
                "input_utf8_bytes": len(text.encode("utf-8")),
                "body_sha256": fingerprint(body),
                "input_sha256": input_hash,
                "embedding_input_id": fingerprint(
                    canonical(
                        [chunk["id"], POLICY_VERSION, mode, contract_hash, input_hash]
                    )
                ),
                "source_char_start": metadata["cleaned_char_start"],
                "source_char_end": metadata["cleaned_char_end"],
                "token_count": None,
                "token_limit_checked": False,
            }
        )
    require(len({row["chunk_id"] for row in rows}) == len(rows), "候选身份重复")
    return rows


def request_body(profile: dict, rows: list[dict]) -> dict:
    """本地构造请求草案，不发送；Ollama不是旧云端input.texts/parameters格式。"""
    return {
        "model": profile["model"],
        "input": [row["text"] for row in rows],
        "truncate": profile["truncate"],
        "keep_alive": profile["keep_alive"],
    }


def make_batches(profile: dict, rows: list[dict], size: int) -> list[dict]:
    require(
        type(size) is int and 1 <= size <= 1000,
        "客户端规划batch-size须为1～1000，不代表服务端上限",
    )
    batches = []
    for start in range(0, len(rows), size):
        group = rows[start : start + size]
        body = canonical(request_body(profile, group))
        batches.append(
            {
                "start": start,
                "end": start + len(group),
                "items": len(group),
                "input_chars": sum(row["input_chars"] for row in group),
                "json_utf8_bytes": len(body.encode("utf-8")),
                "body_sha256": fingerprint(body),
            }
        )
    return batches


def load_plan(corpus: Path, batch_size: int, selected_mode: str) -> dict:
    require(selected_mode in MODES, "编码模式不合法")
    manifest = verify_directory(corpus)
    manifest_bytes = (corpus / "manifest.json").read_bytes()
    require(json.loads(manifest_bytes) == manifest, "校验期间清单发生变化")
    chunk_bytes = (corpus / "chunks.jsonl").read_bytes()
    require(
        sha256(chunk_bytes).hexdigest() == manifest["files"]["chunks.jsonl"]["sha256"],
        "读取期间候选文件发生变化",
    )
    chunks = [
        json.loads(line)
        for line in chunk_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    require(
        bool(chunks)
        and all(chunk["metadata"]["kb_id"] == "kubernetes_zh" for chunk in chunks),
        "本步仅规划Kubernetes语料，不能混入铁路测试库",
    )
    doc_bytes = API_DOCUMENT.read_bytes()
    profile = profile_from_document(doc_bytes.decode("utf-8"))
    profile["local_document_sha256"] = sha256(doc_bytes).hexdigest()
    contract_hash = fingerprint(canonical(profile))
    legacy_model = literal_constants(
        LEGACY_SCRIPT.read_text(encoding="utf-8"), {"MODEL_NAME"}
    )["MODEL_NAME"]
    inputs = {mode: make_inputs(chunks, mode, contract_hash) for mode in MODES}
    selected = inputs[selected_mode]
    batches = make_batches(profile, selected, batch_size)
    input_snapshot_hash = fingerprint(
        canonical(
            [
                [row["chunk_id"], row["embedding_input_id"], row["input_sha256"]]
                for row in selected
            ]
        )
    )
    return {
        "manifest": manifest,
        "corpus_sha256": sha256(manifest_bytes).hexdigest(),
        "legacy_model": legacy_model,
        "profile": profile,
        "contract_sha256": contract_hash,
        "inputs": inputs,
        "selected_mode": selected_mode,
        "batch_size": batch_size,
        "batches": batches,
        "input_snapshot_sha256": input_snapshot_hash,
        "planned_index": ROOT
        / "data/milvus_indexes"
        / f"kubernetes_zh_ollama_{contract_hash[:12]}_{selected_mode.replace('-', '_')}_{input_snapshot_hash[:12]}",
        "ready_for_embedding": False,
        "model_api_calls": 0,
    }


def show_plan(plan: dict, show_chunk: str | None, preview_chars: int):
    profile, manifest = plan["profile"], plan["manifest"]
    modes, selected = plan["inputs"], plan["selected_mode"]
    candidates = modes[selected]
    example_id = show_chunk or next(
        (
            row["chunk_id"]
            for row in modes["body-only"]
            if row["body_chars"] < 500 and not row["text"].lstrip().startswith("#")
        ),
        candidates[0]["chunk_id"],
    )
    require(
        any(row["chunk_id"] == example_id for row in candidates),
        "show-chunk没有匹配的候选ID",
    )
    print(
        f"[配置] Python：{sys.executable}；语料：{manifest['prepared_documents']}篇/{manifest['candidate_chunks']}候选块；原字符目标={manifest['target_chars']}"
    )
    print(
        f"[语料] {manifest['status']}；只使用已导出子集；清单SHA256={plan['corpus_sha256'][:16]}…"
    )
    print(f"[旧链路] {plan['legacy_model']}；保持不动，不复用其请求适配器或向量索引。")
    print(f"[拟用配置] 来源：{API_DOCUMENT}；{profile['model']}；{profile['endpoint']}")
    print(
        f"[维度] 指南记录={profile['documented_dimension']}；本脚本实测=未知；4096这类向量维度不是输入Token上限。"
    )
    print(
        "[接口] Ollama input=[文本,…]；truncate=false；不向文档添加Instruct/Query查询前缀。"
    )
    print("\n[对照] 同一批候选，只改变实际送入模型的文字；本轮两种都不编码：")
    print(
        "模式           | 文本条数 | 正文总字符 | 编码总字符 | 编码UTF-8字节 | 最长输入 | 超原字符目标"
    )
    for mode in MODES:
        rows = modes[mode]
        print(
            f"{mode:14} | {len(rows):8} | {sum(row['body_chars'] for row in rows):10} | {sum(row['input_chars'] for row in rows):10} | {sum(row['input_utf8_bytes'] for row in rows):13} | {max(row['input_chars'] for row in rows):8} | {sum(row['input_chars'] > manifest['target_chars'] for row in rows):12}"
        )
    print(
        "[含义] body-only严格等于page_content；section-prefix只附文档/本块章节路径，不加入本地文件路径、ID或整篇所有功能状态。"
    )
    print(
        f"[边界] 超过原{manifest['target_chars']}字符目标不是超过模型Token上限；追加标题会改变输入指纹和长度，但不会改原块身份或坐标，也不保证检索更准。"
    )
    print(f"\n[同块输入对照] chunk_id={example_id}；不是重新切块：")
    for mode in MODES:
        row = next(row for row in modes[mode] if row["chunk_id"] == example_id)
        print(
            f"[{mode}] {row['input_chars']}字符/{row['input_utf8_bytes']}字节；SHA256={row['input_sha256'][:16]}…"
        )
        print(row["text"][:preview_chars])
        if len(row["text"]) > preview_chars:
            print(f"……仅预览前{preview_chars}字符，计划中的输入未被截断。")
    print(
        "\n[原正文超字符目标的候选] 保持代码完整，不静默截断或重切；全部仍需Token核验："
    )
    for row in sorted(
        modes["body-only"], key=lambda item: item["body_chars"], reverse=True
    ):
        if row["body_chars"] > manifest["target_chars"]:
            print(f"  {row['chunk_id']} | {row['body_chars']}字符 | {row['title']}")
    print(
        f"\n[单路线条件计划] 选择={selected}；客户端每批最多{plan['batch_size']}条；共{len(plan['batches'])}次文档Embedding。"
    )
    print("批次 | 原候选切片[起,止) | 条数 | 输入字符 | JSON请求体字节")
    for index, batch in enumerate(plan["batches"], 1):
        print(
            f"{index:4} | [{batch['start']:3},{batch['end']:3})          | {batch['items']:4} | {batch['input_chars']:8} | {batch['json_utf8_bytes']:14}"
        )
    print(
        "[口径] 批量大小只是客户端设想，不是服务器上限；字节数按本脚本JSON序列化计算，不是Token数或实际网络包大小。"
    )
    print(
        f"[请求次数] 单路线完整成功且无重试时={len(plan['batches'])}次；若两种输入都真正编码约={2 * len(plan['batches'])}次。本轮实际=0，问题/Reranker/聊天模型均0。"
    )
    query = profile["query_format"].format(
        instruction=profile["query_instruction"], query="如何排查Pod的DNS解析失败？"
    )
    print("\n[未来问题输入示例，仅展示]")
    print(query)
    print(
        f"[身份] contract={plan['contract_sha256'][:16]}…；本路线input_snapshot={plan['input_snapshot_sha256'][:16]}…"
    )
    print(
        f"[拟建新索引] {plan['planned_index']}；{'目标已存在，未来执行须另选新目录' if plan['planned_index'].exists() else '尚未创建'}。"
    )
    print(
        f"[待核验] {len(candidates)}条输入的Token数均未知；未核验部署实际上下文、模型digest、当前输出维度和批量容量；没有适配并验证的Tokenizer，不能使用其他模型的计数器充数。"
    )
    print(
        "[计划完成但未就绪] ready_for_embedding=false。没有读取.env、创建HTTP/Milvus客户端、访问服务器、安装Tokenizer或写入任何产物。"
    )


def self_test():
    """纯内存合成检查；不依赖本地服务或真实语料，不创建文件。"""
    code = """EMBEDDING_URL="http://example.invalid:11434/api/embed"
EMBEDDING_MODEL="qwen3-embedding:8b"
QUERY_INSTRUCTION="Retrieve technical passages"
payload={"truncate":False,"keep_alive":"10m"}
raise AssertionError("文档示例绝不执行")
"""
    guide = "| 实测向量维度 | `4096` |\n~~~python\n" + code + "~~~\n"
    profile = profile_from_document(guide)
    assert (
        profile["actual_dimension"] is None and profile["server_context_tokens"] is None
    )
    checks = 1
    for invalid in (
        guide.replace("False", "True"),
        guide.replace("False", "0"),
        guide.replace("False", "None"),
        guide.replace("False", '"false"'),
        guide.replace("4096", "unknown"),
        guide.replace("/api/embed", "/other"),
        guide.replace('"10m"', "unknown_function()"),
    ):
        try:
            profile_from_document(invalid)
        except (ValueError, TypeError):
            checks += 1
        else:
            raise AssertionError("不明确的指南必须拒绝")
    body = "  中文😀\n```text\n代码\n```\n"
    chunks = [
        {
            "id": "one",
            "page_content": body,
            "metadata": {
                "chunk_id": "one",
                "parent_id": "parent",
                "title": "测试文档",
                "sections": [["测试文档", "子节"]],
                "cleaned_char_start": 10,
                "cleaned_char_end": 10 + len(body),
                "feature_states": [{"state": "do-not-append"}],
            },
        }
    ]
    baseline = canonical(chunks)
    a, b = (make_inputs(chunks, mode, "a" * 64)[0] for mode in MODES)
    assert a["text"] == body and b["text"] == "文档与章节：测试文档 / 子节\n\n" + body
    assert (
        a["body_sha256"] == b["body_sha256"] and a["input_sha256"] != b["input_sha256"]
    )
    assert canonical(chunks) == baseline and "do-not-append" not in b["text"]
    assert a["input_utf8_bytes"] > a["input_chars"] and a["token_count"] is None
    checks += 1
    assert make_inputs(chunks, "body-only", "a" * 64)[0] == a
    checks += 1
    batches = make_batches(profile, [a] * 9, 8)
    assert [batch["items"] for batch in batches] == [8, 1]
    assert batches[-1]["start"] == 8 and batches[-1]["end"] == 9
    assert request_body(profile, [a])["input"] == [
        body
    ] and "parameters" not in request_body(profile, [a])
    checks += 1
    for bad in (0, -1, True, 1001):
        try:
            make_batches(profile, [a], bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("非法客户端批量应拒绝")
    bad = json.loads(baseline)
    bad[0]["metadata"]["sections"].append(["其他章节"])
    try:
        make_inputs(bad, "section-prefix", "a" * 64)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("多章节候选不能直接套前缀")
    query = profile["query_format"].format(
        instruction=profile["query_instruction"], query="问题"
    )
    assert "\nQuery: " in query and "\\n" not in query
    checks += 1
    print(
        f"[自测通过] {checks}组：AST不执行、配置拒绝、原文/Unicode/指纹/元数据隔离、批次、查询换行；零请求零文件写入。"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="仅客户端条件计划，不是服务端上限"
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="body-only",
        help="默认遵循指南，正文不附查询指令；两种模式均会展示离线对照",
    )
    parser.add_argument("--show-chunk", help="指定要展示的原始候选ID")
    parser.add_argument("--preview-chars", type=int, default=320)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000 or not 1 <= args.preview_chars <= 5000:
        parser.error("batch-size须为1～1000，preview-chars须为1～5000")
    if args.self_test:
        self_test()
        return
    plan = load_plan(args.corpus.resolve(), args.batch_size, args.mode)
    show_plan(plan, args.show_chunk, args.preview_chars)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(f"[预检失败] {error}；未发送模型请求，未写文件。", file=sys.stderr)
        raise SystemExit(2)
