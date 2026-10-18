"""IT 运维知识库第一步：只读加载三篇文档，对照观察清洗，不切块或入库。

这是受控样本的 Markdown/Hugo 清洗演示，不是完整的网站渲染器。
未知模板保留并告警；代码和文档命令只是数据，绝不执行。
--compare-feature-state 用同一原文对照关闭/开启该项能力，不运行检索。
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import shlex
import sys

import yaml

from qa_platform.settings import PROJECT_ROOT

try:
    import tomllib
except ModuleNotFoundError:  # 用户指定的 Python 3.10 使用环境中已有的 tomli。
    import tomli as tomllib


CORPUS_ROOT = PROJECT_ROOT / "knowledge" / "kubernetes_zh"
LEGACY_CLEANER_VERSION = "k8s-demo-v1"
CLEANER_VERSION = "k8s-demo-v2-feature-state"
DEFAULT_FILES = (
    "tasks/debug/debug-application/debug-pods.md",
    "tasks/debug/debug-application/determine-reason-pod-failure.md",
    "tasks/administer-cluster/dns-debugging-resolution.md",
)
FEATURE_STATE_FILES = (
    "tasks/debug/debug-cluster/crictl.md",
    "tasks/debug/debug-application/debug-running-pod.md",
    "tasks/debug/debug-cluster/resource-metrics-pipeline.md",
)
FEATURE_STATES = {
    "alpha": "早期测试阶段",
    "beta": "测试阶段",
    "stable": "稳定阶段",
    "deprecated": "已弃用",
}
# 核对依据：同一快照的官方模板；本步只支持手工显式指定的版本和状态。
FEATURE_STATE_REFERENCE = (
    "https://github.com/kubernetes/website/blob/"
    "2f85f2802e59eb2128543dbfbf114ad641db4be0/layouts/shortcodes/feature-state.html"
)
HEADINGS = {
    "prerequisites": "准备开始",
    "whatsnext": "接下来",
    "objectives": "目标",
    "cleanup": "清理环境",
}
# 一个未闭合模板不能跨过下一个模板起点，否则可能把中间正文吞掉。
SHORTCODE = re.compile(
    r"\{\{(?:<(?P<angle>(?:(?!\{\{[<%]).)*?)>|%(?P<percent>(?:(?!\{\{[<%]).)*?)%)\}\}",
    re.DOTALL,
)
# 顺序很重要：先识别完整围栏，再识别注释和行内代码。
SPECIAL = re.compile(
    r"^ {0,3}(?:`{3,}|~{3,})[^\n]*(?:\n|$)|<!--|`+|^(?: {4}|\t)[^\n]*(?:\n|$)",
    re.MULTILINE,
)


def fingerprint(data: bytes) -> str:
    return sha256(data).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """只允许读取指定目录内的文件，不跟随越界路径或符号链接。"""
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"不允许绝对路径或上级路径：{relative}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"引用越出允许目录：{relative}")
    return path


def front_matter(text: str) -> tuple[dict, str]:
    """把开头的 YAML 变成元数据，而不是把标题和版本要求直接删掉。"""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("YAML front matter 缺少结束分隔线。")
    values = yaml.safe_load("".join(lines[1:end])) or {}
    if not isinstance(values, dict):
        raise ValueError("front matter 必须是键值对象。")
    return values, "".join(lines[end + 1 :])


class DocumentCleaner:
    """一个实例处理一篇文档，同时累计引用依赖、转换记录和待处理问题。"""

    def __init__(
        self, root: Path, site_params: dict, *, enable_feature_state: bool = True
    ):
        self.root = root.resolve()
        self.site_params = site_params
        self.dependencies: dict[str, dict] = {}
        self.counts: Counter = Counter()
        self.actions: dict[str, tuple[str, str]] = {}
        self.warnings: list[str] = []
        self.protected_code: list[str] = []
        self.admonitions: list[str] = []
        # 关闭开关只用于教学对照；其余清洗逻辑相同，不复制一套旧程序。
        self.enable_feature_state = enable_feature_state
        self.feature_states: list[dict] = []

    def feature_state(
        self, attributes: dict, positional: list[str], original: str, path: Path
    ) -> str:
        """把显式功能状态转为正文，同时保留原参数；不猜默认版本或特性门状态。"""
        if positional or attributes.keys() != {"state", "for_k8s_version"}:
            raise ValueError(
                "feature-state本步仅支持显式state和for_k8s_version；"
                "省略版本、feature_gate_name或额外参数仍需核对"
            )
        state, version = attributes["state"], attributes["for_k8s_version"]
        if state not in FEATURE_STATES:
            raise ValueError("feature-state状态须为alpha/beta/stable/deprecated")
        if not re.fullmatch(
            r"v?[1-9][0-9]*\.(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))?", version
        ):
            raise ValueError("feature-state版本须为明确的数字版本，如v1.25或1.37")
        self.feature_states.append(
            {
                "state": state,
                "for_k8s_version": version,
                "version_source": "explicit_shortcode_parameter",
                "source_path": path.relative_to(self.root).as_posix(),
                "original_shortcode": original,
            }
        )
        return (
            f"\n[功能状态] Kubernetes {version}：{state}（{FEATURE_STATES[state]}）。\n"
        )

    def read(self, path: Path) -> bytes:
        return path.read_bytes()

    def record(self, path: Path, data: bytes, kind: str) -> None:
        relative = path.relative_to(self.root).as_posix()
        self.dependencies[relative] = {
            "path": relative,
            "kind": kind,
            "sha256": fingerprint(data),
        }

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def segments(self, text: str):
        """先保护代码，再删除代码外的 HTML 注释，防止把示例代码误删。"""
        position = 0
        while match := SPECIAL.search(text, position):
            if match.start() > position:
                yield False, text[position : match.start()]
            token = match.group()
            if token == "<!--":
                close = text.find("-->", match.end())
                if close < 0:
                    raise ValueError("发现未闭合的 HTML 注释；停止，避免误删后续正文。")
                end = close + 3
                self.counts["comments_removed"] += 1
                # 留一个空格，避免 a<!-- comment -->b 被拼成 ab。
                yield False, " " + "\n" * text[match.start() : end].count("\n")
            elif opening := re.match(r"^ {0,3}(`{3,}|~{3,})", token):
                marker = opening.group(1)
                closing = re.compile(
                    r"^ {0,3}"
                    + re.escape(marker[0])
                    + "{"
                    + str(len(marker))
                    + r",}[ \t]*(?:\n|$)",
                    re.MULTILINE,
                )
                close = closing.search(text, match.end())
                end = close.end() if close else len(text)
                if close is None:
                    self.warn("存在未闭合代码围栏，已原样保留到文末，需人工核对。")
                self.counts["fenced_blocks_preserved"] += 1
                yield True, text[match.start() : end]
            elif token.startswith("`"):
                closing = re.compile(r"(?<!`)" + re.escape(token) + r"(?!`)")
                close = closing.search(text, match.end())
                end = close.end() if close else match.end()
                yield bool(close), text[match.start() : end]
            else:
                # 四空格/Tab 缩进可能是代码，也可能是列表续行；保守保护而非猜测。
                end = match.end()
                self.warn(
                    "检测到围栏外四空格/Tab 缩进行：保守原样保留，需核对其 Markdown 上下文。"
                )
                yield True, text[match.start() : end]
            position = end
        if position < len(text):
            yield False, text[position:]

    def body(self, text: str, path: Path, params: dict, stack: tuple[Path, ...]) -> str:
        if re.search(r"^ {0,3}(?:>[ \t]*)+(?:`{3,}|~{3,})", text, re.MULTILINE):
            raise ValueError("本演示不处理引用块中的代码围栏；停止，原文件未改动。")
        output = []
        # 合并相邻正文片段，防止删注释后留下大量空行。
        plain = []

        def flush():
            prose = re.sub(r"^[ \t]+$", "", "".join(plain), flags=re.MULTILINE)
            prose = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", prose)
            unmatched = SHORTCODE.sub("", prose)
            if re.search(r"\{\{[<%]", unmatched):
                self.warn(
                    f"{path.relative_to(self.root)}：存在未闭合或跨代码边界的模板标记，已保留原文。"
                )
            # 先压空行，再展开示例，不能压掉外部 YAML 内部的空行或缩进。
            output.append(
                SHORTCODE.sub(lambda m: self.shortcode(m, path, params, stack), prose)
            )
            plain.clear()

        for protected, part in self.segments(text):
            if protected:
                flush()
                self.protected_code.append(part)
                output.append(part)
            else:
                plain.append(part)
        flush()
        return "".join(output)

    def shortcode(
        self, match: re.Match, path: Path, params: dict, stack: tuple[Path, ...]
    ) -> str:
        original = match.group()
        try:
            words = shlex.split(match.group("angle") or match.group("percent") or "")
            if not words:
                raise ValueError("空模板")
            name, arguments = words[0], words[1:]
            attribute_words = [word for word in arguments if "=" in word]
            positional = [word for word in arguments if "=" not in word]
            attributes = dict(word.split("=", 1) for word in attribute_words)
            if len(attributes) != len(attribute_words):
                raise ValueError("存在重复模板属性")
            if name == "heading" and len(arguments) == 1:
                result = HEADINGS[arguments[0]]
            elif (
                name == "param"
                and len(arguments) == 1
                and isinstance(params.get(arguments[0]), str)
            ):
                result = params[arguments[0]]
            elif (
                name == "glossary_tooltip"
                and not positional
                and "text" in attributes
                and attributes.keys() <= {"text", "term_id"}
            ):
                result = attributes["text"]
            elif name in ("note", "warning", "caution") and not arguments:
                self.admonitions.append(name)
                result = (
                    "\n["
                    + {"note": "说明", "warning": "警告", "caution": "注意"}[name]
                    + "]\n"
                )
            elif name in ("/note", "/warning", "/caution") and not arguments:
                if not self.admonitions or self.admonitions[-1] != name[1:]:
                    raise ValueError("说明/警告模板的结束标记不匹配")
                self.admonitions.pop()
                result = (
                    "\n["
                    + {"/note": "说明", "/warning": "警告", "/caution": "注意"}[name]
                    + "结束]\n"
                )
            elif (
                name == "version-check"
                and not arguments
                and params.get("min-kubernetes-server-version")
            ):
                minimum = str(params["min-kubernetes-server-version"])
                requirement = "必须为" if minimum == params.get("version") else "至少为"
                result = f"\n[版本前提] 本文要求 Kubernetes 服务端版本{requirement} {minimum}；可用 `kubectl version` 核对。\n"
            elif name == "feature-state" and self.enable_feature_state:
                result = self.feature_state(attributes, positional, original, path)
            elif name == "include" and len(positional) == 1 and not attributes:
                candidates = [
                    safe_path(self.root / "content/zh-cn/includes", arguments[0]),
                    safe_path(path.parent, arguments[0]),
                ]
                target, data = self.first_available(candidates)
                if target in stack or len(stack) >= 8:
                    raise ValueError("include 存在循环或超过 8 层")
                self.record(target, data, "include")
                extra, included = front_matter(data.decode("utf-8"))
                result = self.body(
                    included, target, {**params, **extra}, (*stack, target)
                )
            elif (
                name in ("code_sample", "codenew")
                and not positional
                and "file" in attributes
                and attributes.keys() <= {"file", "language", "options"}
            ):
                file = attributes["file"]
                zh = safe_path(self.root / "content/zh-cn/examples", file)
                local = safe_path(path.parent, file)
                # 按当前快照官方 shortcode：code_sample 可回退英文，codenew 不回退。
                candidates = (
                    [zh, safe_path(self.root / "content/en/examples", file), local]
                    if name == "code_sample"
                    else [local, zh]
                )
                target, data = self.first_available(candidates)
                self.record(target, data, "example")
                code = data.decode("utf-8")
                marker = "`" * max(
                    3,
                    1
                    + max(
                        (len(m.group()) for m in re.finditer(r"`+", code)), default=0
                    ),
                )
                language = attributes.get("language", target.suffix.lstrip("."))
                if not re.fullmatch(r"[\w+-]*", language):
                    raise ValueError("不支持的代码语言标记")
                origin = target.relative_to(self.root).as_posix()
                result = (
                    f"\n示例来源：{origin}\n\n{marker}{language}\n{code}"
                    + ("" if code.endswith("\n") else "\n")
                    + marker
                    + "\n"
                )
                self.protected_code.append(code)
            else:
                raise ValueError("本演示尚不支持此模板或参数")
            self.counts[name] += 1
            self.actions.setdefault(name, (original, result))
            return result
        except (KeyError, ValueError, FileNotFoundError) as error:
            self.warn(f"{path.relative_to(self.root)}：{original}；{error}")
            return original  # 不静默删除未知语义，也不联网补齐。

    def first_available(self, candidates: list[Path]) -> tuple[Path, bytes]:
        for candidate in candidates:
            try:
                return candidate, self.read(candidate)
            except FileNotFoundError:
                continue
        raise FileNotFoundError("本地快照中缺少引用资源")

    def load(self, relative: str, commit: str) -> dict:
        path = safe_path(self.root / "content/zh-cn/docs", relative)
        if path.suffix != ".md":
            raise ValueError("本步仅接受 docs 目录下的 Markdown 文件。")
        data = self.read(path)
        raw = data.decode("utf-8")
        fields, body = front_matter(raw)
        title = fields.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("正文缺少有效 title，不能悄悄用文件名替代。")
        cleaned = f"# {title}\n\n" + self.body(
            body, path, {**self.site_params, **fields}, (path,)
        ).lstrip("\n")
        if not cleaned.endswith("\n"):
            cleaned += "\n"
        if self.admonitions:
            self.warn(
                f"说明/警告模板未闭合：{self.admonitions}；需人工核对其作用范围。"
            )
        if not all(code in cleaned for code in self.protected_code):
            raise ValueError("代码原文保护检查未通过。")
        source = path.relative_to(self.root).as_posix()
        return {
            "raw": raw,
            "cleaned": cleaned,
            "metadata": {
                "title": title,
                "kb_id": "kubernetes_zh",
                "source": str(path),
                "source_url": f"https://github.com/kubernetes/website/blob/{commit}/{source}",
                "snapshot_commit": commit,
                "kubernetes_release": None,
                "license": "CC-BY-4.0",
                "cleaner_version": CLEANER_VERSION
                if self.enable_feature_state
                else LEGACY_CLEANER_VERSION,
                **(
                    {
                        "feature_states": list(self.feature_states),
                        "feature_states_scope": "document_including_expanded_includes_not_per_chunk",
                    }
                    if self.feature_states
                    else {}
                ),
                "raw_sha256": fingerprint(data),
                "cleaned_sha256": fingerprint(cleaned.encode("utf-8")),
                "front_matter": fields,
                "dependencies": list(self.dependencies.values()),
                "is_navigation": path.name == "_index.md",
                "cleaned_line_numbers_are_source_lines": False,
            },
        }


def preview(text: str, limit: int) -> None:
    lines = text.splitlines()
    print("\n".join(lines[:limit]))
    if len(lines) > limit:
        print(
            f"……仅预览前 {limit} 行，完整内容 {len(lines)} 行；不是已截断的存储正文。"
        )


def compare_feature_state(
    root: Path, site_params: dict, config_data: bytes, files: list[str], commit: str
) -> int:
    """同一份原文件只切换一项能力；不切块、不写文件，也不是检索质量评测。"""
    print(
        "[实验] 旧行为（关闭feature-state支持） vs 新行为（开启）；其他清洗规则相同。"
    )
    print("[控制条件] 同一原文件、站点配置与快照；原文件指纹相等才展示对照。")
    problems, converted = 0, 0
    for relative in files:
        runs = []
        for enabled in (False, True):
            cleaner = DocumentCleaner(root, site_params, enable_feature_state=enabled)
            cleaner.record(root / "hugo.toml", config_data, "site_config")
            runs.append((cleaner, cleaner.load(relative, commit)))
        old, old_result = runs[0]
        new, new_result = runs[1]
        if (
            old_result["raw"] != new_result["raw"]
            or old.dependencies != new.dependencies
        ):
            raise ValueError("对照期间原文件或依赖发生变化，停止比较")
        print(f"\n[文档] {new_result['metadata']['title']} | {relative}")
        print(
            f"[旧行为] 转换{old.counts['feature-state']}处；待处理提示{len(old.warnings)}条"
        )
        print(
            f"[新行为] 转换{new.counts['feature-state']}处；待处理提示{len(new.warnings)}条"
        )
        for item in new.feature_states:
            print(f"[原模板] {item['original_shortcode']}")
            print(
                f"[新正文] [功能状态] Kubernetes {item['for_k8s_version']}：{item['state']}（{FEATURE_STATES[item['state']]}）。"
            )
        if not new.feature_states:
            print("[提示] 本篇没有成功转换的显式feature-state；不会猜测缺失参数。")
        for warning in new.warnings:
            print(f"[仍待处理] {warning}")
        print(f"[原文指纹] 两路相同：{new_result['metadata']['raw_sha256'][:16]}…")
        print(f"[来源] {new_result['metadata']['source_url']}")
        problems += bool(new.warnings)
        converted += new.counts["feature-state"]
    print(
        f"\n[实验完成] {len(files)}篇，共转换{converted}处；仍有{problems}篇存在清洗提示，尚未核查切块。"
    )
    print(
        "[边界] 版本字符串原样保留；功能状态不是全文最低版本，不代表默认启用，也不是已核验的现场集群版本。"
    )
    print(
        "[未支持] 省略版本、feature_gate_name自动查资料及其他参数仍保留并告警；代码中的模板不转换。"
    )
    print(f"[规则依据] {FEATURE_STATE_REFERENCE}")
    print("[成本] 零模型请求；运行不联网、不读.env、不改上游、不写语料或报告。")
    return 0


def self_test() -> None:
    """使用内存虚构文件测试，不创建文件、不依赖真实集群或模型。"""
    root = Path("/virtual/k8s").resolve()
    page = "content/zh-cn/docs/test.md"

    def run(
        body: str,
        extras: dict | None = None,
        header: str = "title: 测试\n",
        *,
        enabled: bool = True,
    ):
        cleaner = DocumentCleaner(
            root, {"version": "v1.37"}, enable_feature_state=enabled
        )
        files = {
            root / page: (f"---\n{header}---\n" + body).encode(),
            **{root / k: v.encode() for k, v in (extras or {}).items()},
        }

        def read(path):
            if path not in files:
                raise FileNotFoundError(str(path))
            return files[path]

        cleaner.read = read
        return cleaner, cleaner.load("test.md", "a" * 40)

    checks = 0
    code = "```shell\n<!-- keep -->\n{{< unknown >}}\n{{range .items}}{{end}}\n```\n"
    cleaner, result = run("甲<!-- English -->乙\n" + code + "`<!-- inline -->`\n")
    assert "English" not in result["cleaned"] and "甲 乙" in result["cleaned"]
    assert code in result["cleaned"] and "`<!-- inline -->`" in result["cleaned"]
    assert not cleaner.warnings
    checks += 1
    cleaner, result = run("    <!-- indented code -->\n")
    assert "<!-- indented code -->" in result["cleaned"] and cleaner.warnings
    checks += 1
    cleaner, result = run(
        '{{< include "shared.md" >}}\n{{% code_sample file="test.yaml" %}}',
        {
            "content/zh-cn/includes/shared.md": "<!-- English -->前提条件\n",
            "content/en/examples/test.yaml": "kind: Pod\n# ```\n\n  value: x\n",
        },
    )
    assert "前提条件" in result["cleaned"] and "English" not in result["cleaned"]
    assert "kind: Pod\n# ```\n\n  value: x\n" in result["cleaned"]
    assert len(result["metadata"]["dependencies"]) == 2 and not cleaner.warnings
    checks += 1
    cleaner, result = run(
        '{{< codenew file="only-en.yaml" >}}',
        {"content/en/examples/only-en.yaml": "must not silently use fallback"},
    )
    assert "{{< codenew" in result["cleaned"] and cleaner.warnings
    checks += 1
    for body, extras in (
        ("{{< unknown >}}", {}),
        ('{{< include "missing.md" >}}', {}),
        ('{{< include "../secret.md" >}}', {}),
        (
            '{{< include "loop.md" >}}',
            {"content/zh-cn/includes/loop.md": '{{< include "loop.md" >}}'},
        ),
    ):
        cleaner, result = run(body, extras)
        assert cleaner.warnings and "{{<" in result["cleaned"]
        checks += 1
    cleaner, result = run(
        "{{< version-check >}}",
        header="title: 测试\nmin-kubernetes-server-version: v1.6\n",
    )
    assert (
        "至少为 v1.6" in result["cleaned"]
        and result["metadata"]["kubernetes_release"] is None
    )
    checks += 1
    for body, header in (("正文", "weight: 1\n"), ("<!-- unclosed", "title: 测试\n")):
        try:
            run(body, header=header)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("本应拒绝不完整输入")
    cleaner, result = run("~~~text\n<!-- keep to EOF -->\n")
    assert "<!-- keep to EOF -->" in result["cleaned"] and cleaner.warnings
    checks += 1
    cleaner, result = run("~~~text\nkeep trailing blank lines\n\n\n")
    assert result["cleaned"].endswith("keep trailing blank lines\n\n\n")
    checks += 1
    cleaner, result = run('{{< include "broken.md"')
    assert '{{< include "broken.md"' in result["cleaned"] and cleaner.warnings
    checks += 1
    cleaner, result = run(
        '{{< glossary_tooltip text="插件" >\n重要正文不能丢\n{{< note >}}\n继续\n'
    )
    assert "重要正文不能丢" in result["cleaned"] and cleaner.warnings
    checks += 1
    cleaner, result = run('{{< glossary_tooltip text="插件" unexpected >}}')
    assert "unexpected" in result["cleaned"] and cleaner.warnings
    checks += 1
    for body in ("{{< warning >}}\n重要警告\n", "{{< /warning >}}"):
        cleaner, result = run(body)
        assert cleaner.warnings
        checks += 1
    try:
        run("> ~~~html\n> <!-- keep -->\n> ~~~\n")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("本演示应拒绝不支持的引用块代码，而不是误删其内容")
    for state in FEATURE_STATES:
        cleaner, result = run(
            f'{{{{< feature-state for_k8s_version="v1.25" state="{state}" >}}}}'
        )
        assert f"v1.25：{state}（{FEATURE_STATES[state]}）" in result["cleaned"]
        assert not cleaner.warnings and cleaner.counts["feature-state"] == 1
        assert result["metadata"]["kubernetes_release"] is None
        checks += 1
    first = '{{< feature-state for_k8s_version="1.37" state="stable" >}}'
    second = '{{% feature-state state="stable" for_k8s_version="1.37" %}}'
    old, old_result = run(first, enabled=False)
    new, new_result = run(first)
    _, reordered = run(second)
    assert first in old_result["cleaned"] and old.warnings
    assert old_result["metadata"]["cleaner_version"] == LEGACY_CLEANER_VERSION
    assert new_result["cleaned"] == reordered["cleaned"] and not new.warnings
    assert new_result["metadata"]["feature_states"][0]["for_k8s_version"] == "1.37"
    assert old_result["metadata"]["raw_sha256"] == new_result["metadata"]["raw_sha256"]
    checks += 1
    for marker in (
        '{{< feature-state state="stable" >}}',
        '{{< feature-state feature_gate_name="NodeSwap" >}}',
        '{{< feature-state state="stabl" for_k8s_version="v1.25" >}}',
        '{{< feature-state state="stable" for_k8s_version="" >}}',
        '{{< feature-state state="stable" for_k8s_version="1.37; command" >}}',
        '{{< feature-state state="stable" for_k8s_version="v1.2５" >}}',
        '{{< feature-state state="stable" for_k8s_version="v1.25" extra="x" >}}',
        '{{< feature-state state="stable" state="beta" for_k8s_version="v1.25" >}}',
        '{{< feature-state state="stable" for_k8s_version="v1.25" positional >}}',
        "{{< /feature-state >}}",
    ):
        cleaner, result = run(marker)
        assert marker in result["cleaned"] and cleaner.warnings
        assert not cleaner.feature_states and cleaner.counts["feature-state"] == 0
        checks += 1
    cleaner, result = run(
        f"```text\n{first}\n```\n`{first}`\n<!-- {first} -->\n{first}\n{first}\n"
    )
    assert (
        f"```text\n{first}\n```" in result["cleaned"]
        and f"`{first}`" in result["cleaned"]
    )
    assert len(cleaner.feature_states) == 2 and cleaner.counts["feature-state"] == 2
    assert cleaner.counts["comments_removed"] == 1 and not cleaner.warnings
    checks += 1
    print(
        f"[自测通过] {checks} 组：代码保护、引用展开、路径边界、功能状态转换/拒绝/旧新对照、版本与不完整输入。"
    )
    print("[边界] 内存合成测试，不证明全库清洗正确；零模型请求、未写测试文件。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        action="append",
        help="相对中文 docs 目录的 .md 路径；可多次指定，默认三篇样本",
    )
    parser.add_argument(
        "--preview-lines",
        type=int,
        default=18,
        help="每份原始/清洗正文预览行数，默认18",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--self-test", action="store_true", help="仅运行内存自测，不读取语料"
    )
    mode.add_argument(
        "--compare-feature-state",
        action="store_true",
        help="只读对照关闭/开启feature-state支持，默认三篇受影响文档",
    )
    args = parser.parse_args()
    if not 1 <= args.preview_lines <= 120:
        parser.error("--preview-lines 必须在 1～120 之间。")
    if args.self_test:
        self_test()
        return 0
    manifest = json.loads(
        (CORPUS_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    commit = manifest["source"]["commit_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("来源清单的提交编号格式不正确。")
    root = CORPUS_ROOT / "upstream"
    config_path = root / "hugo.toml"
    config_data = config_path.read_bytes()
    config = tomllib.loads(config_data.decode("utf-8"))
    site_params = {
        **config.get("params", {}),
        **config.get("languages", {}).get("zh-cn", {}).get("params", {}),
    }
    files = args.file or (
        FEATURE_STATE_FILES if args.compare_feature_state else DEFAULT_FILES
    )
    print(f"[配置] Python：{sys.executable}；清洗规则：{CLEANER_VERSION}")
    print(f"[资料] 官方文档快照 {commit}；许可 CC BY 4.0；选择 {len(files)} 篇")
    if args.compare_feature_state:
        return compare_feature_state(root, site_params, config_data, files, commit)
    print(
        "[控制条件] 同一份原始文件，只比较不清洗与清洗；不修改上游，不联网，不读.env，零模型请求。"
    )
    print("[观察前先预测] 英文注释会怎样影响检索？删掉示例引用而不展开，会丢失什么？")
    problems = 0
    for relative in files:
        cleaner = DocumentCleaner(root, site_params)
        cleaner.record(config_path, config_data, "site_config")
        result = cleaner.load(relative, commit)
        raw, cleaned, metadata = result["raw"], result["cleaned"], result["metadata"]
        print(f"\n[文档] {metadata['title']} | {relative}")
        print(
            f"[原始/清洗] 字符 {len(raw)} → {len(cleaned)}；行数 {len(raw.splitlines())} → {len(cleaned.splitlines())}"
        )
        print(f"[变化] {dict(cleaner.counts)}；已识别代码的原文保护检查通过")
        print("[对照一：原始源码]")
        preview(raw, args.preview_lines)
        print("[对照一：清洗后正文]")
        preview(cleaned, args.preview_lines)
        for name in (
            "include",
            "code_sample",
            "codenew",
            "version-check",
            "feature-state",
        ):
            if name in cleaner.actions:
                before, after = cleaner.actions[name]
                print(f"[对照二：{name}] {before} → 以下展开预览")
                preview(after.strip("\n"), 9)
        print(f"[来源] {metadata['source_url']}")
        print(
            f"[指纹] 原始SHA256={metadata['raw_sha256'][:16]}…；清洗后={metadata['cleaned_sha256'][:16]}…"
        )
        print(
            f"[元数据] front_matter={json.dumps(metadata['front_matter'], ensure_ascii=False, default=str)}"
        )
        for dependency in metadata["dependencies"]:
            print(
                f"[依赖] {dependency['kind']} | {dependency['path']} | SHA256={dependency['sha256'][:16]}…"
            )
        for warning in cleaner.warnings:
            print(f"[待处理] {warning}")
        problems += bool(cleaner.warnings)
    print(
        f"\n[完成] {len(files)} 篇加载并预览；{problems} 篇有待处理提示；未切块、未向量化、未入库、未写报告。"
    )
    print("[理解边界] 字符数不是Token数；变短不等于检索更准；本次未测检索质量。")
    print(
        "[解析边界] 这是受控样本演示，不是完整Markdown/Hugo渲染器，也未审核文档内容是否适用于现场环境。"
    )
    print(
        "[来源边界] SHA256是本次本地内容指纹，不是对GitHub的再次在线校验；清洗行号不等于原文行号。"
    )
    print(
        "[下一步前] 人工核对标题、前提、命令、示例与警告是否保留；有未知模板不能直接当作清洗合格入库。"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"[错误] {error}", file=sys.stderr)
        raise SystemExit(2)
