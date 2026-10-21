# 零模型导出可迁移知识包

入口是 `scripts/export_knowledge.py`。这一步把已经保存的正文、元数据和向量导出为与本机路径解耦的知识包，为后续在 Linux 容器内重建 Milvus 做准备。**不重新编码，不复制运行中的 Milvus 数据目录，也不切换当前问答服务。**

默认来源仍是原稳定索引 `data/milvus_indexes/kubernetes_zh_v1/`：11篇、156块、4096维。其他新索引必须显式通过 `--index-dir` 指定；已有语料、SQLite 缓存和 Milvus 索引均保留。

## 使用顺序

在项目根目录、`langchain` 环境执行，也可把 `python` 换为 `/Users/patton/miniconda3/envs/langchain/bin/python`。

```bash
# 离线预检：检查既有 ready 清单、来源语料与已完成缓存，不写知识包
python -B scripts/export_knowledge.py

# 从完成缓存导出；目标目录必须尚不存在
python -B scripts/export_knowledge.py --run \
  --output-dir data/portable_bundles/kubernetes_zh_export

# 只检查导出的知识包，不再依赖原语料、SQLite 或 Milvus
python -B scripts/export_knowledge.py \
  --verify data/portable_bundles/kubernetes_zh_export
```

导出其他索引时，在 `--run` 后增加 `--index-dir data/milvus_indexes/另一个索引`，并使用新的 `--output-dir`。程序拒绝覆盖任何已有目标，包括空目录、失败后保留的部分目录及符号链接；需要重做时选择新目录，不自动删除旧产物。

复制整个包到另一目录后，再对新位置执行 `--verify`，可以检查迁移后仍能独立读回。验证需要 Python、NumPy 和本仓库代码，并不是一个无依赖的独立可执行程序。此验证不读取原语料、SQLite、Milvus 或 `.env`，不访问网络，也不调用模型。

## 三个产物分别是什么

| 文件 | 职责 |
| --- | --- |
| `records.jsonl` | 每行一个检索块，保留块身份、精确原正文及允许迁移的来源、章节、许可、审核等信息 |
| `vectors.npy` | 与记录顺序一一对应的二维向量矩阵，按小端 float32 保存；不是重新生成的向量 |
| `manifest.json` | 记录格式、数量、维度、模型输入合同和文件校验和，并内嵌来源 `LICENSE` 全文及其 SHA256，避免迁移时丢失许可声明 |

字段采用最小化白名单：不导出本机构建路径、服务地址、环境配置、依赖清单或完整原始 frontmatter。保留公开来源 URL、许可与署名、审核状态及 `feature_states` 功能状态前提；frontmatter 中的最低/最高 Kubernetes 服务端版本单独保存在 `version_constraints`，保留已有字符串或数值，不推断适用版本。**不会把未审核资料升级为已审核资料**。正文逐字保留，所以包内仍可能包含敏感正文、示例地址或内部信息；最小化元数据不等于脱敏，分享前仍需检查。

清单中的 `original_document_snapshot_sha256` 是原始文档快照的历史哈希。移除不适合迁移的字段后，新 `records.jsonl` 有自己的文件 SHA256，不能把两者说成同一份内容的相同哈希。当前格式只接受固定上游 commit 的 Kubernetes 中文 GitHub 来源、`CC-BY-4.0` 和 `unreviewed_official_document` 标记，不是通用任意知识库导出器。

## 模型合同与校验边界

知识包保留模型名称、digest、问题指令、文档 `body-only` 输入方式、4096维、小端 float32 与 COSINE 度量等合同。日后问题编码仍须使用兼容合同；**维度相同，不代表两个模型的向量可以混用。** 这里保存的是既有编码时记录的模型身份，不会联网证明当前服务权重仍未变化。

校验检查文件校验和、结构、记录与向量对应、数值及合同，并做存储层面的自余弦检查。格式当前最多1000块、8192维，足以覆盖本次子集，不冒充大规模通用导出。校验和用于发现意外损坏或改动，不是数字签名，也不能认证发布者；自余弦接近1不等于问句召回质量好、回答正确或资料适用于现场。

## 本次验收（2026-10-21）

已导出 `data/portable_bundles/kubernetes_zh_v2/`，156块、4096维，共2,986,316字节；保留24块的版本前提和49块的功能状态。随后复制到另一目录并运行 `--verify`，独立读回通过，首个已存文档向量自余弦约1，模型/网络请求均为0。早期 `kubernetes_zh_v1/` 导出仍保留，不覆盖；以补齐版本条件的v2为后续迁移来源。

100项专项离线测试覆盖换位置且原临时语料已移除仍可验证、逐字节向量一致、不覆盖、来源字段和合同校验、无网络请求及失败不发布正式清单。这不是跨操作系统实测；当前问答服务仍使用原索引，没有切换到便携包。生成的正文与向量目录被Git忽略，不随代码推送。

连同既有编码缓存、Kubernetes入库/搜索和Compose专项，共223项测试通过（4项既有依赖警告）。Compose只读检查仍为2条会话、2条问答，没有把本次离线校验写入业务数据库。

## 核心代码与下一步

- `src/qa_platform/storage/portable_knowledge.py`：来源预检、字段白名单、导出和独立读回校验。
- `scripts/export_knowledge.py`：命令行入口，默认只预检，显式 `--run` 才写入。
- 对应测试：覆盖不覆盖旧目录、篡改拒绝、迁移后验证及零模型调用等边界。

下一步是在 Linux 容器内读取这个包、复用向量重建 Milvus，再验证检索并制作应用镜像。本步尚未完成容器内入库、应用镜像或运行时索引切换，也没有重新做语义质量评测。
