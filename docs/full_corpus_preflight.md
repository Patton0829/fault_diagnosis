# Kubernetes 全量离线预检

这是规模盘点，不是全量入库。扫描本地固定快照的全部中文 `docs/**/*.md`，复用默认11篇/156块知识库的 `DocumentCleaner` 和 `pack_page`。不调用Embedding、Reranker或聊天模型，不读`.env`，不创建语料、SQLite缓存或Milvus索引，也不切换问答服务。

## 使用方法

在项目根目录使用 `langchain` 环境：

```bash
# 仅确认范围和计划，不逐篇清洗，也不写文件
python -B scripts/check_k8s_full_corpus.py

# 扫描并保存到一个尚不存在的新目录
python -B scripts/check_k8s_full_corpus.py --run \
  --output-dir reports/k8s-full-preflight-new
```

目录中生成 `report.html`（单页静态报告）和 `report.json`（逐篇指标、原因及依赖指纹）。报告不保存全文、文本块或向量；告警可能含原模板及少量上下文，分享前仍需检查。已有目录（包括失败留下的目录或符号链接）不覆盖。默认每100篇打印一次进度，不打印1665篇正文。

`--chunk-size` 默认800个Python字符，可设100～10000；`--batch-size` 默认8，可设1～256，只用于数学估算，不表示验证了模型批量上限。字符不换算成Token。

## 本次结果（2026-10-21）

固定快照：`2f85f2802e59eb2128543dbfbf114ad641db4be0`。最终报告保存在本机 `reports/k8s-full-preflight-2026-10-21-03/`，不提交Git。

| 去向 | 文件数 | 含义 |
| --- | ---: | --- |
| 通过本次规则 | 755 | 共1920个候选块，不是已审核或已编码 |
| 清洗告警 | 563 | 未支持模板/参数、缩进上下文等，整篇暂缓 |
| 读取或清洗未完成 | 130 | 129个未被旧解析器识别到有效frontmatter标题，另1个引用块代码围栏不受支持 |
| 切块待处理 | 46 | 孤立标题不能安全附着到正文，整篇暂缓 |
| 目录导航 | 163 | 本轮范围策略排除，不表示内容无价值 |
| draft/headless | 8 | 不作为独立正文参与 |
| 总计 | 1665 | 分类互斥且覆盖扫描范围 |

待处理共739篇，其最终块数未知；**1920不是全部1665个文件处理完后的最终块数**。按每批8块、从零编码这些候选，数学上需240批；4096维float32原始向量约30MiB，不含索引、正文、元数据或其他开销。当前实际模型请求为0。44块因保留完整代码而超过800字符，最长3186字符，尚未验证模型Token上限或可编码性，不估算总编码耗时。

### 能切块，不等于适合独立入库

755篇中，465篇是特性门数据页，146篇是术语数据页，共611篇；其余144篇也包含贡献指南等非运维内容。报告按路径提供角色提示，不冒充人工分类或独立文章判定。`_build`、`stages`、`removed`及版本条件仍须核对。

129个“无有效标题”中有125个在 `reference/setup-tools/kubeadm/generated/`，多为被上层页面引用的生成式命令片段；另外还有共享片段、工具README等，不能一概说上游文档有错。`quota-pod-namespace.md`实际上有标题，但UTF-8 BOM导致旧frontmatter解析器没有识别；本轮只标记兼容问题，不偷偷修改原规则。

原11篇在本次同范围统计仍为156块。该对照固定了切块规则；新215块多格式索引使用章节递归切分，不混为一套统计。全量结果不外推检索质量，也不证明运维主题已完整覆盖。

新增53项离线专项测试通过；连同既有目录结构、编码缓存、Kubernetes入库/搜索和便携包测试，共224项通过（4项既有依赖警告）。报告已在浏览器核对，截图为同目录`preview.png`；Compose只读检查仍为2条会话、2条问答。

## 核心代码与边界

- `src/qa_platform/ingestion/kubernetes_inventory.py`：逐篇清洗、切块、分词/字段预检、分类计数、依赖前后核对，以及静态报告渲染。只保存指标，不积攒全量正文。
- `scripts/check_k8s_full_corpus.py`：命令入口；默认仅计划，显式`--run`才执行扫描并保存报告。
- `tests/test_kubernetes_inventory.py`：临时合成快照测试，不访问模型服务，验证统计分母、来源变化、零网络、不覆盖和HTML转义等。

原始Markdown、引用的include/示例、站点配置、文件清单和规则源码在扫描前后进行指纹核对；观察到变化则停用编码估算并标为`source_changed`。前后观测一致不等于证明每个中间时刻都未变化，也没有重新认证上游Git blob。

模板相关计数基于清洗器告警，按文档去重；已支持但参数不兼容的模板也会计入。代码中的模板被保护，不能用全文正则冒充未知模板发生次数。同一篇可有多种原因，原因数量不可相加当作文档总数。HTML全部转义，无脚本和远程资源，不执行文档命令。

当前多格式接入与便携导出工具有1000块的学习性保护上限，**不是Milvus容量上限**。本报告不解除限制，也不把1920块直接交给旧入口；后续需先确定哪些文件应独立索引、处理高频清洗障碍，再准备新版本语料和编码计划。

## 把已通过的1920块写入本机Milvus

用户明确要求入库后，使用独立的 `scripts/prepare_k8s_full_corpus.py` 将本报告的755篇通过页重建为兼容原v2格式的新语料。该格式和原编码缓存、Milvus构建器没有1000块总数限制；无需改动绑定旧索引的规则源码。755篇仍保留特性门、术语等资料角色提示及完整frontmatter，不声称它们都是独立运维教程。739篇待处理文档不入库。

```bash
# 只执行一次，生成全新的冻结语料；已有目标拒绝覆盖
python -B scripts/prepare_k8s_full_corpus.py --run

# 使用自部署Ollama，按每批8块编码；中断后同命令续跑
python -B scripts/build_k8s_embeddings.py \
  --corpus data/k8s_corpora/full_preflight_v1 --run

# 不再编码，纯离线核验已提交缓存
python -B scripts/build_k8s_embeddings.py \
  --corpus data/k8s_corpora/full_preflight_v1 --verify

# 复用完成缓存，构建一个新的本机Milvus Lite索引
python -B scripts/build_k8s_milvus.py \
  --corpus data/k8s_corpora/full_preflight_v1 \
  --index-dir data/milvus_indexes/kubernetes_full_preflight_v1 --run
```

准备器会重新核对报告、全部源文档和依赖指纹、规则版本及逐页/总块数，复用原 `write_bundle()` 完成写入与读回。新增 `kubernetes_full_corpus.py` 只负责这个转换，命令入口不调用模型；专项测试为 `tests/test_full_corpus_build.py`。文档编码、SQLite事务检查点和Milvus跨进程验收复用原实现。Embedding仅发送候选正文到已经配置的内网服务，不调用云端Reranker或聊天；`truncate=false`，失败停批、不自动重试，已提交批次保留。

本次构建使用独立目录，保留旧156/215块索引，不切换问答服务。当前查询教学入口 `kubernetes_search.load_ready()` 仍限制1000块，不能直接给现有QA指定1920块新索引；检索入口扩容是后续步骤。索引`ready`只代表存储验收，不表示检索质量或回答正确性通过。

### 实际入库验收（2026-10-21）

- 新语料：`data/k8s_corpora/full_preflight_v1/`，755篇、1920块，旧v2格式严格读回核验通过。
- 编码缓存：`data/embedding_cache/kubernetes_zh_d554e2282033bcbf/embeddings.sqlite3`，1920/1920块、240批，状态`encoded_complete`，随后纯离线审计通过。使用内网Ollama `qwen3-embedding:8b`、4096维、body-only、`truncate=false`；原始语料的阶段标记未改写。
- 模型请求：240次本地Embedding＋480次批次前后digest GET＝720次模型服务HTTP；另有开始前连通性检查1次GET和入Milvus前1次GET，本轮合计722次。无自动重试，无云端Embedding、Reranker或聊天请求。服务返回批次处理Token合计266867，不是云端账单或逐文本Tokenizer验证；Embedding客户端请求耗时合计172.71秒，不是整个流程墙钟耗时。
- 新索引：`data/milvus_indexes/kubernetes_full_preflight_v1/`，集合`kubernetes_chunks_v1`，本机Milvus Lite、FLAT/COSINE，状态`ready`。写入PID64708退出后，独立PID64721重开，全部1920条原文、完整元数据与float32向量SHA一致。
- 存储层搜索验收：同一已存文档向量，全部分数与NumPy余弦对照最大差约`1.47e-07`；自相似度约1，铁路测试库过滤为空。没有调用问题编码，不冒充问句检索质量验证。
- 新增47项准备器测试通过；连同全量盘点、编码缓存和Milvus构建专项共147项测试通过（1项既有依赖警告）。旧156/215块索引严格合同仍通过，Compose只读检查仍为2条会话、2条问答。

本机粗略磁盘占用：语料9.1MiB、编码缓存38MiB、Milvus索引67MiB，均被Git忽略。索引已入库不意味着已有问答页面自动使用新数据；本轮没有修改查询容量限制或切换运行服务。

## 将1920块索引接入问答（2026-10-21）

完成上一阶段存储验收后，查询入口现已扩容并显式切换到新索引；前文“未切换服务”描述的是入库阶段结束时的状态。当前本机入口为 `http://127.0.0.1:8767/qa`，页面顶部显示755篇、1920块、4096维。755篇中包含参考数据页，未纳入739篇待处理文档，不表示知识覆盖或回答质量已验收。

```bash
# 先在旧应用终端按Ctrl-C，等待其正常退出；不停止MySQL/Redis容器
python -B scripts/serve_compose.py --serve \
  --index-dir data/milvus_indexes/kubernetes_full_preflight_v1
```

`--index-dir`只允许与`--serve`配合。启动先核验索引快照，再连接原有Docker MySQL/Redis；无效索引报错，不自动建库或回退。省略该参数仍保持原156块默认索引，因此重启时需要保留显式参数。原156/215块目录、问答历史和缓存均保留，没有迁移、清空或覆盖。

### 核心代码

- `src/qa_platform/storage/kubernetes_search.py`：将本机教学读取范围改为最多5000块，并同时限制原始float32矩阵为128MiB；先检查容量，再加载冻结计划。Milvus原文、元数据和向量每128条读回并核对指纹，按原计划顺序组成矩阵。128MiB不是进程总内存上限，也不是Milvus容量上限；文本、JSON、NumPy临时数组和检索器另有开销。
- `src/qa_platform/deployment.py`：增加显式索引启动参数，确保真实检索使用所选目录；不更改全局默认索引或删除旧数据。
- `src/qa_platform/retrieval_lab.py`与`web/qa.html`：沿用已有计划接口传递严格白名单的知识库摘要，页面仅增加一行范围提示；不新增请求模型的行为，也不展示本机索引路径。
- `tests/test_expanded_index.py`、`tests/test_expanded_deployment.py`、`tests/knowledge_status.test.cjs`及其pytest入口：用合成数据检查容量、分页完整性、CLI限制、失败清理、摘要校验与安全展示，不调用模型。

### 本次运行验收

- 同一DNS排查问题在1920块范围完成BM25、Milvus Dense和RRF，实际只有1次本地问题Embedding、2次digest GET，共3次模型服务HTTP；文档编码、云端重排和聊天均为0。Dense前3项为“调试Service”“调试DNS问题”“调试Service”，RRF首项为“调试Service”；排序变化不等于更准确，尚未增加这批资料的人工相关性标注。
- 当前工作区全套测试1527项通过，4项既有依赖警告；另将本轮暂存版本单独导出验证，扩容、部署、搜索及页面专项165项通过，避免依赖其他未提交功能。测试数量不是问答正确率。
- 旧应用正常退出后再启动新应用；GET计划接口与浏览器页面均核对到755/1920/4096。截图保存在本机原报告目录的`qa-1920.png`，不提交Git。
- MySQL只读检查仍为2条会话、2条问答。恢复历史可能仍来自旧索引，不会自动重检索；新提问使用当前索引，历史摘要仅帮助追问理解，不作为证据。Redis键包含快照等身份，新索引不会复用旧范围的检索结果，无需清空缓存。

此次只复用了已保存向量并切换入口，没有重新编码文档或自动执行云端问答。下一步可在页面提问、核对实际引用，再将代表性问题加入人工审核与回归评测。
