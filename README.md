# 知识库智能问答平台

本项目通过可运行的对照实验，逐步学习文档清洗、切块、Embedding、BM25、RRF、Reranker、Milvus 和带来源引用的 RAG。最初使用铁路故障诊断测试资料，当前正在接入 Kubernetes 中文文档，构建独立的 IT 运维知识库。

目前，铁路测试库已跑通最小 RAG 与 Milvus 对照链路；Kubernetes 当前11篇、156块已完成本地编码与独立 Milvus 入库，并提供真实问句检索、独立的相邻块上下文补全实验，以及浏览器中的BM25／Dense／RRF、云端重排和带引用回答。**网页按步骤显式运行：打开页面0模型，检索、重排、回答各自主动触发；没有标注评测，不把运行成功等同于答案正确。网页当前不混入上下文补全。**

## 目录结构

```text
fault_diagnosis/
├── README.md
├── pyproject.toml                 # 项目包、直接依赖、测试配置
├── src/
│   └── qa_platform/
│       ├── settings.py            # 项目路径、密钥文件位置、接口文档位置
│       ├── providers/             # 模型调用适配器
│       ├── ingestion/             # 文档处理、语料准备与输入规划
│       │   └── _snapshots/        # 历史规则源码快照，只用于旧语料校验
│       ├── retrieval/             # BM25、Dense、RRF和重排流水线
│       ├── storage/               # 本地编码缓存、Milvus验证/建库/搜索
│       ├── rag/                   # 回答、引用校验、评测和Milvus对照
│       ├── web_app.py             # 文档处理可视化的后端
│       └── retrieval_lab.py       # 独立检索学习台，不改变原清洗台的零模型约定
├── scripts/                      # 26个运行入口，不承载业务实现
├── tests/                        # 目录/语料兼容、缓存与入库的离线测试
├── docs/
│   ├── learning/                 # 本机学习笔记，暂不上传
│   ├── model_apis/               # 脱敏模板；原接口说明只保留在本机
│   └── reference/                # 本机个人参考资料，不上传
├── models/                       # 暂留原位的.env；不是模型权重目录
├── knowledge/                    # 原始知识资料、上游快照及来源清单
├── data/                         # 生成语料和数据库，不提交Git
├── web/                          # 可视化前端HTML
├── assets/                       # 笔记截图和界面图片
└── reports/                      # 既有评测报告
```

职责划分：`src/` 中的函数供程序复用；`scripts/` 只负责将原来的命令转交给对应模块。模块内部的算法、教学输出和参数解析保留原实现，本次没有重新设计算法。包内使用 `qa_platform...` 绝对导入，不再依赖其他教学脚本恰好位于同一个目录。

`knowledge/` 保存输入资料，`data/` 保存可重新生成的处理结果，两者不能混为一谈。`storage/embedding_cache.py` 用 Python 自带的 SQLite 保存编码检查点；SQLite 在本步不进行向量检索，后续向量搜索仍交给 Milvus。编码成功后入库失败，可以重用缓存，避免重新调用模型。

## 环境与运行方式

使用现有虚拟环境，不需要重新创建环境：

```bash
cd /path/to/fault_diagnosis
conda activate langchain
python --version
```

为了直接使用新的包模块，在项目根目录执行一次本地可编辑安装：

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

这条命令只将本项目注册到该环境，不下载、安装或升级第三方依赖；源码修改后无需重复安装。`pyproject.toml` 声明实际直接依赖，不是整个 Conda 环境的完整锁定文件。Milvus 已验证组合放在 `milvus` 可选依赖组，测试工具放在 `dev` 组。

已有命令仍然有效，而且兼容入口会自行定位 `src/`，可从其他工作目录通过绝对路径运行：

```bash
python \
  scripts/check_local_embedding.py
```

完成可编辑安装后，也可以用模块入口：

```bash
python -m qa_platform.providers.local_embedding
```

上述两种写法调用同一份实现。不要直接执行 `src/qa_platform/` 内部的 `.py` 文件；使用兼容入口或 `python -m`。当前路径约定面向本仓库的源码开发，不是已经验证的脱离仓库部署方案。

## 当前模型路线与密钥

后续 Kubernetes 问答链路采用：

- 文档编码和问题编码：本地自部署 Ollama `qwen3-embedding:8b`，已验证输出为 4096 维。
- 检索后重排：云端 Reranker，仅提交召回候选，不重排全部知识库。
- 答案生成：云端聊天模型，仅提交问题和选定证据，不发送整库资料。

“本地模型”是指自部署服务，并不表示模型运行在这台 Mac 上。问题与文档向量必须遵循相同的模型版本和输入约定；Reranker 和生成模型处理文本，可以与 Embedding 来自不同供应商。

目录整理没有切换旧演示链路的默认模型：`cloud_embedding.py` 仍保留旧云端 Embedding 演示，`chat.py` 仍支持原有聊天选项；Kubernetes 本地编码不复用旧云端向量索引。新的网页回答默认 `qwen3.8-flash`，也可显式选择已有 `qwen3.8-max` 或 `glm` 配置。GLM现有配置使用明文HTTP，不使用千问密钥；页面会显示目的地址和传输风险，不能向其提交敏感资料。

密钥继续放在本机的 `models/.env`，不提交Git。该路径统一由 `settings.ENV_PATH` 管理。原接口资料可能包含部署地址或字面量密钥，因此只保留本地；公开的 [配置说明与脱敏模板](docs/model_apis/README.md) 不含真实凭据。不要执行资料中的服务重启等运维命令来验证代码。

本地 Embedding 脚本不读取 `.env`，也不在失败时回退云端。云端重排和生成仍可能产生费用，并会接收候选原文；接入内部资料前需另行核对数据外发权限。

## 常用命令

以下命令在项目根目录执行，均使用指定环境。

### 查看本地编码计划

```bash
python -B scripts/check_k8s_embedding_plan.py
python -B scripts/check_local_embedding.py
```

两条命令均不发起模型请求、不创建向量缓存，也不连接 Milvus。前者比较仅正文和附加章节标题两种编码输入；后者规划包含最长正文的 8 块小样本验证。

### 执行本地 Embedding 小样本验证

```bash
python -B scripts/check_local_embedding.py --run
```

该命令会真实请求内网服务器：批量 8 块、单独编码同一个最长块、编码一个问题，共 3 次推理，另有 3 次模型元数据 HTTP 请求。它不保存向量、不全量编码、不调用 Reranker 或生成模型。小样本成功不等于整库通过，也不代表检索准确率。

### 缓存文档向量与观察断点续跑

先查看计划，再仅编码并保存第一批：

```bash
python -B scripts/build_k8s_embeddings.py
python -B scripts/build_k8s_embeddings.py --run --max-batches 1
```

`--max-batches 1` 表示本次最多新增提交一批，已缓存批次不占额度。默认每批8块，第一次正常完成后应看到已提交8/156块、状态仍为 `building`，而不是“全量完成”。仅这一步正常需要1次文档Embedding和前后2次模型digest元数据请求，共3次HTTP。

随后纯离线读回，再看请求数如何减少：

```bash
python -B scripts/build_k8s_embeddings.py --verify
python -B scripts/build_k8s_embeddings.py
```

若首批已保存，剩余148块对应19次Embedding。确认结果后，去掉批次上限继续：

```bash
python -B scripts/build_k8s_embeddings.py --run
```

每批正常完成需1次Embedding和2次digest读取；从零完成20批是20次Embedding、40次元数据请求，共60次HTTP。若失败，请求数可能更少；不自动重试，也不回退云端。问题Embedding、Reranker、生成模型和Milvus请求均为0。

默认缓存位置为 `data/embedding_cache/kubernetes_zh_<合同指纹>/embeddings.sqlite3`，终端会打印实际路径。缓存有三张表：

| 表 | 保存内容 | 用途 |
|---|---|---|
| `metadata` | 语料和输入合同、模型digest、维度、状态 | 判断是否允许复用 |
| `vectors` | 块ID、原正文与完整元数据、float32向量及指纹 | 后续入库的可核验输入 |
| `batches` | 已提交块清单、请求内容指纹、处理Token与耗时 | 跟踪批次提交与处理量 |

缓存规则：

- 每批向量和账本放在同一事务中提交；只复用读回检查通过的记录。中断时不能仅凭屏幕日志判断最后一批是否提交，重跑会先核验缓存再计算缺失。
- 单进程锁覆盖缓存核验、网络请求和提交全过程；同一缓存不允许两个写入进程同时编码。
- 首次初始化完成后才排他发布正式库。强制终止可能留下 `.initializing-*` 临时文件，它们不算有效缓存，也不会被当成完成进度。
- 默认计划和 `--verify` 只读、不建库。`--run` 允许 SQLite 恢复本工具已有的事务日志，再核验内容；损坏或不兼容数据不会被自动覆盖修复。
- 全部有效记录提交并核验后，状态为 `encoded_complete`。再次 `--run` 不创建HTTP客户端，Embedding与元数据请求都为0；这只证明缓存可读，不证明此刻服务器模型没有变化，查询或入库前仍需核验契约。
- 当前仅支持同一语料与输入合同的续跑，允许调整批量大小1～8。修改语料、输入规则或模型配置时要另建缓存；跨语料版本增量更新尚未实现。
- 前后digest只是两个观测时刻的模型别名检查，不是逐次推理权重的独立证明。成功调用和文件指纹也不证明检索质量。

本步骤继续使用正文输入和 `truncate: false`，不重切、不静默截断。Token只记录服务返回的整批处理量，不虚构逐块Token数；批次失败会显示相关块ID，以便后续定位。

### 复用编码缓存导入独立 Milvus 索引

当前156块编码完成后，下一步不是重新调用Embedding，而是将保存的向量、原文和完整元数据导入专门的Kubernetes集合。SQLite保留编码检查点，Milvus提供后续的向量检索，两者职责不同。

```bash
python -B scripts/build_k8s_milvus.py
python -B scripts/build_k8s_milvus.py --run
```

第一条命令只读当前语料与完整缓存，不导入Milvus SDK、不创建客户端或文件、不发HTTP。第二条才执行：

1. 检查目标目录尚不存在、缓存已经 `encoded_complete`、依赖版本匹配；再通过内网Ollama的 `/api/tags` 查询一次模型digest，与缓存身份比对。不是推理请求，不发送文档正文。
2. 在全新目录冻结构建计划；写入进程从缓存读出原文、元数据和4096维float32向量，创建独立 `kubernetes_chunks_v1` 集合及 `FLAT/COSINE` 索引。每32条本地插入一批，不调用任何模型。
3. 写入进程退出后，另一进程重新打开数据库，核验实际schema/索引、全部主键、正文、完整元数据与向量指纹。
4. 用同一个已有文档向量，对照NumPy和Milvus全部余弦分数，并检查知识库过滤；不编码问题，不把该检查当作问句检索质量测试。
5. 验收通过才发布 `manifest.json` 的 `ready` 状态。它只表示本次存储验证通过，不代表官方文档已经人工审核、版本适用于现场、或问答准确率达标；原有 `review_status`、`ready_for_embedding` 等来源标记保留原样。

默认产物在 `data/milvus_indexes/kubernetes_zh_v1/`：`build_plan.json` 记录输入合同与指纹；`model_observation.json` 记录一次模型身份观测；`knowledge.db/` 是当前Milvus Lite版本的数据目录（尽管后缀为.db）；`build_result.json`、`verification.json` 是写入与读回回执；`manifest.json` 是成功验收后的索引清单。

当前156块若重编码，批量8时需要20次文档Embedding；本次复用缓存时文档/问题Embedding、Reranker、生成模型均为0次，仅有1次模型元数据HTTP，另有Milvus本地操作。一次digest观测不能证明未来模型不变，后续问题编码还需检查合同。导入不读取.env、不回退云端、不修改旧铁路索引，也不改变原语料包。

包括失败现场在内的已有目标目录一律不覆盖、不追加；新索引也不能位于原缓存或冻结语料包内部。失败时保留现场且不自动重试；完成缓存仍可复用，确认原因后用 `--index-dir` 指向另一个新目录重建，不必重编码。当前实现面向这份小子集，构建与读回会全量加载；不是全1665份资料的大规模导入/检索性能方案。

### 用真实问题对照 NumPy 与 Milvus 检索

```bash
python -B scripts/check_k8s_search.py
python -B scripts/check_k8s_search.py --run \
  --query "Pod里访问服务域名时DNS解析失败，应该如何排查？" --top-k 3
```

不带 `--run` 时，仅核对已有ready清单、构建/验收回执、冻结语料和SQLite缓存，规划过滤范围与请求；不导入Milvus SDK、不创建HTTP或数据库客户端、不写文件。该教学对照依赖原语料包和SQLite缓存，不是仅依靠Milvus就能独立部署的生产查询服务。

真实执行按下列顺序进行：

1. 打开**已有**Milvus数据库，核验实际schema、主键、4096维、FLAT/COSINE及全部原文、完整元数据、向量指纹；与SQLite缓存向量逐项比对。库缺失或不一致时停止，不自动重建。
2. 从同一个内网Ollama服务读取模型digest；匹配后只编码**一个问题**，再读取一次digest。正常成功共1次推理、2次元数据GET，即模型服务HTTP3次，无文档编码、重试、云端回退、重排或聊天。Milvus本地RPC另计。
3. 问题输入按建库合同组织为 `Instruct: {任务描述}\nQuery: {原始问题}`；这是一次Embedding请求中的输入，不是额外的聊天调用。文档输入仍为原正文，不给文档追加这段查询指令。请求保持 `truncate: false`。
4. NumPy在内存计算SQLite缓存向量的余弦相似度，Milvus搜索独立已建索引；两路共享同一个问题向量、文档向量值、Top-K及知识库/上游commit过滤条件。可用 `--parent-id` 限定同一个父文档，两路同步缩小范围，不能只过滤其中一路。
5. 终端并排显示块ID和分数，再展示Milvus命中的标题、章节、完整来源URL、版本/审核状态与正文摘录。`--preview-chars` 仅限制终端展示长度，不改变编码、入库或检索正文；摘录中的命令不会被执行。

当前参数：`--top-k` 1～50（超出候选数量时取实际数量）、`--repeat` 1～20（默认5）、`--preview-chars` 100～4000（默认600）、`--timeout` 1～600秒。问题最多2000字符，这是客户端防误用限制，不代表Token上限。未知父文档会在请求前被拒绝。

阅读结果要分开回答两个问题：

- **数值是否一致？** 按chunk_id对齐分数，并检查Top-K边界容差。近同分候选的次序或边界成员可能不同，不把合理同分差异直接判为故障。
- **内容是否相关？** 阅读原文，判断是否回答了你的问题、是否缺少上下文、版本是否适用。排名第一、两路重合或相关性分数较高，都不能直接当作答案正确率；没有人工标注评测集，本步不报告检索准确率。

无关问题也可能返回最相似的K条；本步没有设置语义相关性阈值或拒答机制。“Top-K边界校验”只验证计算实现，不是在证明证据充分。全量读回156块是本教学对照的核验成本，不是Milvus生产检索每次必须执行的流程。

计时单列SDK导入/打开、集合读回核验、问题请求及向量校验、两路搜索首轮和后续中位数。模型版本GET、计划阶段/语料/SQLite核验不混入所列问题编码或搜索时间。NumPy先测、Milvus后测；NumPy含矩阵计算/向量检查，Milvus含本地RPC/结果重建/命中检查，且此前已经全量读回过数据。因此不是冷启动、等工作量底层内核或大规模性能基准，不预设Milvus更快。

检索工具不执行插入、更新、删除，不写报告；但本地数据库运行时可能更新锁或日志。查询前后digest观测一致不能独立证明每次推理使用的权重，也不能代替后续检索质量评测。

### 原始块与相邻上下文补全对照

上一轮DNS问题中，第3名是Service文档的第8号块，第1名是同章节第9号块：命令和输出落在两个相邻块中。本步不重切、不重建索引，而是在检索后组织已有正文；不是整篇父文档召回，也不是重排。

先复现这三个已知块，固定起点，只改变上下文组织方式：

```bash
python -B scripts/check_k8s_context.py \
  --seed-chunk a78c05ca1bacb446 \
  --seed-chunk 8b4e4d29bd6d2413 \
  --seed-chunk 4fe4ff99d7c2e8c4
```

不加 `--run` 时只显示本地规则计划。给同一条命令加 `--run`，终端展示A原始块、邻块取舍账本、B合并后的上下文和长度对照。手工模式只读现有ready清单、冻结语料和SQLite缓存，**模型HTTP=0，不导入Milvus SDK或打开数据库**；输入顺序由用户指定，不称为本次检索排名，不填造历史相似度分数。

默认 `--window 1` 是前后各看一个原始chunk_index，`--max-context-chars 6000` 是正文预算，`--preview-chars 2400` 只限制每块/每段在终端的展示长度。这三个起点按默认规则应从3块、2217字符变成5个独特块、2段连续上下文、2949字符；Service原第8/9块接上第10块，DNS第4块接上第5块。结果依赖固定语料，不能推广为其他问题的召回质量结论。

补全规则：

- 先保留全部原始命中。原命中区间去重后仍超过预算就停止并报错，不能悄悄丢证据或截断代码。
- 仅在调用方已经过滤的候选范围内，按父文档和原始 `chunk_index ± 距离` 找邻块。缺少中间块就停止该方向，不越过过滤缺口找下一条。
- 来源、上游commit、清洗/切块版本、审核标记与完整章节路径必须一致。不同章节保守跳过；可能遗漏必要前提，但不猜测其适用性。
- 先近后远；相同距离按原始输入/排名顺序，再按前、后顺序使用预算。预算不足时整块跳过并停止该方向，不宣称这是最优分配。
- 同一邻块只加入一次；连续区间按源位置合并，重叠正文逐字核对后去重。存在字符gap就分成不同片段，不能伪装成一段连续原文。
- 新邻块没有单独检索相关分数；保留初始命中的排名/分数，不为合并上下文伪造一个新分数。扩展段有自己的字符区间和组成块ID，不继承首块的旧坐标冒充新身份。

可以在同一手工命令中加 `--window 0`，观察只做源位置整理/区间去重而不补邻块的结果；或降低正文预算，观察哪些整块被拒绝。字符不是Token，正文预算也不含来源、标题、提示词等包装，不能据此宣布满足模型上下文上限。

需要新问题时，不带 `--seed-chunk`：

```bash
python -B scripts/check_k8s_context.py --run \
  --query "Pod里访问服务域名时DNS解析失败，应该如何排查？" --top-k 3
```

该模式复用已有校验、问题编码和搜索函数，正常1次问题Embedding加前后2次digest GET，共3次模型服务HTTP；NumPy/Milvus各搜索1次。A/B共享这一轮的Milvus命中，补全从本地已核验记录取邻块，额外模型或数据库取块请求为0。真实模式打开数据库可能更新运行时锁/日志，但不修改记录；本步不写报告、不调用重排或生成模型。

补全不改变初始Top-K与排名，但最终阅读上下文包含额外邻块；邻块可能不相关，固定窗口也不能保证排查逻辑完整。是否更有帮助仍需阅读对照，而不是只看“字数增加”。

### 在浏览器中逐步完成检索、重排与证据回答

```bash
python scripts/serve_retrieval_lab.py
```

浏览器访问 `http://127.0.0.1:8766/`，与原清洗台8765端口分开。此界面按本地模式制作，不注册或发布外网网站，也不自动开启浏览器。

1. 打开页面只核验已有清单与缓存，不打开Milvus、不读取.env、不调用模型。
2. 点击“仅查BM25”：在同一份156块正文上分词、建内存BM25并排序，模型请求为0。
3. 点击“三路对照”：BM25与Dense各召回最多8块，同一问题仅编码1次，另有2次模型digest GET；文档编码、Reranker和聊天均为0。Dense复用已有Milvus与数值验算，然后本地计算等权RRF。初始每列展示前5条，额外召回可以展开，折叠不影响融合。
4. 点击任一块，对照它在各列中的名次，并阅读章节、来源、审核状态与完整原文；RRF卡片列出两路的 `1/(60+排名)` 贡献，缺失一路记0，不直接相加BM25和余弦分数。
5. 完成“三路对照”后，可主动运行云端重排。页面先提示：把当前问题和两路召回**完整并集的正文**发送到 `qwen3.7-text-rerank`，可能产生费用。不仅发送RRF前5；每路最多8块，所以并集最多16块，本题可能是13块而不是固定13块。云端不接收来源路径、元数据或向量。
6. 在新增的重排对照中，比较同一候选的完整RRF名次与重排名次，观察新进／离开前5、上升／下降，再点击阅读原文。分数不是概率，两类原始分数不混加；名次变化不等于质量提升。此阶段仍不补上下文、不生成答案。
7. 重排完成后，在“依据证据生成回答”区域预览实际选定的前5块（不足5块取实际数量），编号S1～S5。模型选择、服务地址和传输方式单独显示；确认发送范围后再生成。更换模型需重新确认，既有检索与重排不重新执行。
8. 回答模型只处理问题、选定正文、程序提取的原文片段编号与必要的测试／审核标记，不发送本机路径或向量。输出先收集完整，再进行JSON结构、状态规则、来源编号和原文摘录定位校验；页面展示最多3条回答要点、逐条引用、来源和待补充信息。`answered`只表示模型认为可据资料作答，**不是程序确认事实正确**。证据不足时允许部分回答或拒答。

本次回答阶段只增加1次聊天HTTP，不增加Embedding或Reranker；正常无失败重试的一轮累计为Embedding 1、HTTP 5、Reranker 1、聊天1。回答使用既有流式接口，但不实时展示未经校验的正文，不展示推理过程；输出截断、JSON无效或引用不存在均停止展示。引用检查不证明语义支持、步骤完整、版本适用性或示例值是真实现场值；仍需人工核对。

服务仅在用户主动生成时读取对应认证；千问读取已有环境变量或 `models/.env`，GLM只使用独立占位值，绝不把千问密钥交给GLM端点。禁用环境代理、重定向和SDK自动重试。同一快照一次成功生成后，相同模型重复提交复用结果；本轮不再切换模型另收费，更换模型需显式重开一轮。回答失败保留检索、重排与证据，允许主动重试；失败请求可能已发生并收费，计数只涵盖完成阶段，不以失败当作0请求。新检索或服务重启会使旧快照失效。

重排只复用服务内存中最近一次混合检索的冻结快照；浏览器提交的是不透明的 `experiment_id`，不能另传问题、候选正文、服务地址或模型参数。正常完成只增加1次云端HTTP，问题／文档Embedding都不增加；本轮完成阶段累计为问题Embedding 1、HTTP 4（原本地模型HTTP 3＋云端重排1）、Reranker 1、聊天0。全池只发一批，不是一块一次请求。仅在主动重排时，后端读取已有环境变量或 `models/.env` 的 `DASHSCOPE_API_KEY`；密钥不返回浏览器、不写报告。禁用环境代理和重定向，不自动重试。

同一快照重排成功后，重复提交复用该结果，不再次请求云端；新的检索（包括仅BM25）会使旧快照失效。服务重启后快照也失效，需要重新检索。重排失败时保留原三路候选，不显示未校验的重排结果；已发出的失败请求可能仍收费，页面保留的计数只覆盖已完成阶段，主动重试会另发请求。服务仅在内存中保留最近一轮，不提供多用户会话或历史结果存储。

页面的“分享本轮结果”提供两个按钮，均只使用浏览器已经收到的结果，不重新检索、不编码问题或文档、不请求重排/聊天，也不写入服务器报告目录：

- **复制实验摘要**：复制问题、参数、实际请求计数、三路完整候选并集排名表、RRF贡献与来源/快照/审核状态。折叠的第6～8名也包含在内，但摘要不附候选全文；如果浏览器拒绝剪贴板访问，会展示只读文本框，按`⌘C`或`Ctrl+C`手动复制即可。
- **导出完整JSON**：通过浏览器下载一个`k8s-retrieval-<时间>.json`文件，保存未截断的候选正文、原始分数、完整来源元数据、当前选中块，以及浏览器收到结果和导出的时间。结果时间不是服务器的推理完成时间；JSON不是已标注评测结论，也不能替代模型/索引构建清单。文件可直接作为附件用于分析。

重排完成后，摘要还包括完整候选池的重排名次、升降与新进前5；JSON保留重排分数、输入位置和完整映射。回答完成后还包含模型、证据编号、结构化回答、逐条原文引用和待补充信息；完整JSON保存未截断的选定正文、片段字典和来源元数据。没有结果、正在执行、检索失败或修改问题后，分享按钮不可用；重排／回答失败仍可分享本轮已完成阶段，并标明未知失败请求，不能冒充后续阶段成功。不会把上一轮结果误当成新问题的结果。页面仍只保存当前一轮内存状态，刷新、换题前请先导出。下载需要浏览器允许，JSON可能包含问题、资料正文和本机路径，分享前请检查敏感信息；功能不会额外读取密钥、向量或环境文件。生成的JSON只存在浏览器下载位置，不自动提交Git。

先固定DNS问题看三路差异，再改写问题观察候选变化。重点问：“找到了什么？为什么排在这里？原文是否真正有用？”而不只比较分数。没有标注集，不报告准确率；RRF不保证更准，也无法找回两路均未召回的块。

本步沿用已有分词基线：中文jieba、英文小写、保留部分连字符编号；域名、IP和路径会被拆词。BM25只索引正文，不额外拼入标题，统计基于同一候选范围。标点或停用词过滤后为空时，BM25无候选；Dense仍可能返回最相似但不相关的块。三种分数尺度不同，不能横向比较数值大小。

新增文件的职责：

| 文件 | 输入 → 输出／职责 |
|---|---|
| `retrieval/kubernetes_hybrid.py` | 已核验块＋问题 → 三路候选、命中词、贡献；复用已有算法，不重切或重建库 |
| `retrieval/kubernetes_rerank.py` | 冻结的完整候选并集 → 一次云端打分、原输入位置映射、独立重排名次；不改变正文或原始排名 |
| `rag/kubernetes_answer.py` | 冻结的重排前5块 → S编号、单次云端回答、已校验回答及本地来源；不重跑检索或重排 |
| `rag/answering.py` | 复用严格回答结构、流式收集与引用校验；纯正文模式保留首段，旧铁路标题规则保持不变 |
| `retrieval_lab.py` | 本机页面按钮 → 严格校验请求、串行执行、返回JSON；不接受用户指定文件路径或服务地址 |
| `web/retrieval_lab.html` | JSON → 三列卡片、同块高亮和来源原文；所有资料按文字展示，不执行文档命令 |
| `scripts/serve_retrieval_lab.py` | 仅启动入口，将运行交给包内后端 |
| `tests/test_kubernetes_hybrid.py` | 合成资料和假客户端 → 零网络回归，检查一次编码、范围、融合、失败关闭和API边界 |
| `tests/test_kubernetes_rerank.py` | 合成候选／假云端响应 → 全池覆盖、分数与身份校验、快照隔离、成功复用及API安全边界 |
| `tests/test_kubernetes_answer.py` | 合成证据／假聊天流 → 前5范围、引用校验、拒答／截断、模型认证隔离、成功复用和失败保留 |
| `web/retrieval_export.js` | 已完成的JSON结果 → 深拷贝报告、可复制摘要和安全文件名；不接触网络、数据库或剪贴板 |
| `tests/retrieval_export.test.cjs` | 合成数据／浏览器替身 → 报告完整性、复制回退与页面状态的Node离线回归 |
| `tests/test_retrieval_export.py` | 将Node回归纳入pytest，并验证静态脚本不触发检索；没有Node时明确跳过JS测试 |

Python实现路径相对于 `src/qa_platform/`。检索、重排、回答分步进行，暂不混入上下文补全。服务仅监听127.0.0.1，要求同源JSON和自定义请求头，无CORS；同一时刻只允许一轮检索、重排或生成，不自动重试、不持久保存问题或报告。真实混合模式可能更新Milvus运行锁/日志，但不插入、修改或删除记录。检索中途失败时不展示部分或旧结果；重排／回答失败保留本轮已完成阶段，已发出的模型请求也不会被当作未发生。

### 校验现有 Kubernetes v2 语料

```bash
python -B scripts/prepare_k8s_corpus.py \
  --verify data/k8s_corpora/troubleshooting_v2
```

### 打开文档清洗与切块对照台

```bash
python -B scripts/serve_data_lab.py
```

浏览器访问 <http://127.0.0.1:8765/>。界面使用既有清洗、切块和块清理函数，不在前端另造一套规则，不调用模型。

### 运行离线回归测试

```bash
python -B -m pytest
```

测试不读取真实密钥、不请求模型、不打开项目已有Milvus库。依赖匹配时，集成用例会用真实Milvus Lite运行临时合成数据库；需要文件写入的用例只使用临时目录。本地已有知识包时会增加历史兼容性检查；没有这些生成数据时，对应测试跳过。

各模块仍保留原来的 `--self-test` 能力，例如：

```bash
python -B scripts/check_k8s_documents.py --self-test
python -B scripts/serve_data_lab.py --self-test
```

注意：旧聊天、检索、RAG脚本的默认行为不一定是离线预检。只是了解参数时请用 `--help`，不要将所有脚本无参数批量执行。

## 原入口与实现位置

下表说明本次迁移后的每份实现负责什么。`scripts/` 中保留同名入口，原有运行命令不必修改。

| 原入口 | 当前实现 | 职责 |
|---|---|---|
| `check_chat.py` | `providers/chat.py` | 流式聊天与模型选项 |
| `check_embedding.py` | `providers/cloud_embedding.py` | 原云端Embedding演示 |
| `check_reranker.py` | `providers/cloud_reranker.py` | 原云端重排演示 |
| `check_local_embedding.py` | `providers/local_embedding.py` | 本地Embedding小样本验证 |
| `build_k8s_embeddings.py` | `storage/embedding_cache.py` | 本地文档编码、事务缓存与断点续跑 |
| `build_k8s_milvus.py` | `storage/kubernetes_milvus.py` | 复用已完成缓存、独立Milvus索引及跨进程验收 |
| `check_k8s_search.py` | `storage/kubernetes_search.py` | 真实问题共享编码、NumPy/Milvus过滤检索及数值/计时对照 |
| `check_k8s_context.py` | `retrieval/context_expansion.py` | 原始块与同源同章节邻块补全、区间去重及正文预算对照 |
| `check_chunking.py` | `ingestion/railway.py` | 铁路资料加载、父记录和子块 |
| `check_k8s_documents.py` | `ingestion/kubernetes_cleaning.py` | Markdown与已知Hugo语法清洗 |
| `check_k8s_chunking.py` | `ingestion/kubernetes_chunking.py` | A固定窗口／B结构切块对照 |
| `check_k8s_chunk_cleanup.py` | `ingestion/kubernetes_cleanup.py` | B到C的块清理及去向账本 |
| `prepare_k8s_corpus.py` | `ingestion/kubernetes_corpus.py` | 语料导出、校验及历史版本分派 |
| `check_k8s_embedding_plan.py` | `ingestion/embedding_plan.py` | 编码输入、指纹与批次离线计划 |
| `check_bm25.py` | `retrieval/bm25.py` | 中文分词与关键词召回 |
| `check_dense.py` | `retrieval/dense.py` | 原内存向量召回 |
| `check_hybrid.py` | `retrieval/hybrid.py` | BM25与Dense的RRF融合 |
| `check_retrieval.py` | `retrieval/pipeline.py` | 召回、融合、重排整链路 |
| `check_milvus.py` | `storage/milvus_smoke.py` | 合成向量、跨进程持久化与过滤 |
| `build_milvus.py` | `storage/milvus_build.py` | 原测试库向量入库与读回验收 |
| `check_milvus_search.py` | `storage/milvus_search.py` | Milvus与NumPy检索对照 |
| `check_rag.py` | `rag/answering.py` | 证据回答、结构与引用校验 |
| `check_eval.py` | `rag/evaluation.py` | 原测试库四题开发回归与报告 |
| `check_milvus_rag.py` | `rag/milvus_demo.py` | Milvus接入RAG后的两路对照 |
| `serve_data_lab.py` | `web_app.py` | 可视化界面的只读后端 |
| `serve_retrieval_lab.py` | `retrieval_lab.py` | BM25／Dense／RRF的本地可视化对照 |

实现位置均相对于 `src/qa_platform/`。`settings.py` 只计算统一路径；各目录的 `__init__.py` 声明Python包，不初始化客户端；`ingestion/legacy_verifier.py` 处理旧语料兼容；`tests/test_repository_layout.py` 检查目录与导入边界，`tests/test_embedding_cache.py` 用假模型和临时SQLite验证缓存、续跑及失败处理。`tests/test_kubernetes_milvus.py` 检查缓存导入、单次digest请求及失败发布保护，并在依赖匹配时用临时合成向量执行真实Milvus跨进程测试；不访问实际模型服务。

`tests/test_kubernetes_search.py` 用假问题向量检查查询输入合同、模型版本变化、相同过滤、近同分Top-K与异常关闭；另用3条4维临时Milvus数据跑通两路检索，不访问实际模型或正式索引。

`tests/test_context_expansion.py` 用合成文本检查邻块边界、连续区间、重叠冲突、原命中保留、整块预算、无副作用及离线示例；不请求模型或打开正式数据库。

## 数据与历史规则兼容

原始 Kubernetes 快照位于 `knowledge/kubernetes_zh/upstream/`，来源信息在 [source_manifest.json](knowledge/kubernetes_zh/source_manifest.json)。其中有 1665 个中文 Markdown 文件，包含导航、参考和待筛选内容，不能将文件数视为有效知识文章数。

| 路径 | 内容与当前状态 |
|---|---|
| `data/k8s_corpora/troubleshooting_v1/` | 历史语料包，保留原样；其更早的规则版本不在本次保存的校验快照中 |
| `data/k8s_corpora/troubleshooting_v2/` | 当前编码输入：11篇、156块，状态为部分范围准备完成 |
| `data/milvus_indexes/test_v1/` | 原铁路虚构测试库：8块、1024维，不可混入新的4096维向量 |
| `data/milvus_smoke/` | 既有4维合成向量持久化实验 |
| `data/embedding_cache/` | 执行编码工具后才创建的文档向量检查点；不是Milvus索引 |
| `data/milvus_indexes/kubernetes_zh_v1/` | 新入库工具的默认目标；执行并验收成功后才是可用的独立4096维索引 |

上述目录及 `knowledge/` 的原始资料均未移动或重写。原铁路索引记录了知识文件的绝对路径，所以知识文件也保持原位置。

v2 的清单记录了迁移前四份规则源码的完整 SHA256。直接更改源码路径和导入后，不能拿新文件去假装满足旧清单。为此：

1. 原四份源码逐字节保存在 `ingestion/_snapshots/k8s_prepare_v1/`，不格式化、不修改、不作为当前业务入口。
2. `legacy_verifier.py` 先核对这四份快照的固定指纹，再在独立子进程中调用原校验函数；该校验流程不读取上游、不调用模型、不改写语料。
3. 旧v2仍走原有严格校验，没有更新旧manifest、跳过哈希或将旧包重新标记为ready。
4. 当前构建器版本为 `k8s-prepare-v2-src-layout`，今后新导出的默认目录改为 `troubleshooting_v3`，与旧包分开。目录整理本身没有生成v3。
5. v1缺少对应的更早历史校验器；仍保留数据，不把它原先就不兼容的状态误报为本次迁移损坏。

冻结快照是版本兼容资产，不是需要继续修改的第二份业务代码。后续功能修改应发生在当前包模块；新规则导出新版本数据。

## Git管理与公开仓库边界

远程仓库为 `https://github.com/Patton0829/fault_diagnosis.git`，远程名称 `origin`。提交前先检查 `git status` 和 `git diff --cached`，每个完成并测试过的学习阶段单独提交；需要功能分支时使用 `codex/` 前缀，不强制推送或覆盖远程历史。提交身份采用仓库本地配置，不修改全局设置。

公开范围主要为源码、测试、运行入口、前端、来源清单与脱敏说明。`.env`、原始接口资料、简历、编辑器本机配置、学习截图与笔记、评测报告、上游文档快照、SQLite缓存、Milvus和生成语料均被忽略，只保留在本机，**没有删除它们**。以后要分享某份笔记或图片，单独检查内容后再调整规则。

公开克隆不会自带私有服务配置或已经生成的数据，不能直接打开当前156块索引。需准备本机配置、按来源清单获取上游、导出新语料并重新建缓存/索引；现有索引还绑定路径和规则指纹，不是可随意复制的便携产物。CI可先跑合成离线测试；依赖本机数据的既有测试可能跳过。

推送需要用户本机GitHub认证，例如由用户执行 `gh auth login` 完成浏览器登录。不要把token发到聊天或提交文件中。远程已连接不代表提交/推送已完成，实际以 `git log` 与 `git status -sb` 为准。

## 学习资料与下一步

本机 `docs/learning/` 留有BM25和Milvus笔记；`docs/reference/` 留有个人参考资料，不包含在公开仓库中。
公开配置参考 [模型接口说明](docs/model_apis/README.md)。

当前学习路线：模型调用 → 文档处理 → BM25／Dense → RRF → Reranker → 最小RAG → 开发回归 → Milvus对照 → Kubernetes语料与可视化 → 本地Embedding小样本 → 编码缓存续跑 → 缓存向量独立入库 → 真实问句检索对照 → 相邻块上下文补全。

当前先在页面上比较BM25、Dense与RRF的候选和依据，确认理解后再接云端重排与回答。Kubernetes端到端RAG、跨版本增量更新、LangGraph编排和正式应用界面仍尚未完成。

学习过程中，每增加一份实现文件都说明其输入、输出、职责和调用关系；对比实验应固定控制条件，不将“能运行”“排序一致”或“自动检查通过”直接当作问答正确率。
