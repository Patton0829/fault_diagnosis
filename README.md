# 知识库智能问答平台

本项目通过可运行的对照实验，逐步学习文档清洗、切块、Embedding、BM25、RRF、Reranker、Milvus 和带来源引用的 RAG。最初使用铁路故障诊断测试资料，当前正在接入 Kubernetes 中文文档，构建独立的 IT 运维知识库。

目前，铁路测试库已跑通最小 RAG 与 Milvus 对照链路；Kubernetes 当前11篇、156块已完成本地编码与独立 Milvus 入库。**`/qa` 是最小问答入口：一次主动提交，由LangGraph自动完成真实混合检索、云端重排和带引用回答，并支持最近3轮短期追问；打开页面0模型请求。** 原 `/` 检索实验台保留分步点击和排序对照。两者都不混入相邻块上下文补全；追问仅补全问题指代。尚无标注质量评测，不把运行成功等同于答案正确。

另有 `/workflow` 线性对照页、`/workflow/branching` 条件分支页与 `/workflow/approval` 人工确认页，均是独立的合成教学实验，不是真实问答入口。真实问答默认使用三个顺序节点；可选“多步检索”进入最多两次的真实工具循环。短期会话在API层管理，不加入审批、持久化检查点或自动重试。

## 实习项目复现的总体进度

最终目标仍是复现实习项目的技术链路，**单轮RAG通过基础验收不代表完整实习项目已完成**。按最初方案继续推进：

| 阶段 | 当前状态 |
| --- | --- |
| ① 文档处理与检索 | 已有清洗、切块、Embedding、Milvus和混合检索；多格式解析、跨版本增量更新及正式评测仍待补 |
| ② 单轮RAG | 已有真实检索、重排、结构化回答、来源引用、拒答及简单页面 |
| ③ LangGraph辅助诊断 | 最小主流程已有真实单轮图、原生工具循环及最近3轮短期追问；更多诊断工具、澄清策略与持久记忆仍待补 |
| ④ 平台工程、部署与评测 | 当前推进：已有FastAPI与SSE阶段进度；MySQL/SQLAlchemy已完成代码、离线测试及本机初始化，服务已切换，待真实问答刷新恢复验收；Redis、vLLM、Docker Compose及系统化质量评测仍待补 |

保持简单页面，逐步补齐核心能力；教学分支/审批回放不能当作真实Agent已经完成。当前使用Kubernetes文档复现技术路线，不把它表述为真实铁路故障知识或已部署的现场诊断系统。

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
│       ├── storage/               # 编码缓存、Milvus、MySQL配置和SQLAlchemy问答存储
│       ├── rag/                   # 回答、引用校验、评测和Milvus对照
│       ├── workflows/             # live_rag.py单轮；agent_rag.py工具循环；其余为合成教学
│       ├── web_app.py             # 文档处理可视化的后端
│       ├── web_stream.py          # 工作线程→SSE进度桥接，不转发模型Token
│       └── retrieval_lab.py       # 独立检索学习台，不改变原清洗台的零模型约定
├── scripts/                      # 运行入口，不承载业务实现
├── tests/                        # 目录/语料兼容、缓存与入库的离线测试
├── docs/
│   ├── learning/                 # 本机学习笔记，暂不上传
│   ├── model_apis/               # 脱敏模板；原接口说明只保留在本机
│   └── reference/                # 本机个人参考资料，不上传
├── models/                       # 模型.env及本机database.env；不是模型权重目录
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

### 最小问答入口：一次提问完成真实RAG

```bash
/Users/patton/miniconda3/envs/langchain/bin/python -B scripts/serve_retrieval_lab.py --port 8766
```

打开 <http://127.0.0.1:8766/qa>，输入问题、选择回答模型，阅读服务目的地址和费用提示后点击“提问”。原检索实验台仍在 `/`，不需要先运行它。现有Kubernetes索引、本地Embedding服务和云端认证必须已就绪；不会自动编码文档或重建索引。

一次请求执行：`START → retrieve → rerank → answer → END`。

- `retrieve`：复用 `kubernetes_hybrid.prepare/compare`，完成一次问题编码、Milvus/BM25召回和RRF；不重新编码文档。
- `rerank`：复用 `kubernetes_rerank.rerank`，把两路完整候选并集提交云端，不先截取RRF前5。
- `answer`：复用 `kubernetes_answer.generate`，选择重排前5、准备段落引用、调用所选模型，再进行JSON结构及引用定位校验。不是直接展示未经校验的流式正文。

正常完成为Embedding、Reranker、聊天各1次，另有2次模型身份GET，共5次模型服务HTTP；本机页面API另计。页面在提交前说明发送的数据和目的地；选择GLM时另显示其HTTP明文传输风险。每次主动提问都是新的一轮，可能收费，不自动重试。任一步失败即停止，页面不保留旧答案冒充新结果，也不把异常显示成正常拒答；证据不足由回答结果明确表达。

核心源码按以下顺序阅读：

1. `src/qa_platform/workflows/live_rag.py`：先读`LiveRAGState`（问题、模型、本轮结果），再读`build_graph`内三个节点和连线，最后读`run`。`run`用实际`stream`事件记录完成步骤；同名`result`字段逐步覆盖为新的结果。关闭LangSmith追踪，无checkpointer、循环或重试；只返回已校验的答案及实际引用来源，不返回本机路径、向量或完整候选池。
2. `src/qa_platform/retrieval_lab.py`：`GET /qa`提供页面，`GET /api/qa/plan`只提供固定模型配置；`POST /api/qa`接受严格的`query/model/confirmed`及可选`mode`（默认`single`），沿用同源JSON、自定义头、16KiB和全服务忙碌锁，再调用真实图。不接收浏览器上传的正文、密钥、路径或服务地址，不改原实验台快照。
3. `web/qa.html`：读取配置、展示发送范围，主动提交一次请求，用纯文本展示答案、引用与来源。仅增加新对话按钮及折叠的短期历史，没有对照台、图编辑器或人工审批；`web/retrieval_lab.html`只新增入口链接。
4. `tests/test_live_rag.py`验证真实LangGraph与既有业务函数的连接和API边界；`tests/qa.test.cjs`验证前端状态及安全展示，`tests/test_qa_ui.py`将其纳入pytest。测试使用合成数据和替身，不产生真实模型费用。

这是单机、当前11篇资料的小规模应用；默认短期会话仍在内存，MySQL模式须另行显式启用，不等于生产部署。节点运行成功和引用存在都不能证明事实正确、排查完整或版本适用。

### 最小验收记录（2026-10-20）

本轮只补验收，不增加业务功能。验证代码版本为`f9f3dfd`：

| 检查 | 方式 | 结果与边界 |
| --- | --- | --- |
| DNS正常问答 | 复用2026-10-19的真实网页运行 | 三个节点完成，39.80秒，返回带原文引用的`partial`回答；本轮未重新调用，不能当作新的质量样本 |
| 明显域外问题 | 本轮直接执行真实`live_rag.run`，问“英国的首都是什么？” | 9.616秒，`insufficient_evidence`、空claims、空sources；missing_info说明Kubernetes资料不覆盖该问题，没有泄露“伦敦/London”答案 |
| 检索、重排或回答失败 | 离线替身注入异常 | 后续节点停止、返回脱敏错误，不自动重试，不把失败伪装成证据不足 |
| 引用错误与界面旧答案 | 离线后端/前端测试 | 未知来源、未知片段和不在原文中的摘录被拒绝；失败不展示旧答案 |

指定Conda环境执行`python -B -m pytest -q`：632项通过，4项既有依赖警告。本轮真实模型新增Embedding、Reranker、聊天各1次，含2次身份GET共5次HTTP；离线测试不调用模型。完整本机记录在`reports/qa-acceptance-2026-10-20.json`，不提交Git。浏览器访问本机页面受到策略限制，本轮没有新的浏览器实测或截图，不能把离线前端测试说成浏览器实测。

至此最小单轮闭环及上述基础验收已完成。没有统计答案正确率，也没有证明所有域外问题都会拒答、每个推断都被引用充分支持，或真实服务中断后的恢复能力；正式评测、生产部署和多轮会话仍不在本轮范围内。

### 第3阶段起步：验证原生Tool Calling

这一步不修改`/qa`。先验证模型服务协议，再接入真实检索工具，避免把普通JSON动作误称为原生工具调用。

```bash
# 仅看计划：0请求，不读.env、不创建客户端
python -B scripts/check_tool_calling.py

# 使用既有千问Flash服务验证：正常2次聊天请求
python -B scripts/check_tool_calling.py --run
```

完整流程为：工具定义和合成问题 → 模型返回标准`tool_calls` → 程序校验并执行本地工具 → 使用真实`tool_call_id`回传结果 → 模型回答。格式参考[OpenAI Docs函数调用说明](https://developers.openai.com/api/docs/guides/function-calling?api-mode=chat)；第三方兼容接口是否支持必须实测，不能仅凭文档或SDK推断。

本地唯一工具`lookup_demo_record(code)`只接受固定合成校验码`TC-DEMO-7319`。第一次请求不包含答案，随机`verification_marker`在本地函数真正执行时才生成。最终回复必须准确返回该随机串，从而区分“只返回工具格式”和“实际读取了回传结果”。它不是查询真实知识库，也不是设备维护工具。

核心源码阅读顺序：

1. `src/qa_platform/providers/tool_calling.py`中的`LookupArgs`与`TOOLS`：声明参数结构和工具说明；定义工具不等于已经执行函数。
2. `lookup_demo_record`：程序控制的本地函数；没有模型请求、文件读取、数据库连接或命令执行。
3. `probe`：第一次`tool_choice="auto"`允许模型提出调用；只有标准字段、唯一调用、白名单名称、合法参数及固定校验码全部通过后才执行工具。随后追加`assistant.tool_calls`和`role="tool"/tool_call_id`消息，第二次`tool_choice="none"`要求结束，不扩展成无界工具循环。
4. `main`：默认只看计划，`--run`才读取所选服务认证并创建客户端。此探针使用非流式响应，千问单独设置`enable_thinking=False`，不修改原聊天/RAG默认配置。禁用环境代理、重定向和SDK自动重试，并显式清空OpenAI组织/项目标识，避免继承后发送给第三方服务。GLM仅用占位认证，不读取或复用千问密钥。
5. `scripts/check_tool_calling.py`只负责定位`src/`并启动`main`；`tests/test_tool_calling.py`用真实SDK与内存HTTP替身验证协议、失败停止及认证隔离，不产生真实模型费用。

2026-10-20，既有`qwen3.8-flash`服务的本次合成验证通过：结束原因依次为`tool_calls`和`stop`，本地生成与最终返回的随机串一致；共2次聊天HTTP，10.18秒，无重试。未真实验证Max或GLM。`--model`可选择既有模型；GLM目前是HTTP明文，仅应发送合成资料。

该探针的结论仅适用于所选模型、地址、参数和本次样例；它本身未验证流式工具、复杂参数或真实RAG。失败时仅记录固定阶段和请求尝试数，不展示未经校验的正文，也不据此断言服务一定不支持。真实知识工具循环见下一节。

### 第3阶段：真实知识工具与有上限的多步检索

在同一 `/qa` 页面选择 `qwen3.8-flash`，勾选默认未选中的“多步检索（最多2次）”后提问。不勾选仍走原来的三个顺序节点，Max/GLM也仍可用于原单轮模式；目前只有Flash经过原生工具协议实测，因此多步暂不开放其他模型。

这次变化不是把固定流程再包一次：`plan`节点调用模型，模型通过标准`tool_calls`提出`search_knowledge(query)`；`search`节点校验名称与参数，执行既有混合检索和重排，再用对应`tool_call_id`返回工具消息。模型读取第一次结果后可决定再查不同的问题，或停止。路线为：

```text
START → plan ──提出合法工具调用──→ search
          ↑                        │
          └────已检索1次，再决策────┘
          │                        │
          └─停止──→ answer ←─已检索2次，到达上限
                      ↓
                     END
```

第二次检索后直接进入独立的引用回答节点，不再调用第三次决策。规划器普通文字从不当作答案展示，最终生成只接收原问题及合并证据。两轮证据按各轮名次轮流取、按块ID去重，最多5块；不同子问题的重排分数不混排。两轮必须保持相同正文、向量与Embedding合同，不重新编码文档、不补相邻上下文。此选择策略并不保证最优，也不等于回答已涵盖所有子问题。

| 实际路径 | 问题Embedding | 重排 | 聊天（决策+最终生成） | 模型服务HTTP |
| --- | --- | --- | --- | --- |
| 模型不调用工具，程序本地证据不足拒答 | 0 | 0 | 1+0 | 1 |
| 检索1次后模型停止，再生成 | 1 | 1 | 2+1 | 7 |
| 检索2次后程序收束，再生成 | 2 | 2 | 2+1 | 11 |

表中为正常完成路径，HTTP包含每次问题编码前后的两次模型身份GET；失败可能更少。不自动重试、不回退单轮；非法工具、参数、重复查询或协议截断直接停止。`recursion_limit`是额外防护，实际两次预算由图内条件与计数控制，不靠模型自行遵守。

发送范围也增加：规划服务接收原问题、工具说明及第一次返回的前5块正文、引用片段和必要测试/审核标记；每轮子查询发给自部署Embedding和云端重排；最终回答服务接收原问题及合并后的证据。不外发本机路径、向量或完整元数据，不执行文档命令，不保存会话或外发追踪。多步可能更慢、更贵，不保证优于单次检索；这是**单次提问内多步检索，不是多轮对话记忆**。

核心代码建议按以下顺序阅读：

1. `src/qa_platform/retrieval/knowledge_tool.py`：先看`SearchArgs/TOOLS`（模型可调用什么），再看`search_knowledge`（复用真实检索和重排），最后看`tool_reply/merge_sources`（最小工具消息与跨轮合并）。函数定义与工具schema都不会自行执行检索。
2. `src/qa_platform/workflows/agent_rag.py`：按`AgentState → build_graph`内`plan → search → answer → after_plan/after_search → add_node/add_conditional_edges → run`阅读。状态中的`messages`服务于本次决策，`rounds`保留本次两轮证据，`counts`统计实际成功完成的请求。`run`读取实际图事件，不编造执行步骤。
3. `src/qa_platform/rag/kubernetes_answer.py`：`call_answer`共用既有JSON与引用校验；原`generate`也调用它，避免两种模式各写一份生成逻辑。SDK显式清空OpenAI组织/项目环境标识，不把它们带往第三方服务。
4. `src/qa_platform/retrieval_lab.py`与`web/qa.html`：后端仅按`mode`选择真实图；前端只新增一个选项和折叠的“本次检索”子查询列表，失败清空旧答案。
5. `tests/test_agent_rag.py`与`tests/qa.test.cjs`：用真实LangGraph、SDK内存HTTP及合成检索替身验证0/1/2次路径、原生工具消息、证据保留、预算和界面边界；不将替身测试冒充真实模型验收。

2026-10-20，通过真实浏览器在`/qa`主动执行了一次复合DNS问题。Flash实际先检索“跨命名空间访问 Service 域名格式 namespace.svc.cluster.local DNS 解析失败”，再检索“Pod dnsPolicy dnsConfig 自定义 DNS 配置 resolv.conf search nameserver ndots 相对名称解析失败”；两轮后回答为`answered`、3个带引用要点，页面显示100.26秒、Embedding2/重排2/聊天3/HTTP11。没有重新编码文档或自动重试。本次是技术闭环验收，未据此认定回答完全正确或优于单次检索。单轮模型限制、多步选择及提交中的控件禁用也在浏览器核对；0/1次路径和失败场景通过离线替身验证，不冒充真实服务实测。全量离线测试708项通过，另有4项既有依赖警告；本机验收摘要与截图保存在`reports/agent-rag-acceptance-2026-10-20.*`，不提交Git。

### 最小多轮追问：短期会话，而非历史充当证据

`/qa`现默认开启页面级短期会话：先问完整问题，再问“那它的配置怎么看？”；页面显示本轮回答，较早问答及补全后的独立检索问题默认折叠。默认`--storage memory`时，“新对话”清除当前服务端会话与页面记录，刷新不恢复，服务重启丢失；不发送模型请求。可选MySQL持久化见下一节，与LangGraph运行检查点是两回事。

首次提问仍直接执行原图。追问先增加一次`contextualize`调用，把当前问题与服务端最近3轮摘要交给所选聊天服务，返回独立检索问题，然后重新运行原RAG或工具图。历史只解析指代，不能当作本轮事实证据；最终回答仍必须引用本轮检索原文。改写可能出错，展开“本轮独立检索问题”可核对；新话题应原样保留，不强行混入旧话题。

- 普通首问正常为Embedding1、重排1、聊天1、HTTP5；普通追问增加改写聊天1/HTTP1，共聊天2、HTTP6。
- 多步追问仍最多检索2次；加上改写后，最高Embedding2、重排2、聊天4、HTTP12。达到上限收束，不自动重试或回退。
- 会话只存原问题、独立检索问题、已校验回答摘要（每轮最多1800个序列化字符，必要时尾加`…`截取），不存原文、向量和来源元数据。每会话最近3轮，最多16会话，新建满额按最早创建顺序淘汰。
- 浏览器只传`conversation:true`及后续`conversation_id`，不接受任意`history`。成功才追加会话；失败保持旧会话；未知会话在模型调用前返回410，必须主动新对话。切换模型后，历史摘要会发送给新选择的服务；GLM的HTTP明文风险提示仍保留，不应输入敏感内容。
- 旧API不传会话参数时保持原无历史行为，便于单轮脚本兼容。重置接口`POST /api/qa/conversation/reset`只接受当前会话ID，同源校验与全局忙碌锁沿用，重复清理也成功。

新增`src/qa_platform/workflows/conversation.py`负责内存摘要与一次改写；`retrieval_lab.py`在共用锁内串联取历史、改写、原图及成功提交；`live_rag.py`仅扩展配置与说明；`web/qa.html`保留简单页面。`tests/test_conversation.py`验证会话隔离、历史裁剪、失败不追加、改写协议、默认单轮兼容及请求计数；前端会话测试合并到`tests/qa.test.cjs`。消息状态由应用管理的思路参考[OpenAI Docs会话状态说明](https://developers.openai.com/api/docs/guides/conversation-state)，实际仍使用现有第三方兼容服务，未改为OpenAI托管会话。

2026-10-20真实浏览器验收：先问“Pod 里访问 Service 域名时 DNS 解析失败，应该检查哪些地方？”，97.38秒、`partial`、HTTP5；再追问“那它里面的 nameserver 和 search 应该怎么看？”，实际改写为“在排查 Pod 内 Service 域名 DNS 解析失败时，如何检查 Pod 内本地 DNS 配置文件中的 nameserver 和 search 字段？”，102.83秒、`answered`、HTTP6。两轮各重新编码一个问题、重排一次，第二轮聊天2次（改写+回答），历史及独立问题可在页面展开。没有为提速而省略重新检索，也没有把该样例当作正确率证明。全量离线测试741项通过、4项既有依赖警告；前端Node测试25项通过。验收记录和截图仅保留本机`reports/conversation-*-2026-10-20.*`，未提交Git。

### SSE实时阶段进度：可见等待，不提前显示未校验答案

`/qa`现在使用`fetch`主动POST到`/api/qa/stream`，接收`text/event-stream`。同一条请求依次收到实际节点的开始/完成通知：理解追问、规划、检索、重排、生成与校验。页面仍只有原状态文字，不加大对照台、虚构百分比或定时模拟进度。该改动不增加模型调用，也不改变原单次/多步/追问预算；SSE本机HTTP不计入模型服务HTTP。

这是**阶段进度流，不是逐字答案流**。进度只携带固定阶段名与状态，不含问题、证据、Token、密钥或模型草稿。后端校验完成并处理本轮会话后才发送唯一`result`终态，前端完整收到终态、流结束且原回答校验通过，才展示答案和追加历史。运行失败发送固定脱敏`error`，前置的403/409/410/422仍在流头之前返回。每10秒的注释心跳只维持连接，不代表完成了工作。

浏览器断开后不会自动重连、重试或回退JSON重复付费。已发出的同步SDK请求无法强行撤销，可能继续运行并计费；程序在后续阶段边界合作停止，收到取消标记后在提交会话前再检查。全局忙碌锁由工作线程真正退出时释放，不因网页离线提前放开。已经提交的事务不承诺回滚，断线也不能用来推断“没有生成、没有费用”。事件队列最多64条；仍是本机单任务实现，尚未验证生产代理、并发部署或服务重启恢复。

原`POST /api/qa`保留JSON接口，页面只有在配置缺少stream时才选择旧接口；一旦已启动流，绝不因失败再调用旧接口。客户端支持UTF-8分片、LF/CRLF、多行data和注释，有缓冲长度上限；未知事件、坏JSON、重复终态、截断或缺终态一律拒绝，不保存旧答案冒充新结果。

实现文件：`workflows/progress.py`用ContextVar隔离本次通知与取消；原图和知识工具在真实工作前后报告阶段；`web_stream.py`用一个同步工作线程和有界异步队列桥接到StreamingResponse；`retrieval_lab.py`让JSON与SSE共用执行逻辑及校验；`web/qa.html`解析进度但只渲染已校验终态。协议参考[MDN的SSE事件格式](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)，响应实现参考[Starlette StreamingResponse](https://starlette.dev/responses/#streamingresponse)。

2026-10-20，真实浏览器提问“Pod 与 Service 不在同一命名空间时，DNS 查询应该指定什么？请简短回答。”：答案到达前实际看到“混合检索中…”，最后收到`answered`及原文引用，20.50秒，Embedding1/重排1/聊天1/HTTP5。此问题和前轮不同，不能把耗时差归因于SSE；SSE也不保证模型生成更快。断线、心跳、失败、多步与追问进度使用离线测试验证，不冒充新增真实模型样例。当时全量765项测试通过、4项既有警告；Node前端32项通过。`tests/test_qa_stream.py`含原始ASGI测试，证明工作尚未结束时已交付进度，不能仅凭会缓冲的TestClient断言实时性。本机验收记录在`reports/sse-acceptance-2026-10-20.json`，不提交Git。

### MySQL + SQLAlchemy：问答留存与刷新恢复

用户已通过MySQL命令行验证`127.0.0.1:3306`、MySQL 9.7.0、`root@localhost`连接及查询成功。Navicat是独立客户端，打不开不阻塞本功能。**2026-10-21用户在本机终端完成初始化；专用账号与表结构只读核验通过，8766服务已切换MySQL模式。真实问答写入及刷新恢复仍待验收，不能只凭初始化成功宣称通过。**

先查看零连接计划；默认不会读取密码、配置、连接数据库或写文件：

```bash
/Users/patton/miniconda3/envs/langchain/bin/python -B scripts/setup_mysql.py
```

然后在自己的交互终端运行初始化，按提示输入MySQL root密码。不要把密码发到聊天，不要在命令参数中写密码：

```bash
/Users/patton/miniconda3/envs/langchain/bin/python -B scripts/setup_mysql.py --bootstrap
```

初始化只针对**新**`fault_diagnosis`库和`fault_qa@127.0.0.1`账号：先检查重名，再独占创建权限`0600`的`models/database.env`保存随机专用密码，建两张表，最后授予项目库`SELECT/INSERT/UPDATE/DELETE`并以专用账号执行只读结构检查。root密码不保存；应用不使用root、无建表或全局权限。旧库、账号或配置已存在即停止，不覆盖、不删库、不重置账号。MySQL DDL不是跨这些操作的整体事务；中途失败会保留配置及可能创建的局部资源，需人工核查，不应反复初始化。[MySQL CREATE USER文档](https://dev.mysql.com/doc/refman/9.7/en/create-user.html)

配置模板为`models/database.env.example`；真实`models/database.env`被Git忽略，和模型`models/.env`分开。SQLAlchemy使用`URL.create`处理密码特殊字符，关闭SQL日志并隐藏参数；每次会话操作独立Session/事务。参考[引擎与URL配置](https://docs.sqlalchemy.org/en/20/core/engines.html)及[Session事务](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html)。依赖在`database`可选组；当前环境已有SQLAlchemy与PyMySQL，不自动升级。

初始化成功后先做只读检查，再停止原8766服务（原终端`Ctrl+C`）并显式启用MySQL：

```bash
/Users/patton/miniconda3/envs/langchain/bin/python -B scripts/setup_mysql.py --check
/Users/patton/miniconda3/envs/langchain/bin/python -B scripts/serve_retrieval_lab.py --port 8766 --storage mysql
```

仍打开`http://127.0.0.1:8766/qa`，没有新增管理大面板。旧内存对话不会自动迁入数据库，切换后需重新提问。成功问一题后刷新同一标签页，再追问即可验收；刷新恢复不检索、不请求模型。只有启用MySQL的页面在`sessionStorage`保存一个随机会话ID，不在浏览器持久存正文或密码。关闭标签页不保证保留ID；目前没有账号登录、历史会话列表或跨设备恢复。数据库不可用时启动/请求会明确失败，不静默回退内存，不重复调用模型。

| 保存位置 | 保存内容及用途 |
| --- | --- |
| `qa_conversations` | 随机会话ID、创建时间、绝对轮次、是否仍可继续 |
| `qa_turns` | 每轮原问题、独立检索问题、模型、已校验完整回答、真正引用的原文和公开来源、请求计数、步骤、耗时及摘要 |
| 最近3轮摘要 | 追问时取出用于解析指代；不把历史原文自动发送给模型，事实仍来自新检索 |

只在整轮生成及引用校验成功后，以一笔数据库事务更新轮次并保存问答；失败不追加。数据库保留全部已成功记录，不沿用内存模式的16会话淘汰。“新对话”把旧会话设为结束、清除页面及浏览器ID，**不删除数据库历史**。运行中的LangGraph节点状态、失败草稿、未引用候选、向量、本机来源路径和密钥不保存；这不是LangGraph检查点恢复，不承诺重启后继续一半的模型请求。数据保存在本机数据库也不等于加密、备份、权限隔离或合规留存方案；当前只供本机单用户学习。

本阶段文件：`storage/mysql_setup.py`负责安全配置、初始化与只读检查；`storage/sql_conversations.py`负责两张ORM表和事务存取；`scripts/setup_mysql.py`只是入口；`retrieval_lab.py`注入存储并提供恢复API；`web/qa.html`在原页面恢复会话。离线测试用临时SQLite验证ORM存取/事务/重开连接，用Mock验证初始化安全边界，用Node验证刷新恢复；**SQLite只用于测试，应用不回退SQLite，也不能替代真实MySQL方言、权限和服务重启验收。**

2026-10-21，全量810项测试通过，4项既有警告；其中数据库存储/API与初始化安全测试45项，Node前端40项。上述自动测试没有使用真实MySQL凭据或请求模型。随后用户完成真实初始化；额外只读核验确认专用账号为`fault_qa@127.0.0.1`、两表列及主外键匹配、仅有项目库CRUD权限且无DDL或有效全局权限。当时两表均0条记录，页面实际显示MySQL持久化提示；本次切换没有请求模型或写入合成记录，真实问答刷新恢复仍待验收。

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

#### 引用粒度：按行与按段的零模型对照

完成本轮重排后，可单独点击“预览引用切分（零模型请求）”，选择S1～S5对照旧 `line-v1` 与新 `paragraph-v1`。两路使用同一块正文，且都保持单片段最多240个Unicode码点：

- `line-v1` 沿用旧按行规则，文本里的单次换行也会分开编号。
- `paragraph-v1` 将普通段落内的软换行保留在同一个引用中；空行、标题、列表新项和代码围栏形成边界。列表续行与代码正文保留内部原始换行，标题、围栏和说明标签本身不作为事实证据。
- 长单位优先在句末或换行处分开，必要时仍会硬切；拆分会标记。短于6字符的单位不独立引用，可能出现旧规则无片段、新规则合并后有片段的情况。这是轻量结构规则，不是完整Markdown解析器，也不保证句义或代码语义完整。

每个新片段提供原文的 `[start,end)` 码点范围、单位类型和与之重叠的旧Q编号。片段始终是原文的连续子串：**不把换行替换成空格，不改写、拼接、跨chunk补文，不改变向量或任何召回／重排名次**。例如某块原文中的“需要检查你Pod中的”与下一行的文件名，可以在同一段引用里显示，而不再分别占用两个引用名额；这仍不保证模型一定会选对引用。

预览通过同源本机 `/api/quotes` 只提交当前实验编号，不读取密钥、不调用模型、不重新检索或重排，模型服务HTTP计数不变（本机页面API往返不计入该口径）。预览可在生成前或生成后查看，保留既有回答及其引用；失败不触发付费重试。刷新或服务重启后旧实验编号失效，仍需先建立新的检索／重排轮次。

新的Kubernetes回答明确记录 `answer_generation.quote_strategy=paragraph-v1`，模型只收到当前按段片段，不收到旧规则对照。完整JSON与复制摘要同时保留策略、引用字典和位置映射。旧报告没有该字段时仍按 `line-v1` 解释，不能把旧Q14直接当作新版Q14；旧铁路演示及未显式启用新策略的调用保持原规则。

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
| `rag/quote_units.py` | 单块原文 → 有位置的段落／列表／代码引用单位；纯本地规则，不补写或跨块取文 |
| `retrieval_lab.py` | 本机页面按钮 → 严格校验请求、串行执行、返回JSON；不接受用户指定文件路径或服务地址 |
| `web/retrieval_lab.html` | JSON → 三列卡片、同块高亮和来源原文；所有资料按文字展示，不执行文档命令 |
| `scripts/serve_retrieval_lab.py` | 仅启动入口，将运行交给包内后端 |
| `tests/test_kubernetes_hybrid.py` | 合成资料和假客户端 → 零网络回归，检查一次编码、范围、融合、失败关闭和API边界 |
| `tests/test_kubernetes_rerank.py` | 合成候选／假云端响应 → 全池覆盖、分数与身份校验、快照隔离、成功复用及API安全边界 |
| `tests/test_kubernetes_answer.py` | 合成证据／假聊天流 → 前5范围、引用校验、拒答／截断、模型认证隔离、成功复用和失败保留 |
| `tests/test_quote_units.py` | 合成Markdown／假API → 软换行、代码与列表边界、码点位置、长度上限、旧规则兼容及零模型预览 |
| `web/retrieval_export.js` | 已完成的JSON结果 → 深拷贝报告、可复制摘要和安全文件名；不接触网络、数据库或剪贴板 |
| `tests/retrieval_export.test.cjs` | 合成数据／浏览器替身 → 报告完整性、复制回退与页面状态的Node离线回归 |
| `tests/test_retrieval_export.py` | 将Node回归纳入pytest，并验证静态脚本不触发检索；没有Node时明确跳过JS测试 |

Python实现路径相对于 `src/qa_platform/`。检索、重排、回答分步进行，暂不混入上下文补全。服务仅监听127.0.0.1，要求同源JSON和自定义请求头，无CORS；同一时刻只允许一轮检索、重排或生成，不自动重试、不持久保存问题或报告。真实混合模式可能更新Milvus运行锁/日志，但不插入、修改或删除记录。检索中途失败时不展示部分或旧结果；重排／回答失败保留本轮已完成阶段，已发出的模型请求也不会被当作未发生。

### LangGraph：普通函数与图编排的零模型回放

仍使用同一个启动入口，另开 <http://127.0.0.1:8766/workflow>。本页不依赖Kubernetes索引就绪，不需要先检索、重排或生成回答；不会更改检索台当前实验快照。首次打开只GET本地计划，点击“运行两路零模型对照”才POST空对象执行两路回放。运行后点击流程节点、步骤或“下一步”，只是查看已完成轨迹，不再次执行节点或请求模型。

本步验证并固定 **LangGraph 1.2.16**：截至2026-10-19的[PyPI最新稳定版本](https://pypi.org/project/langgraph/1.2.16/)，2026-10-16发布，支持Python≥3.10。[官方版本说明](https://github.com/langchain-ai/langgraph/releases/tag/1.2.16)。在指定Conda环境升级后，使用`pip check`与全量离线回归检查兼容性；该环境原有的LangChain 1.2.15限制LangGraph<1.2，因此同步升级LangChain到兼容的1.4.6。项目核心直接依赖仍是`langchain-core`；LangChain高层包不是本回放的必需调用组件。

新的环境可先查看`python -m pip install --dry-run -e '.[workflow]'`的依赖计划，再安装；不要为追新而无差别升级整个环境。`pyproject.toml`只固定本步LangGraph入口版本，不是完整依赖锁文件。本页会显示实际安装版本；缺依赖时不提供伪造的LangGraph执行结果。

**控制条件与真实执行边界：**

| 节点 | 本步实际执行 |
|---|---|
| `retrieve` | 读取手工合成的BM25/Dense候选；不运行真实检索或问题编码 |
| `fuse` | 调用已有`reciprocal_rank_fusion`，在本地重新计算RRF |
| `rerank` | 读取合成分数，并对完整候选并集排序；不调用Reranker |
| `prepare_evidence` | 选合成重排前2块，调用已有段落引用规则，构造S/Q字典 |
| `generate` | 读取手工编写的回答草稿；没有模型推理或原始模型输出 |
| `validate` | 调用已有`parse_answer`，真正检查JSON、状态规则及引用编号／原文定位 |

快照中的正文、分数、草稿均为教学合成资料，不是用户上一轮的真实结果，也不证明模型能力。两路使用同一组节点函数，节点不原地改输入，只返回本步更新：

- **普通Python：** 调用方在显式顺序列表里调用函数，再执行`state.update(update)`，并自行记录轨迹。
- **LangGraph：** `ReplayState`声明共享状态；`add_node`注册相同函数；`add_edge`声明顺序；`compile`构建运行器；`stream`产生节点更新与合并后的状态。没有自定义reducer时，各字段按覆盖方式更新。
- 每条轨迹保存`before`（读入状态）、`update`（节点返回值）、`after`（合并后状态）。图路线的`after`来自LangGraph实际`values`事件，不是浏览器拼出来的假状态；页面流程图也来自实际编译图，不请求外部绘图服务。
- 两路对照检查节点顺序、每步更新、最终状态及答案相同；相同是固定输入下预期的编排等价结果，**不是准确率、性能提升或“LangGraph更聪明”的证明**。普通Python也能实现分支、日志和状态管理，LangGraph提供统一的图调度机制。

本线性页模型请求为0：Embedding、Reranker、聊天、模型服务HTTP均为0；浏览器与127.0.0.1的页面/API通信另计。不加载`.env`、不创建模型或Milvus客户端、不读写向量缓存或报告。执行期间显式关闭LangSmith tracing，即使环境中已开启追踪也不外发。线性页没有条件分支、自动重试、checkpointer、跨进程恢复或多轮记忆；这些能力不能仅凭采用LangGraph就算已经完成。条件边另见下方简化课件。

#### 从定义、执行图到引用关系

学习页提供定义探查器，可以查看`ReplayState`、`shared_nodes`创建／传递、当前节点函数以及`build_graph`的实际源码，附其在`replay.py`中的行号。源码由后端对已知Python对象进行只读提取，页面按文字展示，不是可执行编辑器，也没有接受任意文件路径的接口。`ReplayState`的字段类型来自实际类型注解；节点读写说明来自本实验的`NODES`声明，不冒充自动静态分析。

- `ReplayState(TypedDict, total=False)`是类型约定：状态可以从只有`query`开始，再逐步增加候选、证据、草稿和答案；它不会自动创建这些字段，也不会自动进行运行时类型校验。`total=False`不意味着某个节点可以缺少自己实际使用的输入字段。
- `make_nodes(fixture)`创建六个Python函数，并返回名称到函数对象的字典。`_execute_shared`把它命名为`shared_nodes`，交给普通执行器和图构建器。因此它不是LangGraph保留名称，更不是另一组模型。
- 绘制的SVG节点和箭头来自实际`graph.get_graph()`结果，在本机浏览器完成绘图，不调用外部Mermaid绘图服务。图节点可选择相应执行步骤；本图只有顺序边，尚未添加分支。
- 状态字段、节点定义和轨迹联动；查看不同步骤时，字段的“已存在／本步写入”以该步`before/update/after`为准，不把最终状态提前当成节点输入。

`evidence`与`draft`是两类数据，不是同一份内容的两种格式：

| 路径 | 含义 |
|---|---|
| 顶层`evidence` | 回答所选的原文资料字典，不是整座知识库；由`prepare_evidence`构造 |
| `evidence.S1.chunk_id` | 资料块的身份；`S1`只是本轮分配的短编号，换轮后可能指向另一块 |
| `evidence.S1.text` | 所选块的完整正文 |
| `evidence.S1.quotes.Q1` | 该资料内一个可引用的连续原文片段；`Q1`在不同S来源下可重复，须用`S1/Q1`定位 |
| `draft.status` | 草稿表达的回答充分性；`partial`表示只能部分回答，不表示程序执行了一半或JSON不完整 |
| `draft.claims[i].text` | 草稿中的第i个回答要点，允许概括原文，不必与原文逐字相同 |
| `draft.claims[i].evidence` | 此要点的引用数组，只保存`source_id/quote_id`，通过它返回原文；不是再次保存资料全文 |
| `draft.missing_info` | 需要补充或核实的信息，不是HTTP／程序异常日志 |

引用探查器按**当前所选步骤**展示：第4步才有`evidence`，第5步才有`draft`，第6步校验通过后才有`answer`。点击草稿的`S1/Q1`会定位到对应资料与原文片段，并显示引用路径。编号切换只改变显示，不请求模型或重新回放。

本例S1、S2正文都只有一个可引用段落，所以各自`quotes.Q1`与`text`相同；长块一般会有多个Q片段，二者不应混为一谈。JSON中的`\n`是原文换行的转义表示，并不代表从邻块补写了新内容。`validate`检查结构、编号与原文定位后写入`answer`，本例草稿与答案可以完全相同：改变的是它们所处的处理阶段，不代表已经证明每个推断正确。草稿仍来自手工合成快照，不是本轮新模型生成。

**按文件阅读：**

| 文件 | 为什么需要／输入与输出 |
|---|---|
| `src/qa_platform/workflows/__init__.py` | 将工作流目录声明为Python包；不执行回放或联网 |
| `src/qa_platform/workflows/replay_fixture.json` | 手工合成的文档、召回、重排分数和草稿，固定两路输入；不包含密钥或真实调用结果 |
| `src/qa_platform/workflows/replay.py` | 核心课件：`ReplayState`→共享节点→普通执行器/StateGraph→两路轨迹与一致性结果；`describe_definitions`只读提取类型和真实源码，`_execute_shared`展示同一函数字典如何交给两路 |
| `src/qa_platform/retrieval_lab.py` | 新增固定页面、只读计划和空请求回放接口；复用同源/JSON/大小限制/串行锁，不改变真实检索快照 |
| `web/workflow_lab.html` | 渲染真实回放轨迹、源码定义、SVG执行图和资料／草稿引用关系；选择节点、字段与引用均只更新本地显示 |
| `web/retrieval_lab.html` | 只增加另开工作流学习页的链接，原有检索/重排/回答逻辑不变 |
| `tests/test_workflow_replay.py` | 验证真实图执行、状态隔离、错误停止、零模型/追踪外发、API门禁及原快照保留 |
| `tests/workflow_lab.test.cjs` | 验证页面合同、流程图、步骤切换、重复点击/失败、响应校验及文字安全展示 |
| `tests/test_workflow_lab_ui.py` | 将上述Node前端回归纳入pytest；没有Node时明确跳过 |
| `pyproject.toml` | 声明可选`workflow`依赖及合成JSON包资源；不自动安装或升级环境 |

#### 下一课：最小条件分支与建图阅读顺序

打开 <http://127.0.0.1:8766/workflow/branching>，仍使用原启动命令和LangGraph 1.2.16，不新增依赖。主界面只保留场景选择、实际分支图、当前LangGraph步骤；不再展示或执行普通Python路线。真实源码和完整JSON收进默认折叠区，原线性对照页保留为上一课。

先选“有候选资料”运行，再选“召回为空”运行。后者保持同一问题和资料，只将BM25/Dense的合成召回列表置空；清空是受控实验条件，不是对真实模型的质量结论。更换场景立即清除旧结果，必须主动运行才产生新轨迹。

| 场景 | prepare_evidence后的实际走向 | 未执行的节点 |
|---|---|---|
| 有候选资料 | `generate → validate`；读取原合成草稿 | `refuse` |
| 召回为空 | `refuse → validate`；固定模板、空claims及资料缺口 | `generate` |

共同前四步仍是`retrieve → fuse → rerank → prepare_evidence`。`route_after_evidence(state)`只检查证据字典是否非空，返回`generate`或`refuse`，**不写状态**；路由函数不是新的状态节点。LangGraph用`add_conditional_edges`注册该判断，运行器按返回值选择下一条边。两个分支都注册在编译图中，但一轮只执行其中一个；图中“未执行”不能当成执行过的轨迹。该用法参见[官方条件边说明](https://docs.langchain.com/oss/python/langgraph/graph-api#conditional-edges)。

本课最重要的边界：**非空不等于相关、充分、可信**；没有用任意重排阈值代替证据审核。固定拒答是正常空证据路径，不是程序异常兜底。结构或引用校验失败仍报错停止，既不重试，也不悄悄转为成功拒答。页面入口`run_replay`只执行一次LangGraph；原`compare_replay`保留给离线回归核对，不被本页API调用，也不返回其普通Python轨迹。图路线的状态仍取自真实stream values。

展开代码区后，先按编号阅读，不必自己猜下拉框顺序。这里是**源码阅读顺序**，不是节点执行顺序；“源码下一步”也不会推进图或重新请求接口。

1. `replay.py → ReplayState`：先知道共享状态能装什么；类型声明不等于已经有值。
2. `replay.py → make_nodes`：看六个基础函数如何定义、各自返回什么更新，最后如何放进字典。初读可先跳过具体计算细节。
3. `branching.py → make_nodes`：复用上述函数，加上`refuse`绑定，组成七个节点的注册表；返回函数对象不是调用函数。
4. `branching.py → route_after_evidence`：了解条件判断的输入和两个返回值；它选择后继，不产生节点更新。
5. `branching.py → build_graph`：按真实源码中的六条编号注释读：`StateGraph`创建构建器 → `add_node`注册 → 固定边 → 条件边 → 汇合和END → `compile`。编译到此完成，仍未执行节点。
6. `branching.py → run_replay`：回到页面实际入口，看固定输入、注册表、图、执行器和输出如何连起来。`_observe_router`只是路由日志包装，不改变判断规则。
7. `branching.py → _run_langgraph`：重点看`graph.stream`才真正开始执行；`updates`是节点返回值，`values`是框架合并后的状态。其余代码主要负责轨迹记录和一致性检查。

每步显示一个阅读目标并提供源码“上一步／下一步”。“当前节点函数”是补充入口，随图中所选执行节点变化；不会打乱上述阅读进度。先理解**定义 → 注册 → 连边 → 编译 → 执行**这条主线，再深入节点计算。

新增/修改文件的学习职责：

- `src/qa_platform/workflows/branching.py`：复用上一课的状态与六个节点，加固定拒答、纯路由函数和真正的条件图；页面走`run_replay`，只读提取七步涉及的真实源码。旧Python条件执行器仅供自动测试。
- `src/qa_platform/retrieval_lab.py`：只增加固定页面、只读计划和场景回放接口。POST仅允许`scenario`枚举，沿用同源、JSON、16KiB及同一个串行锁；不改变真实检索实验快照。
- `web/branching_lab.html`：只显示LangGraph的图和当前步骤，按七步阅读路线切换源码；展开时才显示代码和完整状态。源码阅读导航与执行步骤导航相互独立，不新增请求。
- `tests/test_workflow_branching.py`：离线验证两种分支、未选节点确实不运行、真实图/状态/源码、零外部IO以及API门禁。
- `tests/branching_lab.test.cjs`、`tests/test_branching_lab_ui.py`：检查场景切换清空旧结果、默认折叠、分支展示、响应校验与安全文本，并接入pytest。

只读计划为`GET /api/workflow/branching/plan`，主动运行是`POST /api/workflow/branching/replay`，请求为`{"scenario":"with_evidence"}`或`{"scenario":"empty_evidence"}`。模型推理及模型服务HTTP仍全部为0；没有接管真实RAG、增加持久化或实现自主Agent。

#### 再下一课：生成前暂停、人工确认与恢复

打开 <http://127.0.0.1:8766/workflow/approval>，仍使用同一个服务和LangGraph 1.2.16。与上一课“执行完后查看轨迹”不同，这次图会**真的暂停**：`retrieve → fuse → rerank → prepare_evidence → review`，`review`内调用`interrupt()`后返回控制权，此时尚无`approved/draft/answer`。

点击“继续”才会向相同的`thread_id`提交`Command(resume=True)`，完成`review → generate → validate → END`；选择取消则提交`False`，走`review → cancel → END`，只有`cancelled=True`，不创建草稿或答案。取消不是证据不足拒答，也不是程序异常。这里的确认只控制合成教学回放，不代表资料审核通过，不授权任何云端调用。

检查暂停、继续后的真实计数：

| 观测对象 | 暂停时 | 继续后 |
|---|---|---|
| 前四个节点的完成次数 | 各1次 | 仍各1次，没有重跑 |
| `review`进入/完成次数 | 1 / 0 | 2 / 1 |
| `generate`完成次数 | 0 | 1；取消路径仍为0 |

恢复不是从Python函数内部的那一行直接续跑：包含`interrupt`的节点会**从头重新进入**，因此`interrupt`之前的代码会再执行。不要在它之前放不可重复的业务写入或收费请求。本课计数包装器有意记录实际函数进入与正常返回，除此以外没有业务副作用。状态与完成事件来自真实LangGraph `stream`和`get_state`，没有把暂停画面伪造为一次完整回放。[官方中断与恢复说明](https://docs.langchain.com/oss/python/langgraph/interrupts#resuming-interrupts)。

本课使用`InMemorySaver`，检查点只保留在服务进程内存。刷新页面会通过只读接口找回本app当前会话；重启服务、重新开始或运行失败后会话失效。**不是磁盘持久化、断电恢复、多轮对话记忆或多用户会话系统。** 每个app只保留最近一轮，另一标签页重新开始也会使旧编号失效。`thread_id`只是这轮教学图的检查点指针，不是Codex聊天ID。相同编号恢复原检查点，不是把`query`重新提交来启动新图。

主界面只保留开始、继续、取消、当前状态和关键计数。源码与完整状态仍默认折叠，按六步阅读：`ApprovalState → review_evidence → route_after_review → build_graph → ApprovalLesson.start → ApprovalLesson.resume`。重点看`compile(checkpointer=...)`、`interrupt(...)`、同一`thread_id`和`Command(resume=...)`怎样配合；不必一开始就深挖事件记录辅助代码。

新增/修改文件职责：

- `src/qa_platform/workflows/approval.py`：复用原六个节点，加人工确认、取消和内存检查点；单会话管理器控制开始/恢复/只读观察，旧会话及重复恢复被拒绝。真实计数明确区分进入与完成；运行错误使会话失效，不自动重试。
- `src/qa_platform/retrieval_lab.py`：增加`/workflow/approval`页面与三个本地接口：GET `/api/workflow/approval/state`、POST `/api/workflow/approval/start`（空JSON）、POST `/api/workflow/approval/resume`（当前32位`session_id`及严格布尔`approved`）。沿用同源、自定义头、16KiB和同一个忙碌锁；不修改真实RAG快照。
- `web/approval_lab.html`：展示真实暂停状态、检查点与执行次数，主动提交决定；刷新只读状态，不自动开始、恢复或重试。`web/branching_lab.html`只加下一课链接。
- `tests/test_workflow_approval.py`：验证真实中断/恢复、前序不重跑、确认节点重进、取消与重复恢复、只读状态、检查点隔离、零外部调用和API门禁。
- `tests/approval_lab.test.cjs`、`tests/test_approval_lab_ui.py`：验证按钮状态、暂停/恢复生命周期、旧会话清理、源码顺序与安全展示，并纳入pytest。

所有文档和分数仍为手工合成数据，没有新模型推理、文档编码或数据库操作；没有读取`.env`，没有向LangSmith发送追踪。浏览器与本机服务的HTTP请求不等于模型服务请求。

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
| `setup_mysql.py` | `storage/mysql_setup.py` | 显式初始化MySQL专用账号与问答表、只读检查；默认零连接计划 |
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

当前学习路线：模型调用 → 文档处理 → BM25／Dense → RRF → Reranker → 最小RAG → 开发回归 → Milvus对照 → Kubernetes语料与可视化 → 本地Embedding小样本 → 编码缓存续跑 → 缓存向量独立入库 → 真实问句检索对照 → 相邻块上下文补全实验 → 全池重排 → 带段落引用回答 → LangGraph教学与真实单轮编排 → 原生Tool Calling验证。

当前Kubernetes子集已跑通单轮问答、受控多步知识检索、短期追问及SSE阶段进度；MySQL/SQLAlchemy代码与离线测试、本机初始化和服务切换已完成，下一步验证真实问答写入与刷新恢复，再补缓存、增量知识管理、部署和系统评测。这不是整个实习项目已经结束。评测样本随开发逐步积累；指标必须由真实标注与测量得出，不能用离线测试通过数量代替。合成学习页独立保留，不继续扩展为主流程，也不额外加入复杂审批功能。

学习过程中，每增加一份实现文件都说明其输入、输出、职责和调用关系；对比实验应固定控制条件，不将“能运行”“排序一致”或“自动检查通过”直接当作问答正确率。
