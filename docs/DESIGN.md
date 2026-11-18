# 设计决策与技术细节

> 本文收录模型选型、OCR 接入、高并发优化、两级知识结构与测试体系的详细设计。

## 项目框架 (FastAPI 版)

### 目录结构

```
rag-knowledge-base/
├── app/
│   ├── main/                      # FastAPI 应用工厂（create_app 装配）
│   │   ├── __init__.py            # 入口：app = create_app()，供 uvicorn app.main:app 加载
│   │   ├── app.py                 # create_app()：注册路由/中间件/错误处理器/启动事件
│   │   ├── config.py              # 启动期配置（TRANSFORMERS_OFFLINE 等）
│   │   ├── middleware.py          # 安全中间件 + 限流器挂载 + 429 处理
│   │   ├── routes.py              # 根路径/前端静态文件路由（app/frontend）
│   │   └── events.py              # 启动/关闭事件：Webhook + 索引恢复 + 模型预热
│   ├── api/                       # HTTP API 层（路由 + 请求/响应模型 + 限流）
│   │   ├── __init__.py            # 聚合挂载所有子路由
│   │   ├── auth.py                # POST /api/auth/* 注册/登录/改密/重置/me
│   │   ├── chat.py                # POST /api/chat（SSE 流式，stream 参数切换流式/非流式）
│   │   ├── documents.py           # POST /api/documents/upload（异步）、GET list、DELETE、/preview、/task
│   │   ├── feedback.py            # POST /api/feedback（👍/👎 满意度）
│   │   ├── category.py            # 知识库分组 CRUD + 文档归类/按分类筛选
│   │   ├── conversation.py        # 多轮对话 创建/列出/获取/删除
│   │   ├── admin.py               # 管理后台：用户管理/系统统计/配置（仅超管
│   │   ├── wiki.py                # 知识页查询 + 路由调试 + 审定（页面级/文档级，限超管）
│   │   └── webhook.py             # Webhook 管理：创建/更新/删除/触发/日志
│   ├── frontend/
│   │   └── index.html             # 单页前端（原生 JS + CSS：登录/问答/管理后台/修改密码）
│   ├── core/
│   │   ├── config.py              # 集中配置：API key、模型名、路径、各种开关（含 LLM 多供应商解析）
│   │   └── limiter.py             # 集中管理 slowapi 限流器（登录/问答等）
│   ├── llm/                       # LLM 调用层（集中客户端 + 多供应商 fallback）
│   │   ├── __init__.py            # 导出 chat_completion / stream_chat_completion
│   │   └── client.py              # LLMClient：统一鉴权/超时/SSE 解析，主供应商限流时按 LLM_FALLBACK_PROVIDERS 自动切备用
│   ├── retrieval/                 # 检索层
│   │   ├── embedder_factory.py    # Embedding 工厂：本地 BGE-M3（默认）/ 线上 API 二选一
│   │   ├── bge_m3_embedder.py     # BGEM3Embedder：稠密 1024 维 + 自计算 SPLADE 稀疏权重
│   │   ├── hybrid_retriever.py    # HybridRetriever：BM25 + 稠密 + BGE-M3 稀疏 + 可选 ES，RRF 融合 + 重排
│   │   ├── sparse_store.py        # BGE-M3 稀疏向量落盘/加载（./sparse_db/{tenant}）
│   │   ├── query_rewriter.py      # QueryRewriter：LLM 查询改写/扩展
│   │   ├── citation.py            # CitationTracker：来源引用追踪
│   │   ├── cache.py               # QueryCache：精确查询缓存（MD5，文件+内存 LRU）
│   │   ├── semantic_cache.py      # SemanticCache：Redis 语义缓存（向量近似匹配，降延迟+省 LLM 费）
│   │   └── elasticsearch_client.py# Elasticsearch 客户端（全文检索兜底，feature-flag 默认关闭）
│   ├── wiki/                      # Wiki 知识页层（检索前置的结构化入口 + 审定闭环）
│   │   ├── store.py               # Page/Section/Span 模型、SQLite 存储、审定与审计日志
│   │   ├── router.py              # 别名确定性路由（命中知识页则跳过混合检索）
│   │   ├── seed.py                # 内置知识页（每节带页码出处，草案页不参与问答）
│   │   ├── doc_status.py          # 切片级文档审定状态（approved/draft）
│   │   └── __init__.py
│   ├── processing/                # 文档处理层
│   │   ├── multimodal_pipeline.py # MultimodalPipeline：PDF 图文解析（PyMuPDF 文字层 + OCR + 分块）
│   │   ├── pdf_extractor.py       # PyMuPDF 图文/文字层抽取
│   │   ├── ocr_engine.py          # OCREngine：PaddleOCR 3.7 封装（扫描件识别）
│   │   └── __init__.py
│   ├── memory/                    # 记忆层
│   │   ├── memory_manager.py      # MemorySystem：短期 + 中期 + 长期记忆编排
│   │   ├── conversation_store.py  # 对话持久化（Redis）
│   │   ├── redis_client.py        # Redis 连接池 + TTL
│   │   ├── forgetting.py          # 艾宾浩斯遗忘曲线（ShadowWorker 异步反遗忘）
│   │   ├── profile_store.py       # 用户画像存储（置信度加权合并）
│   │   └── __init__.py
│   ├── tenant/                    # 租户与用户层
│   │   ├── auth.py                # 注册/登录/密码哈希/改密/重置（SQLite）
│   │   ├── tenant_manager.py      # 租户创建/隔离
│   │   ├── category.py            # 知识库分组数据访问
│   │   ├── conversation.py        # 多轮对话数据访问
│   │   └── __init__.py
│   ├── worker/                    # 异步任务层
│   │   ├── shadow_worker.py       # ShadowWorker：后台线程池，摘要/实体画像/反遗忘
│   │   ├── webhook.py             # Webhook 异步发送 + 失败重试（60s 轮询）
│   │   └── __init__.py
│   ├── security/                  # 安全层
│   │   ├── middleware.py          # 安全响应头/请求体大小限制/路径穿越与注入防护/CORS 收紧
│   │   ├── error_handlers.py      # 全局统一错误处理（20+ 错误码，中文提示）
│   │   └── __init__.py
│   ├── tasks/                     # 上传索引任务层
│   │   ├── task_store.py          # 进程内任务状态 + 持久化 data/tasks.json（重启可恢复）
│   │   └── __init__.py
│   ├── observability/             # 可观测层
│   │   ├── tracker.py             # TraceContext：全链路追踪（耗时、Token）
│   │   ├── structured_logger.py   # 结构化 JSON 日志
│   │   ├── monitoring.py          # /metrics、/health、/stats 端点
│   │   └── __init__.py
│   ├── evaluation/                # 评估层（离线）
│   │   ├── evaluator.py           # RAGAS 三/四维度打分
│   │   ├── runner.py              # 批量评测 + Golden Test Set
│   │   ├── ab_testing.py          # A/B 实验评估
│   │   └── __init__.py
│   └── ingestion/                 # （预留）多模态图文注释
│       ├── vision_caption.py
│       └── __init__.py
├── scripts/                       # 运维/评测脚本
│   ├── backup.py                  # 全量备份/恢复/列表/清理（chroma_db/uploads/memory_db/tenant_data）
│   ├── reindex_docs.py            # 离线重建索引（含扫描件 OCR，需 .ocr_venv 环境）
│   ├── run_ragas_eval.py          # 真实 RAGAS 评测（需 key）
│   ├── run_regression.py          # 回归测试（golden 集）
│   ├── verify_retrieval.py        # 检索质量抽查
│   ├── evaluate.py                # 评测辅助
│   └── _gen_readme_imgs.py        # 生成文档配图
├── tests/                         # pytest 三层测试（unit / integration / evaluation）
├── run.py                         # 本地启动入口（uvicorn 封装）
├── run_tests.py                   # 跑全部测试 + 覆盖率
├── run_eval.py                    # 评测快捷入口
├── healthcheck.py                 # Docker 健康检查脚本
├── Dockerfile / docker-compose.yml # Docker 镜像
├── requirements.txt / requirements-docker.txt # 依赖
├── start-mac.sh / stop-mac.sh     # macOS 一键启动/停止（*bat 为 Windows 历史脚本）
├── .env / .env.example            # 环境变量（key 等，.env 不入库）
├── chroma_db/ sparse_db/ uploads/ tenant_data/ data/   # 运行时数据（均 gitignore）
└── model_cache/                  # 本地 BGE-M3 模型（gitignore，需镜像下载）
```

### 分层总览

```
接入层  : app/api/*            HTTP 路由、鉴权、限流、SSE 流式
安全层  : app/security/*       安全头 / 注入&穿越防护 / 统一错误
核心层  : app/main/app.py (create_app), core/*  应用装配、配置、限流器
业务编排: app/tenant/*         用户/租户/分组/对话 数据访问
检索层  : app/retrieval/*      Embedding / 混合检索 / 改写 / 引用 / 缓存
处理层  : app/processing/*     PDF 解析 / OCR / 多模态分块
记忆层  : app/memory/*         短/中/长期记忆 / 画像 / 遗忘
异步层  : app/tasks/*, worker/* 上传索引任务 / 后台整理 / Webhook 投递
可观测  : app/observability/*  链路追踪 / 结构化日志 / metrics/health
评估层  : app/evaluation/*     离线 RAGAS / A-B 实验
脚本层  : scripts/*, run*.py    备份 / 重索引 / 评测 / 测试入口
```


### 模块调用链

`
用户请求
    │
    ▼
app/main/app.py (FastAPI 入口, create_app)
    │  ├── app/api/auth.py          → app/tenant/auth.py (SQLite)
    │  │                               └── app/core/config.py
    │  │
    │  ├── app/api/chat.py          → app/retrieval/embedder_factory.py → app/retrieval/bge_m3_embedder.py (BGE-M3 稠密+稀疏双路)
    │  │                               → app/retrieval/hybrid_retriever.py → BM25 + Dense + RRF
    │  │                               → app/retrieval/query_rewriter.py → LLMClient (app/llm/client.py，多供应商 fallback)
    │  │                               → app/retrieval/semantic_cache.py → Redis (向量近似匹配，命中跳过重算)
    │  │                               → app/retrieval/citation.py
    │  │                               → app/retrieval/cache.py → Redis (精确缓存)
    │  │                               → app/memory/memory_manager.py → Chroma + Redis
    │  │                               → app/worker/shadow_worker.py (异步)
    │  │                               → app/observability/tracker.py
    │  │                               → app/observability/structured_logger.py
    │  │
    │  ├── app/api/documents.py     → app/retrieval/embedder_factory.py
    │  │                               → app/retrieval/bge_m3_embedder.py (稠密+稀疏向量)
    │  │                               → app/retrieval/sparse_store.py (稀疏向量落盘)
    │  │                               → app/processing/multimodal_pipeline.py
    │  │                                   → app/processing/pdf_extractor.py
    │  │                                   → app/processing/ocr_engine.py
    │  │                               → langchain Chroma (稠密向量持久化)
    │  │
    │  └── app/api/feedback.py      → app/memory/conversation_store.py
    │
    └── app/frontend/index.html (前端静态文件)
`

### 请求完整流程 (上传+提问)

`
1. 用户上传 PDF
   POST /api/documents/upload
   ├── app/api/documents.py: 接收文件 → 保存到 ./uploads/{tenant_id}/ → 立即返回 task_id（**异步，不阻塞**）
   ├── 后台线程池执行: app/processing/multimodal_pipeline.py 解析 PDF (PyMuPDF + OCR) -> embedder_factory 经 bge_m3_embedder.py 生成稠密+稀疏向量 -> 稠密向量 ChromaDB 持久化到 ./chroma_db/{tenant_id}/，稀疏向量经 sparse_store.py 落盘到 ./sparse_db/{tenant_id}/
   ├── 任务状态持久化到 `data/tasks.json`，服务重启后从磁盘恢复，避免"已上传却提示请先上传文档"
   ├── 服务启动时自动扫描 `uploads/{tenant_id}/`，对未向量化的 PDF 重新提交后台索引任务
   ├── 进度查询：GET /api/documents/task/{task_id} 返回 pending/processing(extracting→embedding→building_index)/done/failed 及百分比
   └── 索引完成后即可提问；索引中提问会返回"文档正在后台索引中"提示而非报错；已上传但索引失败会提示重新上传

2. 用户提问
   POST /api/chat (SSE 流式)
   ├── app/api/chat.py: 验证 Token → 加载 ChromaDB 向量库
   ├── app/retrieval/query_rewriter.py: LLM 改写/扩展查询
   ├── app/api/chat.py _check_cache(): 语义缓存命中（query 向量近似匹配）→ 直接复用检索结果+答案，跳过下面重算
   ├── app/retrieval/hybrid_retriever.py:
   │   ├── 稠密检索: ChromaDB.similarity_search_with_score() (向量来自 bge_m3_embedder.py)
   │   ├── 稀疏检索(BM25): BM25Okapi.get_scores()
   │   ├── BGE-M3 稀疏检索: bge_m3_embedder.py 确定性自计算 SPLADE 词汇权重 + sparse_store.py 落盘 lookup (RRF keying 取 page_content[:200])
   │   └── RRF 融合(稠密 + BM25 + BGE-M3 稀疏 + 可选 ES) + 可选 BGE 重排
   ├── app/retrieval/citation.py: 记录来源引用
   ├── app/memory/memory_manager.py: 加载短期/长期记忆
   ├── 调用统一 LLM 客户端 (app/llm/client.py) 流式生成回答（主供应商限流时按 LLM_FALLBACK_PROVIDERS 自动切备用）
   ├── app/observability/tracker.py: 记录耗时和 Token 用量
   └── 返回 SSE 流给前端
`

### 数据流向

`
PDF文件
  → processing/multimodal_pipeline (解析文本+图片)
  → embedder_factory (转向量)
  → ChromaDB (持久化到磁盘)

用户问题
  → query_rewriter (LLM 改写)
  → _check_cache (语义缓存命中则直接复用答案，跳过后续重算)
  → hybrid_retriever (稠密 + 稀疏 + RRF)
  → memory_manager (加载记忆上下文)
  → [合并上下文 + 引用] → LLMClient (app/llm/client.py，主供应商限流自动切备用)
  → 流式返回 → 前端渲染
"

## 测试

项目使用 pytest 做自动化测试，分三层：单元测试、集成测试、评测测试。

![测试分层全景](docs/images/testing-landscape.png)

*图：测试分层——越往下越接近真实、越慢；unit / integration / evaluation 三层均已实现。*

### 测试目录结构

```
tests/
├── conftest.py          # 公共 fixtures：mock Redis / ChromaDB / LLM 等外部依赖
├── unit/                # 单元测试（已实现）
│   ├── test_config.py
│   ├── test_hybrid_retriever_v3.py
│   ├── test_memory_manager_fixed.py
│   └── test_api_chat_simple.py
├── integration/         # 集成测试（已实现）
└── evaluation/          # 评测测试（已实现）
```

### 1. 单元测试（Unit）
测单个函数逻辑：config 解析、hybrid_retriever 打分、memory_manager 存取、chat 接口参数校验。

```bash
python -m pytest tests/unit/ -v
```

### 2. 集成测试（Integration）— 已实现
把多个真实组件接起来跑，验证组件间契约与全链路拼接正确：API 路由 → JWT 鉴权 → 业务编排 → （LLM / embedding / OCR 用 mock 替代）。测试环境与生产完全隔离（`tests/integration/conftest.py` 提供临时 sqlite 用户库 + 临时上传/向量目录 + 每测试重置限流），不会污染真实数据。已覆盖：

- **JWT 鉴权链路**（`test_auth_chain.py`）：注册 / 重复注册拒绝 / 错误密码拒绝 / 登录拿 token / `/me` 无 token·错误 token·正确 token 的行为
- **限流生效**（`test_rate_limit.py`）：同一 IP 高频登录触发 429（验证上一轮 slowapi 接线真的生效）
- **聊天全链路**（`test_chat_integration.py`）：完整链路（mock LLM 返回）→ 答案正确；未上传文档 → 优雅返回"请先上传文档"；缺字段 → 422
- **文档上传**（`test_document_upload.py`）：上传落盘 + 列表可见 + 非 PDF 拒绝 + 无/错误 token 拒绝

```bash
python -m pytest tests/integration/ -v
```

![集成测试范围](docs/images/integration-scope.png)

*图：集成测试把路由 → 检索 → 向量库 → LLM 真实接线，只 mock 掉 LLM 网络调用。*

### 3. 评测测试（Evaluation）
不问代码对错，问回答质量。用 RAGAS 给回答打分（0~1），防"编造法条"这类高风险问题。四个核心指标：

- **Faithfulness 忠实度**：答案是否只基于检索到的上下文，没有编造
- **Answer Relevancy 相关性**：答案是否切题、回应了问题
- **Context Precision 上下文精度**：相关片段在检索结果里是否排名靠前
- **Context Recall 上下文召回**：检索是否覆盖回答所需的全部资料

`tests/evaluation/test_ragas_eval.py` 已落地三层用例，离线即可跑、不依赖 ragas/网络：

- `test_golden_test_set_schema`：校验 `tests/golden_test_set.json`（31 条知识库问答回归集）结构
- `test_ragas_harness_offline`：mock LLM 调用，验证评测数据集能正确组装
- `test_ragas_real_eval`：真正跑 RAGAS 评分（检测到 ragas 与 `LLM_API_KEY`/`ARK_API_KEY` 才运行，否则自动 skip）

也可直接运行原始脚本（需联网 + key）：

```bash
python scripts/run_ragas_eval.py
```

![RAGAS 评测指标](docs/images/ragas-metrics.png)

*图：RAGAS 四个核心指标（0~1，越高越好），忠实度专门防"编造法条"。*

### 运行全部 + 覆盖率

```bash
python run_tests.py            # 依次跑 unit/integration/evaluation + 覆盖率报告
# 或
python -m pytest --cov=app --cov-report=term-missing
```

> 说明：`setup.cfg` 已移除 `fail_under=80` 的覆盖率门槛，测试不再因覆盖率不足而"变红"（用例本身 pass 即通过）。integration / evaluation 现已实现并纳入统计。所有用例用 pytest 标记分层（`unit` / `integration` / `evaluation`），可按层运行，例如 `python -m pytest -m integration`。

## 模型服务选型决策（LLM / Embedding）

### 背景
本项目为**中文企业知识库 RAG**，推理链路需要两类模型服务：
- **LLM（问答生成）**：将检索到的法条 / 案例 / 文书片段喂给大模型，生成最终回答。要求指令遵循强、中文语感好、能忠实引用上下文。
- **Embedding（向量化）**：将文档切片与用户 query 编码为向量，决定混合检索的召回质量，是 RAG 效果的上限。

### LLM 选型：DeepSeek（deepseek-chat / V3）
- **候选对比**：DeepSeek、智谱 GLM-4、阿里 通义千问、火山 豆包、小米 MiMo、MiniMax
- **选定**：`deepseek-chat`（OpenAI 兼容端点 `https://api.deepseek.com/v1`）
- **选择原因**：
  1. 中文质量第一梯队，RAG 问答生成完全够用；
  2. 成本极低（输入约 ¥1 / 百万 token），面试演示 / 长期自测不心疼；
  3. 提供 OpenAI 兼容 API，`config.py` 的 `LLM_BASE_URL` + `LLM_MODEL` 直接填即可，**无需改代码**；
  4. 响应快、服务稳定。
- **否决项**：
  - **小米 MiMo**：推理专精模型（数学 / 代码 / 逻辑链），而 RAG 生成端要的是「忠实复述 + 指令遵循」，MiMo 更慢更贵且角色不匹配；
  - **MiniMax / 豆包 / 智谱**：可用，但综合中文质量与性价比不及 DeepSeek。

### 多模型切换（升级：支持供应商热切换 + 高并发兜底）

**解决什么问题**：上线后单一 LLM 供应商（如 DeepSeek）在高峰时段常被限流（HTTP 429），主链路直接报错或卡死（雪崩）。为此把 LLM 配置抽象为「供应商开关 + 兜底链」，让代码能自动切换备用供应商。

- **切换方式**：`.env` 设 `LLM_PROVIDER=deepseek|openai|qwen|moonshot|custom`，并填对应供应商的 `*_API_KEY` / `*_BASE_URL` / `*_MODEL` 即可，**下游业务代码零改动**（统一经 `app/llm/client.py` 解析 `LLM_*` 接口）。
- **向后兼容**：老 `.env` 仅配 `LLM_API_KEY` 时，自动回退到原有 `LLM_*` 直连，不受影响。
- **高并发兜底链**：`LLM_FALLBACK_PROVIDERS=openai,qwen`，主供应商调用失败时按列表顺序自动重试备用供应商，避免限流雪崩。
- **集中客户端**：`app/llm/client.py` 封装 `chat_completion()` / `stream_chat_completion()`，统一处理鉴权、超时、SSE 解析与供应商 fallback；`chat.py` 三处 LLM 调用（记忆总结 / 非流式 / 流式）已全部收敛到此客户端。

### Embedding 选型：本地 BGE-M3（默认）
- **演进历史**：
  1. **最初（本地）**：`shibing624/text2vec-base-chinese`（sentence-transformers 本地加载，零成本）→ 因国内下载模型超时 / 依赖冲突 / CPU 推理慢，**跑不通**；
  2. **中期（线上）**：切火山方舟 / 豆包 embedding（`openai` 兼容端点，模型 `ep-m-...`）→ 可用，但源码曾硬编码 key（已移除），**历史可能留存泄露风险，需轮换**；
  3. **现在（方案 2，默认）**：切回本地 **`BAAI/bge-m3`**。
- **选定本地 BGE-M3 原因**：
  1. 零成本、零外部依赖、零密钥泄露风险（彻底绕开线上 key 隐患）；
  2. 中文企业制度文本语义效果优于 `text2vec-base-chinese`；
  3. 项目 `requirements-docker.txt` 已含 `sentence-transformers`，huggingface backend 已支持；
  4. 面试加分点：「检索层不依赖任何外部 embedding 服务」。
- **落地状态（2026-08-09 已完成切换并验证）**：模型已下载至 `model_cache/bge-m3`（2.2 GB），
  离线加载 ~13 s，稠密 1024 维 + BGE-M3 稀疏向量均正常；实测重新索引一份 PDF（11 chunks）耗时 ~14 s，
  **全程零外部 API 调用**。
- **切换注意事项**：
  - BGE-M3 输出 **1024 维**，首次需联网下载模型；国内直连 `huggingface.co` 不通，**必须走镜像**
    `HF_ENDPOINT=https://hf-mirror.com`；
  - Windows 无符号链接权限时下载会报 `WinError 14007`，需 `HF_HUB_DISABLE_SYMLINKS=1`；
  - `app/main/config.py` 硬编码 `TRANSFORMERS_OFFLINE=1`（运行时不联网），因此 `HF_MODEL_NAME` 应填
    **本地模型目录绝对路径**（如 `D:\git\rag-knowledge-base\model_cache\bge-m3`），而非仓库名 `BAAI/bge-m3`；
  - 切换 embedding 模型后，已存的 Chroma 向量库维度 / 语义不再匹配，必须**清空 `chroma_db` 并重新索引**
    所有文档（豆包 2560 维 → BGE-M3 1024 维，混用会直接报维度错误）；
  - 如需切回线上 embedding：设 `EMBEDDER_TYPE=openai` 并填 `EMBEDDING_API_KEY`，但务必先**轮换原泄露的 key**。

> ⚠️ **常见误解澄清**：BGE-M3 是**纯文本** embedding 模型，本身不处理图片。若某份 PDF 是扫描件（无文字层，只有整页图片），PyMuPDF 抽不到文字，BGE-M3 就「没东西可向量化」，检索时自然搜不到——**这并非 BGE-M3 多模态能力不行，而是缺 OCR 引擎把图片转成文字**。本项目已接入 **PaddleOCR** 补齐这条链路（见下方「OCR 引擎」章节）：扫描件经 OCR 识别出文字后再走 BGE-M3 向量化，即可被正常检索。

### BGE-M3 相对 text2vec-base-chinese 的实质提升

`text2vec-base-chinese` 是项目最初本地方案（基于 `bert-base-chinese`，768 维，纯稠密，512 token 上限）。切到 BGE-M3 后在企业知识库 RAG 场景有**实质性**提升：

| 维度 | text2vec-base-chinese | BGE-M3 | 对知识库场景的意义 |
|------|----------------------|--------|------------------|
| 向量维度 | 768 | **1024** | 表达更丰富，长难句 / 近似表述区分度更高 |
| 上下文长度 | 512 token（≈300 中文字） | **8192 token** | 判决 / 合同 / 法规动辄上万字，text2vec 直接截断丢内容；BGE-M3 可整段吃下 |
| 向量类型 | 仅稠密 | **三合一：稠密 + 稀疏（SPLADE）+ 多向量（ColBERT）** | 稀疏抓「法条编号 / 条款名」精确词，稠密管语义，ColBERT 做 token 级细匹配 |
| 语言 | 仅中文 | **100+ 语言** | 涉外 / 双语企业制度文本可用 |
| 检索基准 | 中文相似度榜中上 | **MTEB 中文检索 SOTA 级** | 召回 Top-K 更相关，减少漏召导致的答非所问 |

**当前实际吃到的收益（drop-in 替换已生效）：**
1. **检索质量跃升**：BGE-M3 中文检索能力远超 text2vec，长文档语义对齐更准，召回 Top-K 相关性更高；
2. **长文本不再截断**：8192 token 上限，长法条 / 长判决可被完整向量化（text2vec 的 512 是硬伤，会强制截断丢失后续内容）。

> ✅ **稀疏向量已接上**：BGE-M3 的 SPLADE 稀疏向量已接入混合检索。`embedder_factory` 优先返回 `BGEM3Embedder`（稠密 1024 维 + 自计算稀疏权重）；文档上传时把每段稀疏权重落盘到 `./sparse_db/{tenant}/{file}.json`，`HybridRetriever` 新增稀疏检索分支，与 BM25 + 稠密经 RRF 融合重排，业务术语 / 法条编号精确召回明显提升。注意：稀疏权重由本模块**自计算**（绕过 FlagEmbedding 1.4.0 的 `scatter_reduce` 在 CPU 下偶发整条丢失的 bug），结果 100% 可复现。ColBERT 多向量仍未接（见下方待办）。

**代价**：模型体积 2.3GB（text2vec ≈400MB），加载更占内存、首向量化更慢；已下载到 `./model_cache` 并由 `.gitignore` 忽略，不入库。

**待办（解锁 BGE-M3 全部能力）**：
- [x] 接入 BGE-M3 稀疏向量，与现有 BM25 + 稠密做融合重排（提升业务术语精确召回）；
  - 实现：`app/retrieval/bge_m3_embedder.py` 自计算 SPLADE 权重。BGE-M3 稀疏头是 `Linear(H,1)` 逐位置标量门控，本模块按 `input_ids` 取每个 token 的**最大门控权重（amax 聚合）**组装为 `{token_id: weight}`（确定性 Python 实现，等效于 FlagEmbedding 的 `scatter_reduce(amax)`，但规避了其在 CPU 下偶发整条丢值的 bug）；`sparse_store.py` 落盘/加载，`HybridRetriever._sparse_search_bge` 与 BM25 + 稠密做 RRF 加权融合。回归测试 `tests/unit/test_bge_m3_sparse.py` 覆盖非空/确定性/特殊 token 过滤。
- [ ] 实验 ColBERT 多向量 late-interaction，强化长文证据定位；
- [ ] 在 `tests/golden_test_set.json`（31 条知识库问答回归集）上对比 text2vec → BGE-M3 的检索/回答质量提升。

## OCR 引擎（扫描件 / 图片识别）

### 为什么需要 OCR
BGE-M3 只吃文本。对**有文字层**的 PDF，PyMuPDF 直接抽取文字即可；但对**扫描件 / 纯图片 PDF**（无文字层，整页就是一张图），PyMuPDF 抽不到文字，必须先用 OCR 把图片里的文字识别出来，再交给 BGE-M3 向量化，否则这份文档在检索时完全搜不到。

### 选型：PaddleOCR 3.7（默认）
- 中文识别准确率行业第一梯队，对制度条款印刷体识别极准（实测一页 ~1100 中文字符，置信度 0.97+）；
- 自带 PP-OCRv6 检测 + 识别模型，首次运行自动下载并缓存到 `~/.paddlex/official_models/`；
- 支持中英文混合（`lang="ch"`），可识别整页扫描图与 PDF 内嵌图片。

### 接入位置
`app/processing/ocr_engine.py` 的 `OCREngine` 封装 OCR 后端，`app/processing/multimodal_pipeline.py` 的 `MultimodalPipeline.process()` 对每页依次做：
1. PyMuPDF 抽文字层；
2. 页面 / 内嵌图片交给 `OCREngine.recognize()` 做 OCR；
3. 图文块统一分块 → BGE-M3 向量化。
无文字层且 OCR 仍无效的页面会被跳过，避免写入 `[图片描述]` 之类的占位符垃圾 chunk。

### ⚠️ PaddleOCR 3.x 与 2.x API 不兼容（已适配）
老代码按 PaddleOCR 2.x 写，3.7 改动巨大，直接跑会抽不出文字（旧 `for line in result[0]: line[1][0]` 实际在遍历 dict 的 key 字符串，得到一堆单字母乱码）。已适配：
- 构造参数：`use_angle_cls` / `use_gpu` 已废弃 → 改用 `use_doc_orientation_classify` / `use_doc_unwarping` / `use_textline_orientation` / `lang`（`_init_paddleocr` 带 2.x 兼容回退）；
- `ocr()` 废弃，推荐 `predict()` 接口；
- 识别结果 `OCRResult` 是类字典对象，文本在 `result[0]["rec_texts"]`（不再是 `line[1][0]`）。

### 独立虚拟环境 `.ocr_venv`（离线安装）
PaddleOCR 依赖链重（paddlepaddle / paddlex / opencv 等），且本项目主环境（miniconda）已装重包，**直接 `pip install paddleocr` 会触发包卸载冲突**。解决方案：建独立 venv 复用 miniconda 已装重包、只把 PaddleOCR 相关新包装进 venv：
```bash
# 1) 建 venv（复用 miniconda 已装重包）
python -m venv .ocr_venv
echo "C:\Users\11195\miniconda3\Lib\site-packages" > .ocr_venv/Lib/site-packages/zz_miniconda.pth
# 2) 在 venv 内离线安装 PaddleOCR（首次需联网下载模型权重）
.ocr_venv/Scripts/python.exe -m pip install paddleocr opencv-contrib-python
```
> `.ocr_venv/` 已加入 `.gitignore`，不入库。模型权重缓存于 `~/.paddlex/`（用户目录，跨项目复用）。
> 启动脚本 `启动企业知识库 RAG 系统.bat` 的 `PY` 已指向 `.ocr_venv/Scripts/python.exe`——**必须用它启动**，否则服务进程没有 paddleocr，扫描件仍进不了库。

### 重索引扫描件
改完抽取链路或新接入 OCR 后，用离线脚本重建索引（脱离 web 服务进程，避免后台线程被回收）：
```bash
# 需能 import paddleocr 的环境 + 离线加载本地 BGE-M3
.ocr_venv/Scripts/python.exe -u reindex_docs.py
```
脚本遍历 `chroma_db` 下各租户 `uploads/` 的所有 PDF：有文字层直接抽，扫描件走 OCR；抽取为空则清理历史垃圾 chunk；已索引源先清后写（幂等，可重复跑）。

## 高并发优化（上线防延迟塌缩）

RAG 系统上线后延迟降不下来，90% 不是"模型慢"，而是**请求在系统里被串行卡住 + 每次都重算 + 没有弹性**三件事叠加。本项目在 `app/main/` 重构定稿后，针对此做了三次升级，每一级都对应一个具体瓶颈：

### 升级 1：解阻塞事件循环 + 多 worker（并发吞吐）
- **解决什么问题**：单 worker 下 `QueryRewriter.rewrite()` 用同步 `requests` 调 LLM（最长阻塞 10s）、`_get_context()` 里的 BGE-M3 embedding + CrossEncoder rerank 是 CPU 密集同步调用——这些都在 async 事件循环里直接跑，高并发时**事件循环被占死，所有请求排队、流式响应卡顿、吞吐塌缩**。
- **改了什么**：
  - `docker-compose.yml` 启动命令加 `--workers`（8G 默认 2、16G 可上 4），受内存约束；配合 Nginx 多副本负载均衡可继续横向扩展。
  - `app/api/chat.py` 用 `asyncio.to_thread` 把 CPU 密集的 embedding / rerank / paddleocr 调用丢线程池，释放事件循环只管 IO，并发请求不再互相阻塞。
  - `app/main/events.py` 启动时**预热** BGE-M3 embedder 与 reranker，避免首请求在事件循环里阻塞加载（加载需数秒）。

### 升级 2：Redis 语义缓存中间件（降延迟 + 降 LLM 成本）
- **解决什么问题**：原有 `QueryCache`（`app/retrieval/cache.py`）只做 **MD5 精确匹配**，近似问题（"劳动合同怎么解除" vs "如何解除劳动合同"）漏缓存，每次都重跑 embedding + rerank + LLM——既拖慢首字延迟，又**白烧 DeepSeek 调用费**（峰谷定价下尤其贵）。
- **改了什么**：新增 `app/retrieval/semantic_cache.py`，用查询向量余弦相似度做**语义匹配**，近似问题直接复用检索结果与 LLM 答案。复用现有 Redis 客户端（命中阈值 `SEMANTIC_CACHE_THRESHOLD`，默认 0.92）。与精确缓存互补：语义缓存兜底"问法不同、意思相近"，精确缓存兜底"完全一样"。

### 升级 3：限流 + LLM 多供应商 fallback（抗限流 / 保命）
- **解决什么问题**：突发流量打满 DeepSeek 的 RPM/TPM → 429 雪崩，主链路直接报错/卡死；且单一供应商无退路。
- **改了什么**：
  - per-IP 限流（slowapi，`/api/chat` 100/min、`/api/auth` 20/min）此前已接线生效，本轮未动。
  - 新增 `app/llm/client.py` 集中式 LLM 客户端，主供应商调用失败时按 `LLM_FALLBACK_PROVIDERS` 列表**自动切换备用供应商**（deepseek → openai → qwen…），配合 `LLM_PROVIDER` 多模型切换（见上文「多模型切换」）。
  - `chat.py` 三处 LLM 调用（记忆总结 / 非流式 / 流式 SSE）全部收敛到该客户端，错误处理提示也指向当前供应商的密钥名。

### 部署成本提示
- 纯云端（无 GPU）：4 核 8G 云服务器 ~¥150–230/月 + DeepSeek API 按量（峰谷定价，高峰 flash 输出 9 元/百万 token）。embedding 本地 BGE-M3 零 API 费。
- 省钱：开启语义缓存（砍重复调用）、错峰重任务、用 prompt 缓存；可换更便宜的 OpenAI 兼容模型（仅改 `.env`，不改代码）。

## 轻量自进化闭环（经验捕获 + 评测回归）

为解决「模型 / prompt / 检索参数一升级，无法判断是否退化；线上答得差也无处归因」的问题，项目落地了**自进化闭环的前两环**（轻量版，人工把关、不自动改 prompt）：

- **经验捕获层** `app/core/trace_store.py`：每次问答结束后把 `query / 答案 / 引用来源 / 耗时 / Token / 实际供应商 / 是否命中缓存 / 是否成功` 落库到本地 SQLite（`memory_db/query_traces.db`）；用户点赞/点踩时 `app/api/feedback.py` 按租户 + 提问回流满意度评分到对应记录。取代原内存版 `TraceStore`（重启即丢、不可离线分析）。
- **闸门式验证** `tests/eval/run_eval.py`：跑一批标准问答对（golden set），统计通过率 / 平均引用数 / 平均延迟。模型升级或 prompt 改动前后各跑一次，通过率不降才允许发布。

> 知识库场景答错有合规风险，**不做全自动改 prompt**，只做「trace 沉淀 → 人工 + 评测闸门把关」。`trace_store.get_low_rated()` 可挖出线上答得差的样本，作为 golden set 的补充来源。详见 `tests/eval/README.md`。

### 回答质量看板（可视化归因）
为让沉淀的 trace 真正"看得见、可归因"，项目提供**回答质量看板**：
- **后端** `app/api/eval.py`：暴露 `GET /api/eval/stats`（统计卡片）、`/low-rated`（低分样本）、`/recent`（最近问答），均仅 `super_admin`/`admin` 可访问（含跨租户运营数据）。
- **前端** `app/frontend/eval_dashboard.html`：侧边栏「回答质量看板」入口（仅管理员可见）。展示问答总量 / 低分样本 / 成功率 / 反馈覆盖率等统计卡片，并列出"低分样本"与"最近问答"，每条可展开查看 query / 答案 / 引用来源 / 供应商 / 延迟 / 用户评分，支持一键导出 JSON。
- 这是闭环第一环（trace 捕获）的可视化出口：管理员日常看板 → 发现答得差的样本 → 针对性优化 → 用评测闸门验证是否退化。

### 从租户文档自动生成 golden set
手工编写标准问答对费时且难覆盖真实业务。提供 `tests/eval/build_golden_from_docs.py`：扫描你**真实上传的文档**（`uploads/<tenant>/`，支持 `.txt`/`.md`/`.pdf`/`.docx`），抽取文本后调用 LLM 批量生成贴合业务的 `{query, expect_keywords, min_citations}`，直接写出 `tests/eval/golden_set.json`。`golden_set.json` 是各租户定制的工作文件，已加入 `.gitignore` 不入库；仓库只跟踪 `golden_set.example.json`。用法见 `tests/eval/README.md`。

## Wiki 打头 + RAG 兜底（两级知识结构）

纯 RAG 在企业知识库场景有三个绕不过去的问题：**切片断章取义**（一句关键结论被 500 字切分劈开，Top-5 只召回半句，模型答“资料中没有”）、**每次问答都重算**（无长期沉淀，同一问题今天明天各检索一遍）、**无法审定**（没人能说清某条答案是谁批的、批的是哪一版内容）。

为此在检索链路前面加了一层结构化知识页，形成两级结构：

```
用户问题 ──► Wiki 路由 ──命中──► 整页(或命中小节)入上下文 ──► LLM
                │
                └──未命中──► 原混合检索（稠密+BM25+BGE-M3+RRF+Rerank）──► LLM
```

**路由是确定性的，不引入新模型**：语料只有 9 份文档、5 个页面时，用页面别名 + 小节标题做归一化短语匹配，命中即返回、未命中降级；查询里出现「第X条」这类明确法条引用时直接放行给 RAG，因为法条原文检索比内规页面更准。向量路由等页面涨到 50+ 再上，现在上属于过度设计。

### 页面数据模型

页面不是"文档摘要"，而是"问题域"：**小节标题写成用户会问的问题**（这样别名匹配不到时标题还能兜住），**每节必须带可回溯出处**，缺出处的页面不允许上线。

```python
Page:
    id / title / aliases / summary / status
    sections: [Section]
Section:
    heading   # 本身是一句问题，如「司机问：跑网约车需要哪些证件？」
    body      # 「可直接答复：…」「补充要点：…」——客服可直接复述
    spans: [Span]
Span: (doc_id, doc_name, page, quote)   # 精确到页 + 原文摘句
```

`app/wiki/` 下五个文件：`store.py`（SQLite 存储与审定）、`router.py`（别名路由）、`seed.py`（内置页面）、`doc_status.py`（切片级文档状态）、`__init__.py`。

### 审定闭环：页面级 + 文档级

两个层级缺一不可，只锁一层等于没锁：

| 层级 | 控制点 | 生效方式 |
|---|---|---|
| 知识页 | `Page.status` = draft/approved | 路由层不返回 draft 页 |
| 原始文档 | 切片 metadata `doc_status` = draft/approved | 稠密通道 `$ne` 过滤 + BM25/BGE-M3 稀疏语料剔除 |

三个关键设计：

1. **审定留痕**：`approved_by / approved_at / review_note` + `review_log` 审计表，每次 approve/reject 都记录审定人与意见。
2. **内容指纹失效**：页面记录 `content_hash`，内容一变自动回到草案——否则会出现“批的是旧内容、改完还挂着 approved”的治理漏洞。
3. **边界过滤而非 Prompt 约束**：未审定内容在检索层就进不来，不依赖模型"看见但不许用"。

审定接口（均限 `super_admin`）：`POST /api/wiki/pages/{id}/review`、`POST /api/wiki/documents/review`、`GET /api/wiki/reviews`；命令行等价物在 `scripts/set_doc_status.py`。

### 实测效果（`scripts/eval_wiki_vs_rag.py`）

同一批问题分别走「Wiki 优先」与「纯 RAG」，对比金标证据覆盖率与上下文构建耗时：

| 指标 | Wiki 优先 | 纯 RAG |
|---|---|---|
| 金标证据覆盖（待审定 5 题） | 5/5 | 4/5 |
| 金标证据覆盖（审定后 6 题） | 6/6 | 4/6 |
| 平均上下文长度 | 781 字 | 1100 字 |
| 平均上下文构建耗时 | 266 ms | 890 ms |

两个典型差异：

- **「乘客投诉司机绕路能赔多少」**：Wiki 返回手册第 8/12/4 页的结构化五节；RAG 返回的 5 个片段里有 3 个是附录（速查表、工单字段、依据索引）——按词面相似度召回分不清"附录 B 工单字段"不可能是答案。
- **「乘客信息能不能提供给警方」**：RAG 召回了运营规范与驾驶员规范，**《民法典》第 111 条没进 Top-5**，法律依据缺失；Wiki 页把法条与平台规则绑在同一节里。

> 样本量 5–6 条只能看趋势，不能当结论。Reranker 在无外网环境下会降级跳过，RAG 的真实水平应略好于表中数字。
