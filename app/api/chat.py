"""
chat.py —— 对话(RAG 检索增强生成)API 模块

【作用与功能】
本模块实现 rag-knowledge-base 的核心问答接口，串联完整的 RAG 流程:按租户构建检索管道 ->
查询重写 -> 缓存命中检查 -> 混合检索(稠密+稀疏)与重排序 -> 引用追踪 -> 构建提示词 ->
调用大模型生成回答，并以 SSE 流式或非流式方式返回，同时维护缓存与长期记忆。

【主要组成】
- `chat` / `chat_stream`:问答主入口，支持流式与非流式两种响应模式。
- `_build_pipeline`:按租户组装嵌入器、向量库、重写器、缓存、引用追踪、记忆等组件。
- `_get_context`:执行混合检索、重排序并拼接记忆与历史构建最终提示词。
- `_handle_streaming_response` / `_handle_non_streaming_response`:分别处理两种响应模式。
- `_handle_post_processing`:回答后写缓存、更新记忆并触发后台任务。
- `call_llm`:异步调用大模型(用于记忆总结等非流式场景)。
- `_get_memory` / `_get_reranker`:记忆系统与重排序器的缓存/单例获取。

【适用场景】
- 前端用户向系统提问，系统基于已上传知识库文档检索并生成带引用的回答。
- 长文档问答、多轮对话记忆、基于分类实验的回答质量对比(A/B 测试)。

【依赖关系】
- 上游调用方:前端对话页。
- 下游依赖:app.retrieval(嵌入/检索/重写/引用/缓存)、app.memory、app.worker.shadow_worker、
  app.observability(日志/追踪/监控)、app.tasks.task_store、Chroma 向量库、langchain_community。
"""

import os, sys, json, requests, time, re

# 将项目根目录添加到系统路径中，以便能够正确导入项目内的其他模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse 
from pydantic import BaseModel
from typing import List, Optional
from app.retrieval.embedder_factory import create_embedder
from app.retrieval.hybrid_retriever import HybridRetriever, Reranker
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.citation import CitationTracker, Source
from app.retrieval.cache import QueryCache
from app.retrieval.semantic_cache import SemanticCache
from app.memory.memory_manager import MemorySystem
from app.worker.shadow_worker import get_worker
from app.observability.tracker import TraceContext
from app.observability.structured_logger import StructuredLogger
from app.observability.monitoring import record_query
from langchain_community.vectorstores import Chroma
import app.core.config as cfg  # 从.evn中加载配置，便于后期维护
from app.core.trace_store import save_query_trace  # 问答 trace 持久化(轻量自进化闭环-经验捕获)
from app.api.auth import get_user_from_token, require_user
import asyncio
import threading
from typing import Dict, Any, AsyncGenerator
from types import SimpleNamespace
from fastapi import Request
from app.core.limiter import limiter
from app.llm.client import complete_chat, stream_chat, LLMHardError, LLMAllFailed
from app.tasks.task_store import get_active_task_for_tenant, has_failed_task_for_tenant

# 初始化结构化日志记录器，标识为 "chat" 模块
_log = StructuredLogger("chat")

def _validate_config():
    """
    验证必要的配置项是否存在。
    
    该函数会检查全局配置对象 cfg 中是否包含运行系统所必需的关键配置项(如大模型URL、密钥等)。
    如果发现缺失项，会记录错误日志并抛出 ValueError 异常。
    
    异常:
        ValueError: 当缺少必要的配置项时抛出，错误信息中包含所有缺失的配置项名称及描述。
    """
    # 定义必需的配置项字典:键为配置属性名，值为该配置的中文描述
    required_configs = {
        'LLM_BASE_URL': '大语言模型API基础URL',
        'LLM_API_KEY': '大语言模型API密钥',
        'LLM_MODEL': '大语言模型名称',
        'CHROMA_PERSIST_DIR': '向量数据库持久化目录'
    }
    
    missing_configs = []
    # 遍历检查每个必需配置是否存在于 cfg 对象中且不为空
    for config_key, config_desc in required_configs.items():
        if not getattr(cfg, config_key, None):
            missing_configs.append(f"{config_key}({config_desc})")
    
    # 如果存在缺失配置，则抛出异常
    if missing_configs:
        error_msg = f"缺少必要的配置项: {', '.join(missing_configs)}"
        _log.error(error_msg)
        raise ValueError(error_msg)

# 在应用启动时验证配置
try:
    _validate_config()
    _log.info("配置验证通过")
except ValueError as e:
    _log.error(f"配置验证失败: {str(e)}")
    # 可以选择在这里退出程序，或者设置默认值
    # sys.exit(1)

# 创建一个API路由器，用于处理与聊天相关的API请求，统一添加 /api/chat 前缀
router = APIRouter(prefix="/api/chat", tags=["chat"]) 


class ChatRequest(BaseModel):
    """
    聊天请求的数据模型。
    
    用于验证和解析前端发送过来的聊天请求数据。
    
    属性:
        message (str): 用户当前输入的提问消息。
        history (Optional[List[dict]]): 历史对话记录列表，默认为空列表。
        stream (Optional[bool]): 是否开启 SSE (Server-Sent Events) 流式输出，默认为 True。
        conversation_id (Optional[int]): 所属会话 id，用于把短期/中期记忆隔离到单个会话。
    """
    message: str
    history: Optional[List[dict]] = []
    stream: Optional[bool] = True  
    conversation_id: Optional[int] = None


# 内存系统缓存字典，按租户ID缓存 MemorySystem 实例，避免重复初始化
_memory_cache = {}
# 创建一个线程锁对象，用于在多线程环境下保护 _memory_cache 的并发读写
_cache_lock = threading.Lock() 


async def call_llm(prompt: str) -> str:
    """
    异步调用大语言模型 API(用于记忆总结等非流式后台任务),带供应商故障转移。

    参数:
        prompt (str): 发送给大模型的提示词。

    返回:
        str: 大模型生成的文本响应；调用失败(网络/鉴权/全部供应商不可用)则返回空字符串 ""。
    """
    try:
        return await complete_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
    except Exception as e:
        _log.error(f"LLM 调用失败(记忆总结): {str(e)}")
        return ""


def _get_memory(tenant_id, embedder, session_key: str = ""):
    """
    获取或创建指定会话的记忆系统实例。

    记忆实例按「租户 + 会话键」缓存:此前只按租户缓存，且 session_id 构造后
    固定不变，导致同一租户下所有会话共用短期记忆与摘要、互相污染。
    长期记忆仍落在租户级 Chroma 集合中(跨会话共享)，短期/中期记忆按会话隔离。

    参数:
        tenant_id (str): 租户唯一标识符。
        embedder: 嵌入模型实例，用于记忆系统的向量化处理。
        session_key (str): 会话键(用户 + 会话 id)，为空时退化为按租户共享。

    返回:
        MemorySystem: 该会话对应的记忆系统实例。
    """
    global _memory_cache
    cache_key = f"{tenant_id}:{session_key}" if session_key else tenant_id
    memory_system = _memory_cache.get(cache_key)
    if memory_system is None:
        # 定义该租户记忆数据的持久化存储路径
        pd = os.path.join("memory_db", tenant_id)
        # 初始化记忆系统，并把 session_id 固定为会话键，重启后仍能命中同一份 Redis 记忆
        memory_system = MemorySystem(
            embedding_model=embedder,
            persist_dir=pd,
            tenant_id=tenant_id,
            session_id=cache_key,
        )
        with _cache_lock:
            # 简单 FIFO 上限:按会话缓存后条目会变多，避免无限增长
            if len(_memory_cache) >= 256 and _memory_cache:
                _memory_cache.pop(next(iter(_memory_cache)))
            _memory_cache[cache_key] = memory_system
    return memory_system


# 可以在应用启动时添加一个定时任务
async def cleanup_cache():
    """
    定期清理内存缓存的后台任务。
    
    该协程函数会无限循环运行，每隔 2 小时触发一次清理操作，
    用于释放长时间未被使用或过期的内存资源。
    """
    while True:
        await asyncio.sleep(7200)  # 每2小时清理一次
        # 检查缓存对象是否具备 cleanup 方法，如果有则执行
        if hasattr(_memory_cache, "cleanup"):
            _memory_cache.cleanup()


# 全局重排序器实例
_reranker = None
# 创建一个锁对象，用于保护 _reranker 的单例实例化过程
_reranker_lock = threading.Lock() 

def _get_reranker():
    """
    获取重排序器单例实例。
    
    使用双重检查锁机制确保在多线程环境下只创建一次 Reranker 实例。
    
    返回:
        Reranker: 重排序器实例。
    """
    global _reranker
    with _reranker_lock:
        if _reranker is None:
            _reranker = Reranker()
        return _reranker


# 进程内推理锁:序列化 BGE-M3 嵌入与 CrossEncoder 重排序等 CPU 密集的模型推理。
# 背景:下面把 _get_context 丢进 asyncio.to_thread 的线程池并发执行,其内部会调用
# 进程内单例的 BGE-M3 / CrossEncoder 模型做 forward。多线程并发推理共享同一份模型
# 权重存在线程安全隐患,故用本锁把「真正的模型推理」串行化,既保证安全,又释放了
# 事件循环去处理连接/流式 IO(事件循环不再被阻塞)。
_infer_lock = threading.Lock()


def _try_wiki_context(message, pipeline, history, mem_ctx=None):
    """
    用原始问句尝试命中知识页，命中则直接构建提示词。

    必须在查询改写之前调用:别名是面向用户说法的字面匹配，
    改写后的规范表述反而会让别名命中率下降;命中后也无需检索与精排。

    参数:
        message (str): 用户原始输入。
        pipeline (SimpleNamespace): 当前请求的处理管道组件集合。
        history (list): 对话历史。
        mem_ctx (str, optional): 已取好的记忆上下文，避免重复查询。

    返回:
        str | None: 命中时返回构建好的提示词，未命中返回 None。
    """
    wiki_router = getattr(pipeline, "wiki", None)
    hit = wiki_router.route(message) if wiki_router else None
    if not hit and wiki_router:
        # 短语未命中时做语义兜底:别名无法穷举用户说法。
        # 嵌入模型不可用/加载失败时降级为 RAG，不能因为兜底路由把整个问答打成 500
        try:
            hit = wiki_router.route_semantic(message, getattr(pipeline, "embedder", None))
        except Exception as e:  # noqa: BLE001
            _log.warning(f"Wiki 语义路由失败(降级到 RAG): {e}")
            hit = None
    if not hit:
        return None

    if mem_ctx is None:
        mem_ctx = pipeline.mem.get_context(message) if pipeline.mem else ""
    pipeline.ct.add_source(Source(
        document_id=hit.page.id,
        filename=f"{hit.page.title}（内部知识页）",
        content=hit.render(),
    ))
    return _build_prompt(pipeline.ct.format_context(), mem_ctx, history, message)

def _retrieve_context(query, pipeline, message, history, tenant_id=None):
    """
    在进程内推理锁保护下执行检索 + 重排序,供 asyncio.to_thread 调用。

    仅把模型推理(嵌入/重排)锁住;检索前后的内存/IO 操作仍可在锁外并发。
    """
    with _infer_lock:
        return _get_context(query, pipeline, message, history, tenant_id)

def _build_pipeline(tenant_id: str, session_key: str = ""):
    """
    构建当前请求的 RAG (检索增强生成) 处理管道。
    
    根据租户ID初始化并组装所需的各个组件:嵌入器、向量库、查询重写器、缓存、引用追踪器和记忆系统。
    
    参数:
        tenant_id (str): 租户唯一标识符，用于隔离不同租户的数据。
        session_key (str): 会话键(用户 + 会话 id)，用于把短期/中期记忆隔离到单个会话。
        
    返回:
        SimpleNamespace: 包含所有管道组件对象的命名空间；构建异常时返回 None。
            向量库目录尚不存在时 vector_store 为 None，但 Wiki 路由仍可用。
    """
    try:
        # 1. 创建文本嵌入器
        embedder = create_embedder()
        # 2. 获取或创建记忆系统(按会话隔离短期/中期记忆)
        mem = _get_memory(tenant_id, embedder, session_key) if embedder else None
        # 3. 构建租户专属的向量库持久化路径
        persist_dir = os.path.join(cfg.CHROMA_PERSIST_DIR, tenant_id)
        # 向量库未建立时不中断管道:Wiki 知识页不依赖向量库，仍应可回答
        vector_store = (
            Chroma(embedding_function=embedder, persist_directory=persist_dir)
            if os.path.exists(persist_dir) else None
        )
        
        # 4. 初始化各个管道组件
        query_rewriter = QueryRewriter(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL)
        cache = QueryCache(cache_dir=os.path.join("cache", tenant_id))
        # 语义缓存:基于 Redis,用查询向量近似匹配复用答案(近似问题也能命中)
        semantic_cache = SemanticCache(cfg.REDIS_URL, tenant_id)
        citation_tracker = CitationTracker()

        # 5. Wiki 知识页路由(第一级):初始化失败时降级为纯 RAG,不影响问答
        wiki_router = None
        if os.getenv("WIKI_ENABLED", "1") == "1":
            try:
                from app.wiki.router import WikiRouter
                from app.wiki.store import get_store
                wiki_router = WikiRouter(get_store(), tenant_id)
            except Exception as e:  # noqa: BLE001
                _log.warning(f"Wiki 路由初始化失败(降级为纯 RAG): {e}")

        # 6. 将组件打包为 SimpleNamespace 返回，方便通过属性访问
        return SimpleNamespace(
            embedder=embedder,
            vector_store=vector_store,
            qr=query_rewriter,
            cache=cache,
            semantic_cache=semantic_cache,
            ct=citation_tracker,
            mem=mem,
            wiki=wiki_router,
        )
    except Exception as e:
        _log.error(f"管道构建失败: {str(e)}")
        return None

@router.post("")
@limiter.limit("100/minute")
async def chat(request: Request, req: ChatRequest, user: dict = Depends(require_user)):
    """
    处理聊天请求的入口 API 端点 (POST: /api/chat)。
    
    执行完整的 RAG 流程:验证向量库 -> 查询重写 -> 缓存检查 -> 检索上下文 -> 调用大模型生成回答。
    支持流式和非流式两种响应模式。带有速率限制(100次/分钟)。
    
    参数:
        request (Request): FastAPI 原生 Request 对象，用于速率限制。
        req (ChatRequest): 包含用户消息和历史记录的请求体。
        user (dict): 依赖注入获取的当前用户信息字典，包含 tenant_id 等。
        
    返回:
        StreamingResponse | JSONResponse: 流式响应对象或包含完整答案的 JSON 响应。
    """
    # 记忆按「用户 + 会话」隔离，避免同一租户下不同会话互相污染
    session_key = f"{user.get('username') or ''}:{req.conversation_id or ''}"
    # 初始化管道和链路追踪
    pipeline = _build_pipeline(user["tenant_id"], session_key)
    trace = TraceContext()
    trace.begin_span("total")
    trace.set_input(req.message)

    if not pipeline:
        return _handle_vector_store_error(user["tenant_id"])

    # Wiki 路由用原始问句:别名面向用户说法做字面匹配，改写会破坏命中;
    # 命中即跳过查询改写与向量检索，省掉一次 LLM 调用(约 2s)。
    # 注意:Wiki 页是内置知识页，不依赖向量库，所以必须放在向量库门槛之前。
    # 精确缓存统一用原始问句做键(写入时也用 req.message)，否则改写后的查询永远查不到。
    wiki_context = _try_wiki_context(req.message, pipeline, req.history)

    if wiki_context:
        cached_result = await asyncio.to_thread(
            _check_cache, req.message, pipeline.cache, req.stream,
            getattr(pipeline, "semantic_cache", None),
            pipeline.embedder, req.message,
        )
        if cached_result:
            return cached_result
        context = wiki_context
    else:
        # 检查向量库是否可用:Wiki 未命中时才需要 RAG
        if not pipeline.vector_store:
            return _handle_vector_store_error(user["tenant_id"])

        # 向量库存在但为空(已上传但未完成索引或数据损坏)
        try:
            docs_count = len(_get_all_texts(pipeline.vector_store))
        except Exception:
            docs_count = 0
        if docs_count == 0:
            return _handle_vector_store_error(user["tenant_id"])

        # 寒暄/致谢不触发改写与检索:直接返回简短问候，避免拿无关片段硬答并罗列引用
        if _is_smalltalk(req.message):
            msg = ("您好，我是企业内部知识库助手，可咨询公司制度、客服流程与赔付规则，"
                   "我会基于已审定的资料解答。")
            if req.stream:
                return _create_streaming_response(msg)
            return JSONResponse({"answer": msg, "citations": [], "context_mode": "smalltalk"})

        # 处理查询重写(内部用同步 requests 调 LLM,最长阻塞 10s,丢线程池释放事件循环)
        query = await asyncio.to_thread(_process_query, req.message, pipeline.qr)

        # 检查缓存中是否已有结果(同步文件 IO + 语义缓存,丢线程池)
        cached_result = await asyncio.to_thread(
            _check_cache, req.message, pipeline.cache, req.stream,
            getattr(pipeline, "semantic_cache", None),
            pipeline.embedder, req.message,
        )
        if cached_result:
            return cached_result

        # 获取 RAG 上下文(检索 + 重排 + 构建Prompt,CPU 密集,丢线程池并在推理锁内串行)
        context = await asyncio.to_thread(
            _retrieve_context, query, pipeline, req.message, req.history, user["tenant_id"]
        )
    
    # 根据前端请求选择流式或非流式响应处理
    if req.stream:
        return await _handle_streaming_response(context, req, pipeline, trace, user)
    else:
        return await _handle_non_streaming_response(context, req, pipeline, trace, user)


@router.post("/stream")
@limiter.limit("100/minute")
async def chat_stream(request: Request, req: ChatRequest, user: dict = Depends(require_user)):
    """
    流式聊天接口(兼容前端 /api/chat/stream)。
    
    本质上是强制开启流式模式，然后调用核心的 chat 方法进行处理。
    
    参数:
        request (Request): FastAPI 原生 Request 对象。
        req (ChatRequest): 包含用户消息和历史记录的请求体。
        user (dict): 依赖注入获取的当前用户信息字典。
        
    返回:
        StreamingResponse: 流式响应对象。
    """
    req.stream = True
    return await chat(request, req, user)


def _handle_vector_store_error(tenant_id: str):
    """
    处理向量库不可用的情况，根据系统当前状态返回不同的提示信息。
    
    依次检查:是否有正在进行的后台索引任务 -> 是否有最近失败的任务 -> 是否有上传但未索引的文件 -> 确实没有上传文件。
    
    参数:
        tenant_id (str): 租户唯一标识符。
        
    返回:
        JSONResponse: 包含针对性提示信息的 JSON 响应。
    """
    # 1. 有正在后台索引的任务
    active = get_active_task_for_tenant(tenant_id)
    if active:
        return JSONResponse(content={
            "answer": (
                f"文档「{active['filename']}」正在后台索引中"
                f"({active['stage']}，进度 {active['progress']}%)，请稍候再问。"
            ),
            "citations": [],
            "token_usage": 0,
            "task_id": active["task_id"],
        })

    # 2. 最近有索引失败的任务
    failed = has_failed_task_for_tenant(tenant_id)
    if failed:
        return JSONResponse(content={
            "answer": (
                f"文档「{failed['filename']}」索引失败({failed.get('error', '未知错误')})，"
                "请删除后重新上传。"
            ),
            "citations": [],
            "token_usage": 0,
        })

    # 3. 已上传文件但向量库尚未建立(服务重启/索引中断)
    upload_dir = os.path.join(cfg.UPLOAD_DIR, tenant_id)
    has_uploads = os.path.exists(upload_dir) and any(
        f.lower().endswith(".pdf") for f in os.listdir(upload_dir)
    )
    if has_uploads:
        return JSONResponse(content={
            "answer": (
                "检测到您已上传文档，但向量索引尚未完成或已被中断，"
                "系统正在后台恢复索引，请稍候再试。"
            ),
            "citations": [],
            "token_usage": 0,
        })

    # 4. 确实没有上传过文档
    return JSONResponse(content={"answer": "请先上传文档", "citations": [], "token_usage": 0})

def _process_query(message, qr):
    """
    处理查询重写逻辑。
    
    使用查询重写器将用户的原始提问转换为更适合检索的表达方式。如果重写器不可用，则使用原始消息。
    
    参数:
        message (str): 用户原始输入的消息。
        qr (QueryRewriter): 查询重写器实例。
        
    返回:
        str: 重写后的查询语句；如果重写失败或无结果，则返回原始消息。
    """
    queries = qr.rewrite(message, num_variants=1) if qr else [message]
    return queries[0] if queries else message

def _check_cache(query, cache, is_stream, semantic_cache=None, embedder=None, original=None):
    """
    检查缓存中是否已存在当前查询的结果(精确匹配 + 语义匹配)。

    先走原有的精确缓存(QueryCache,按问题 MD5 命中完全相同的提问);
    再走新增的语义缓存(用原始问题向量近似匹配,近似提问也能复用答案)。

    参数:
        query (str): 用户原始提问(读写缓存都用它，保证键一致)。
        cache (QueryCache): 精确缓存实例。
        is_stream (bool): 是否为流式请求。
        semantic_cache (SemanticCache|None): 语义缓存实例(Redis 支撑)。
        embedder: 嵌入模型,用于计算原始问题的向量。
        original (str|None): 用户原始提问,语义缓存以它做匹配键。

    返回:
        StreamingResponse | JSONResponse | None: 命中则返回响应对象,否则返回 None。
    """
    # 1) 精确缓存:完全相同的问题直接命中(连同引用一起回放)
    if cache:
        entry = cache.get_entry(query) if hasattr(cache, "get_entry") else None
        if isinstance(entry, dict) and entry.get("answer"):
            citations = (entry.get("metadata") or {}).get("citations") or []
            return _fmt_cached(entry["answer"], is_stream, citations)

    # 2) 语义缓存:向量近似匹配,解决「近似问题漏缓存」问题
    if semantic_cache and embedder:
        try:
            orig = original or query
            emb = embedder.embed_query(orig)
            ans = semantic_cache.get(orig, emb)
            if ans:
                return _fmt_cached(ans, is_stream)
        except Exception as e:
            _log.warning("语义缓存查询失败,跳过: {}", e)

    return None


def _fmt_cached(content, is_stream, citations=None):
    """把缓存到的完整答案包装为流式或非流式响应(引用一并回放)。"""
    if is_stream:
        return _create_streaming_response(content, citations)
    return JSONResponse(content={"answer": content, "citations": citations or [], "token_usage": 0})

def _create_streaming_response(content, citations=None):
    """
    将已有的完整文本内容包装为 SSE 流式响应格式。
    
    主要用于缓存命中时，将非流式的缓存数据转换为前端期望的流式事件格式。
    
    参数:
        content (str): 需要发送的完整文本内容。
        citations (list, optional): 该答案的引用列表，缓存命中时一并回放。
        
    返回:
        StreamingResponse: 模拟的流式响应对象。
    """
    async def stream_generator():
        # 先发送内容 token 事件
        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
        # 发送结束 done 事件
        yield f"data: {json.dumps({'type': 'done', 'citations': citations or [], 'token_usage': 0})}\n\n"
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

def _get_context(query, pipeline, message, history, tenant_id=None):
    """
    获取 RAG 检索上下文和记忆上下文，并构建最终的提示词。
    
    流程:获取记忆上下文 -> Wiki 知识页命中则直接构建 Prompt(第一级)
          -> 未命中则混合检索 (稠密+稀疏) -> 重排序 -> 格式化引用 -> 构建 Prompt(兜底)。

    注:正常链路已在查询改写前完成 Wiki 路由(chat 中调用 `_try_wiki_context`)，
    本函数内的路由仅用于直接调用场景(如评测脚本)。
    
    参数:
        query (str): 重写后的查询语句。
        pipeline (SimpleNamespace): 当前请求的处理管道组件集合。
        message (str): 用户原始输入消息(用于记忆系统)。
        history (list): 对话历史。
        tenant_id (str, optional): 租户ID，用于加载稀疏向量库。
        
    返回:
        str: 构建好的包含上下文、记忆、历史和问题的完整提示词。
    """
    # 获取记忆上下文
    mem_ctx = pipeline.mem.get_context(query) if pipeline.mem else ""

    # 第一级:Wiki 知识页命中则直接用整页/命中小节作上下文，跳过混合检索
    # (正常链路已在查询改写之前路由，这里保留是为兼容直接调用本函数的评测脚本)
    wiki_prompt = _try_wiki_context(message or query, pipeline, history, mem_ctx)
    if wiki_prompt:
        return wiki_prompt

    # 第二级:未命中 Wiki，走混合检索兜底
    # RAG 检索:获取向量库中的所有文本，用于 BM25 等稀疏检索
    all_texts, text_meta = _get_all_texts_with_meta(pipeline.vector_store)
    
    # 加载 BGE-M3 稀疏向量库(按租户)，检索时与 BM25 + 稠密做 RRF 融合
    sparse_store = None
    if tenant_id:
        try:
            from app.retrieval.sparse_store import load_sparse_lookup
            sparse_store = load_sparse_lookup(tenant_id)
        except Exception:  # noqa: BLE001
            sparse_store = None
            
    # 初始化混合检索器并执行检索
    retriever = HybridRetriever(
        pipeline.vector_store, all_texts, k=10, sparse_store=sparse_store,
        text_meta=text_meta,
    )
    docs = retriever.retrieve(query)
    
    # 对检索结果进行重排序，取 top 5
    docs = _get_reranker().rerank(query, docs, top_k=5)
    
    # 格式化上下文，将检索到的文档添加到引用追踪器中
    pipeline.ct.add_sources(docs)
    context = pipeline.ct.format_context()
    
    # 构建提示模板并返回
    return _build_prompt(context, mem_ctx, history, query)

def _get_all_texts_with_meta(vector_store):
    """
    从 Chroma 取出全部切片正文与其 metadata(一一对应)，并滤除未审定文档。

    与 `_get_all_texts` 相比多返回 文本→metadata 映射:BM25 与 BGE-M3 稀疏通道
    只返回纯文本，融合时用该映射补回来源文件名与页码，避免引用显示“未知来源”。
    doc_status 为 draft 的切片在此直接剔除，保证稀疏通道不会召回未审定内容。

    参数:
        vector_store (Chroma): Chroma 向量库实例。

    返回:
        tuple: (文本列表, {文本: metadata})；异常时返回 ([], {})。
    """
    try:
        all_data = vector_store._collection.get()
        texts = all_data.get("documents", []) or []
        metas = all_data.get("metadatas", []) or []
        text_meta = {}
        for t, m in zip(texts, metas):
            if not t or not m:
                continue
            if m.get("doc_status") == "draft":
                continue
            text_meta[t] = m
        return [t for t in texts if t in text_meta], text_meta
    except Exception:
        return [], {}


def _get_all_texts(vector_store):
    """
    从 Chroma 向量数据库中提取所有的原始文本内容。
    
    主要用于为稀疏检索(如 BM25)提供语料库。
    
    参数:
        vector_store (Chroma): Chroma 向量库实例。
        
    返回:
        list: 包含所有文档文本的列表；如果发生异常则返回空列表。
    """
    try:
        # 直接访问 Chroma 底层的 collection 获取所有数据
        all_data = vector_store._collection.get()
        return all_data.get("documents", []) or []
    except Exception:
        return []

# 寒暄/致谢类短句:不触发检索，避免拿无关片段硬答并罗列引用
_SMALLTALK_RE = re.compile(
    r"^[\s,，。！!？?~～]*(你好|您好|hi|hello|在吗|在不在|嗨|哈喽|谢谢|感谢|多谢|"
    r"早上好|下午好|晚上好|早安|晚安|好的|收到|辛苦了)[\s,，。！!？?~～啊呀呢吧]*$",
    re.IGNORECASE,
)


def _is_smalltalk(text: str) -> bool:
    """判断是否为寒暄类输入(无实质业务问题)。

    参数:
        text (str): 用户原始输入。

    返回:
        bool: True 表示只是打招呼/致谢，无需检索。
    """
    return len((text or "").strip()) <= 12 and bool(_SMALLTALK_RE.match(text or ""))


def _build_prompt(context, mem_ctx, history, query):
    """
    构建发送给大语言模型的提示词模板。
    
    将检索到的参考文本、记忆上下文、对话历史和当前问题组合成结构化的 Prompt。
    
    参数:
        context (str): RAG 检索并格式化后的参考文本。
        mem_ctx (str): 长期记忆提取的上下文。
        history (list): 对话历史列表。
        query (str): 当前用户提问。
        
    返回:
        str: 填充完毕的完整提示词字符串。
    """
    # 截取最近 4 轮对话历史，并限制单条内容长度以防止 Token 溢出
    history_text = ""
    for m in (history or [])[-4:]:
        history_text += f"{m.get('role', 'user')}: {m.get('content', '')[:200]} " + chr(10)
    
    # 定义企业内部知识库助手的系统提示模板
    PROMPT_TEMPLATE = """你是企业内部知识库的问答助手，服务对象是公司客服与工作人员。回答必须基于下方参考文本。

    参考文本:
    {context}

    记忆上下文:
    {memory}

    对话历史:
    {history}

    问题: {question}

    要求:
    1. 只使用与问题直接相关的片段，用 [N] 标注来源；与问题无关的片段不要引用。
    2. 参考文本不包含答案时，明确说明「资料中没有相关内容」，不要编造制度、金额、流程或法条结论。
    3. 参考文本为空时，直接告知无相关资料，并建议补充资料或请示主管。
    4. 用户问「某份文件里有没有某项规定」时，先依据参考文本给出该规定的内容与出处
       (文件名与页码)，不要因为文件名不完全吻合就回答「没有相关内容」。"""
    
    # 填充模板并返回
    return PROMPT_TEMPLATE.format(
        context=context,
        memory=mem_ctx,
        history=history_text,
        question=query
    )

async def _handle_streaming_response(context, req, pipeline, trace, user, record_source="api_stream"):
    """
    处理流式响应逻辑。
    
    向大模型发起流式请求，并将返回的 SSE 数据流实时转发给前端。
    包含错误处理、心跳保活、以及响应结束后的缓存和记忆更新逻辑。
    
    参数:
        context (str): 构建好的提示词。
        req (ChatRequest): 原始请求对象。
        pipeline (SimpleNamespace): 管道组件集合。
        trace (TraceContext): 链路追踪上下文。
        
    返回:
        StreamingResponse: SSE 流式响应对象。
    """
    async def generate():
        full_answer = ""
        provider = ""  # 实际命中的 LLM 供应商(供 trace 落库做 fallback 归因)
        start_time = time.time()
        last_yield_time = time.time()
        client = None
        resp = None
        
        try:
            # 通过统一 LLM 客户端打开流式连接(主供应商限流/5xx 时自动故障转移)
            try:
                client, resp, provider = await stream_chat(
                    [{"role": "user", "content": context}],
                    temperature=0.1, max_tokens=1024,
                )
            except LLMHardError as e:
                # 鉴权等硬错误:给出指向对应供应商密钥的友好提示
                if e.status in (401, 403):
                    hint = (f"LLM 鉴权失败（供应商: {e.provider}）。请检查 .env 中 "
                            f"{e.provider.upper()}_API_KEY 或 LLM_API_KEY 是否有效,更新后重启服务。")
                else:
                    hint = f"LLM 服务返回错误(供应商 {e.provider}, HTTP {e.status}):{e.body[:200]}"
                yield f"data: {json.dumps({'type': 'token', 'content': '[系统] ' + hint})}\n\n"
                yield f"data: {json.dumps({'type': 'error', 'content': hint})}\n\n"
                return
            except LLMAllFailed:
                hint = "所有 LLM 供应商均不可用,请稍后重试或检查 .env 中配置的供应商 API Key。"
                yield f"data: {json.dumps({'type': 'token', 'content': '[系统] ' + hint})}\n\n"
                yield f"data: {json.dumps({'type': 'error', 'content': hint})}\n\n"
                return

            # 逐行读取 SSE 响应流(首个成功供应商)
            async for line in resp.aiter_lines():
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]": break

                    try:
                        chunk = json.loads(data_str)
                        # 部分供应商会在末尾发送 choices 为空的 usage chunk，直接跳过
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        token = (choices[0].get("delta") or {}).get("content") or ""
                        if token:
                            full_answer += token
                            # 将 token 实时推送到前端
                            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                            last_yield_time = time.time()
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                        continue

                    # 心跳检测:如果超过 30 秒没有新内容产生，发送空心跳保持连接
                    if time.time() - last_yield_time > 30:
                        yield f"data: {json.dumps({'type': 'token', 'content': ''})}\n\n"
                        last_yield_time = time.time()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return
        finally:
            # 正常结束、异常、前端断开都要释放连接，否则每次请求都会漏一个 httpx 客户端
            if resp is not None:
                try:
                    await resp.aclose()
                except Exception:  # noqa: BLE001
                    pass
            if client is not None:
                try:
                    await client.aclose()
                except Exception:  # noqa: BLE001
                    pass
        
        # 流结束后的后处理:更新缓存和记忆系统,并落库 trace(含耗时与供应商)
        await _handle_post_processing(
            req.message, full_answer, pipeline,
            tenant_id=user["tenant_id"],
            duration_ms=round((time.time() - start_time) * 1000, 1),
            provider=provider,
        )
        
        # 获取引用列表并发送结束信号
        citations = _get_citations(pipeline.ct)
        yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'token_usage': 0})}\n\n"
        
        # 记录监控指标
        record_query(
            duration_ms=(time.time() - start_time) * 1000,
            token_usage=0,
            success=True,
            source=record_source
        )
    
    return StreamingResponse(generate(), media_type="text/event-stream")

async def _handle_non_streaming_response(context, req, pipeline, trace, user):
    """
    处理非流式响应逻辑。
    
    向大模型发起普通请求，等待完整结果返回后，一次性以 JSON 格式返回给前端。
    
    参数:
        context (str): 构建好的提示词。
        req (ChatRequest): 原始请求对象。
        pipeline (SimpleNamespace): 管道组件集合。
        trace (TraceContext): 链路追踪上下文。
        user (dict): 当前登录用户信息(后处理落库需 tenant_id)。
        
    返回:
        JSONResponse: 包含完整答案、引用和 token 使用情况的 JSON 响应。
    """
    trace.begin_span("llm")
    answer, citations, token_usage, errored = "", [], 0, ""

    # 复用唯一的流式实现，消费完整 SSE 后聚合成一次性响应
    stream = await _handle_streaming_response(
        context, req, pipeline, trace, user, record_source="api")

    async for raw in stream.body_iterator:
        payload = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for line in payload.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "token":
                answer += event.get("content", "")
            elif event.get("type") == "error":
                errored = event.get("content", "")
            elif event.get("type") == "done":
                citations = event.get("citations", [])
                token_usage = event.get("token_usage", 0)

    if errored and not answer:
        answer = f"[系统] {errored}"

    # 记录追踪和日志信息
    trace.end_span()
    trace.set_output(str(answer)[:500])
    trace.set_tokens(token_usage)
    trace.print_summary()
    _log.query(req.message, len(answer), token_usage, trace.total_duration_ms(), False)

    return JSONResponse(content={"answer": answer, "citations": citations, "token_usage": token_usage})

async def _handle_post_processing(message, answer, pipeline, tenant_id="", duration_ms=0.0, provider="", cached=False, success=True):
    """
    请求结束后的统一后处理逻辑:更新缓存、记忆系统,并落库问答 trace。

    参数:
        message (str): 用户原始提问。
        answer (str): 大模型生成的回答。
        pipeline (SimpleNamespace): 管道组件集合。
        tenant_id (str): 租户 id,用于 trace 隔离与回流。
        duration_ms (float): 端到端耗时(毫秒),供延迟分析。
        provider (str): 实际命中的 LLM 供应商名,供 fallback 归因。
        cached (bool): 是否命中缓存(命中缓存的问答不进入此后处理,通常 False)。
        success (bool): 是否成功生成(异常路径不调用本函数,通常 True)。
    """
    # 先取引用:既要随缓存一起保存(缓存命中时回放)，也要落进 trace
    try:
        citations = _get_citations(pipeline.ct)
    except Exception:
        citations = []

    # 1. 将问答对写入精确缓存(键用原始提问，与读取侧保持一致)
    if pipeline.cache:
        try:
            pipeline.cache.set(message, answer, metadata={"citations": citations})
        except Exception as e:
            _log.error(f"缓存写入失败: {str(e)}")

    # 2. 将问答对写入语义缓存(以原始问题向量为键,近似问题可复用答案)
    if getattr(pipeline, "semantic_cache", None) and pipeline.embedder:
        try:
            emb = pipeline.embedder.embed_query(message)
            pipeline.semantic_cache.set(message, emb, answer)
        except Exception as e:
            _log.error(f"语义缓存写入失败: {str(e)}")
    
    # 3. 将问答对写入记忆系统，并触发后台任务(如记忆总结)
    if pipeline.mem:
        try:
            pipeline.mem.add("user", message)
            pipeline.mem.add("assistant", answer)
            pipeline.mem.trigger_background_jobs(call_llm)
        except Exception as e:
            _log.error(f"记忆更新失败: {str(e)}")

    # 4. 落库问答 trace(轻量自进化闭环-经验捕获层):沉淀 query/answer/引用/耗时/供应商,
    #    供后续失败归因与评测回归使用。SQLite 写是同步 IO,丢线程池避免阻塞事件循环。
    await asyncio.to_thread(
        save_query_trace,
        tenant_id, message, answer, citations,
        duration_ms, 0, provider, cached, success,
    )

def _get_citations(context_tracker):
    """
    从引用追踪器中提取格式化的引用列表。
    
    参数:
        context_tracker (CitationTracker): 引用追踪器实例。
        
    返回:
        list: 格式化后的引用列表，每个元素包含 source (来源) 和 content (内容片段)。
    """
    sources = context_tracker.get_sources()
    return [{"source": s.filename or "未知来源",
             "content": (s.content[:200] if s.content else "")}
            for s in sources]
