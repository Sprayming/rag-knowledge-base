"""
test_memory_consolidate.py —— 记忆整合的 LLM 调用兼容性测试

【测试覆盖范围】
- `_run_llm`:同步函数与 async 函数都能拿到字符串结果;空返回归一为 ""。
- `_do_consolidate`:传入 async 的 llm_func 时，摘要以字符串写入，而不是协程对象。
- `_do_extract_entity`:带 Markdown 围栏 / 裸列表 / 混入非字典元素 / 非法 JSON 时的解析行为。

【依赖】
- 不依赖 Redis / Chroma:用最小替身对象覆盖 store / redis / worker。
"""
import asyncio

from app.memory.memory_manager import MemorySystem, _run_llm


class _StubStore:
    """记录写入文本的向量库替身。"""

    def __init__(self):
        self.texts = []

    def add_texts(self, texts, metadatas=None, ids=None):
        """记录写库的文本，供断言其类型为 str。"""
        self.texts.extend(texts)


class _StubRedis:
    """中期记忆替身(内存实现)。"""

    def __init__(self):
        self.mid = ""

    def get_mid_term(self, session_id):
        """返回已保存的中期摘要。"""
        return self.mid

    def set_mid_term(self, session_id, value):
        """保存中期摘要。"""
        self.mid = value


def _make_system():
    """构造一个绕过外部依赖的 MemorySystem 实例。"""
    system = MemorySystem.__new__(MemorySystem)   # 跳过需要 Chroma 的 __init__
    system.tenant_id = "t1"
    system.session_id = "s1"
    system.max_short_term = 2
    system.mid_term = ""
    system.short_term = [
        {"role": "user", "content": f"问题{i}"} for i in range(4)
    ]
    system.store = _StubStore()
    system.redis = _StubRedis()
    return system


def test_run_llm_sync():
    assert _run_llm(lambda p: "摘要", "x") == "摘要"


def test_run_llm_async():
    async def llm(p):
        await asyncio.sleep(0)
        return "异步摘要"

    assert _run_llm(llm, "x") == "异步摘要"


def test_run_llm_empty():
    assert _run_llm(lambda p: None, "x") == ""


def test_consolidate_accepts_async_llm():
    """核心回归:async llm_func 必须产出字符串摘要，不能把协程写进向量库。"""
    async def llm(p):
        return "- Intent: 咨询证件\n- Facts: 无"

    system = _make_system()
    system._do_consolidate(llm)

    assert len(system.store.texts) == 1
    assert isinstance(system.store.texts[0], str)
    assert "Intent" in system.mid_term


class _StubProfile:
    """记录合并结果的画像存储替身。"""

    def __init__(self):
        self.merged = []

    def merge_entities(self, tenant_id, entities):
        """记录本次合并的实体列表。"""
        self.merged.extend(entities)


def _make_entity_system():
    """构造只带画像存储的 MemorySystem 实例。"""
    system = MemorySystem.__new__(MemorySystem)   # 跳过需要 Chroma 的 __init__
    system.tenant_id = "t1"
    system.profile = _StubProfile()
    return system


def test_extract_entity_fenced_json():
    """模型输出带 ```json 围栏时必须能解析。"""
    reply = '```json\n{"entities": [{"key": "城市", "value": "长沙", "confidence": 0.9}]}\n```'
    system = _make_entity_system()
    system._do_extract_entity("我在长沙跑车", "已记录", lambda p: reply)

    assert system.profile.merged == [{"key": "城市", "value": "长沙", "confidence": 0.9}]


def test_extract_entity_bare_list():
    """模型直接返回裸列表时也要兼容。"""
    system = _make_entity_system()
    system._do_extract_entity(
        "我在长沙跑车", "已记录",
        lambda p: '[{"key": "城市", "value": "长沙", "confidence": 0.8}]',
    )

    assert len(system.profile.merged) == 1
    assert system.profile.merged[0]["key"] == "城市"


def test_extract_entity_filters_non_dict():
    """混入非字典元素时只保留合法实体，不整批丢弃。"""
    reply = '{"entities": [{"key": "城市", "value": "长沙", "confidence": 0.9}, "垃圾数据"]}'
    system = _make_entity_system()
    system._do_extract_entity("我在长沙跑车", "已记录", lambda p: reply)

    assert len(system.profile.merged) == 1


def test_extract_entity_invalid_json_skipped():
    """非法 JSON 静默跳过，不抛异常。"""
    system = _make_entity_system()
    system._do_extract_entity("x", "y", lambda p: "抱歉，我无法提取")

    assert system.profile.merged == []
