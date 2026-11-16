"""
test_bugfix_regressions.py —— P0~P2 缺陷修复的回归用例

【测试覆盖范围】
- 精确缓存:读写键一致(都用原始问句)、命中时回放引用、get_entry 不重复计数。
- 缓存清理:版本变化时必须清掉 cache/<tenant>/*.json(此前为空转)。
- 会话越权:跨租户/跨用户的会话读写一律 404。
- 记忆整合:整理期间新写入的短期记忆不被丢弃。

【依赖】
- 仅用临时目录与轻量替身，不依赖 Redis / Chroma / LLM。
"""
import json
import os

import pytest
from fastapi import HTTPException

from app.api import conversation as conversation_api
from app.api.chat import _check_cache
from app.memory.memory_manager import MemorySystem
from app.retrieval.cache import QueryCache
from app.wiki.store import WikiStore


def test_cache_roundtrip_keeps_citations(tmp_path):
    """写入后按同一问句命中，并把引用一起回放。"""
    cache = QueryCache(cache_dir=str(tmp_path))
    cache.set("司机需要哪些证件", "需要三证", metadata={"citations": [{"source": "手册.pdf"}]})

    resp = _check_cache("司机需要哪些证件", cache, is_stream=False)

    assert resp is not None
    body = json.loads(resp.body)
    assert body["answer"] == "需要三证"
    assert body["citations"][0]["source"] == "手册.pdf"


def test_cache_miss_on_rewritten_query(tmp_path):
    """改写后的查询不应命中原始问句的缓存(读写键必须一致)。"""
    cache = QueryCache(cache_dir=str(tmp_path))
    cache.set("司机需要哪些证件", "需要三证")

    assert _check_cache("网约车司机上岗需要什么证件", cache, is_stream=False) is None


def test_cache_hit_counted_once(tmp_path):
    """get() 复用 get_entry()，命中次数不应重复累加。"""
    cache = QueryCache(cache_dir=str(tmp_path))
    cache.set("问题", "答案")

    assert cache.get("问题") == "答案"
    assert cache.stats()["hits"] == 1


def test_clear_query_cache_clears_tenant_dirs(tmp_path, monkeypatch):
    """缓存清理必须覆盖 cache/<tenant>/ 下的文件，而不只是根目录。"""
    monkeypatch.chdir(tmp_path)
    os.makedirs("cache/t1")
    open("cache/t1/a.json", "w", encoding="utf-8").close()
    open("cache/b.json", "w", encoding="utf-8").close()

    WikiStore.__new__(WikiStore)._clear_query_cache()

    assert not os.path.exists("cache/t1/a.json")
    assert not os.path.exists("cache/b.json")


def test_conversation_guard_rejects_other_tenant(monkeypatch):
    """跨租户访问会话统一 404，避免越权与 ID 枚举。"""
    monkeypatch.setattr(
        conversation_api, "get_conversation_owner",
        lambda cid: {"tenant_id": "t2", "user_id": ""},
    )
    with pytest.raises(HTTPException) as exc:
        conversation_api._require_owned_conversation(
            1, {"tenant_id": "t1", "username": "u"}
        )
    assert exc.value.status_code == 404


def test_conversation_guard_rejects_other_user(monkeypatch):
    """同租户但 user_id 不同的会话也不可读写。"""
    monkeypatch.setattr(
        conversation_api, "get_conversation_owner",
        lambda cid: {"tenant_id": "t1", "user_id": "alice"},
    )
    with pytest.raises(HTTPException):
        conversation_api._require_owned_conversation(
            1, {"tenant_id": "t1", "username": "bob"}
        )


class _StubStore:
    """记录写库调用但不落盘的向量库替身。"""

    def add_texts(self, **kwargs):
        """忽略写入，仅保证接口存在。"""


class _StubRedis:
    """中期记忆替身(内存实现)。"""

    def get_mid_term(self, session_id):
        """返回空摘要。"""
        return ""

    def set_mid_term(self, session_id, value):
        """忽略写入。"""


def test_consolidate_keeps_new_messages():
    """整理期间新写入的短期记忆不能被覆盖丢弃。"""
    system = MemorySystem.__new__(MemorySystem)
    system.tenant_id = "t1"
    system.session_id = "s1"
    system.max_short_term = 2
    system.mid_term = ""
    system.short_term = [{"role": "user", "content": f"问题{i}"} for i in range(4)]
    system.store = _StubStore()
    system.redis = _StubRedis()

    def llm(prompt):
        # 模拟 LLM 调用期间用户又发来一条消息
        system.short_term.append({"role": "user", "content": "新问题"})
        return "摘要"

    system._do_consolidate(llm)

    assert [m["content"] for m in system.short_term] == ["问题2", "问题3", "新问题"]
