"""
test_wiki.py —— Wiki 知识页层单元测试

【测试覆盖范围】
- WikiStore:建表与种子写入、按 ID 读取、草稿页不进入问答(仅 approved 可路由)。
- WikiRouter:别名命中、小节命中、明确法条引用(query 含“第X条”)不拦截。
- Page.render:上下文包含标题、小节标题与可溯源出处行。

【依赖】
- 仅依赖 app.wiki 与临时 SQLite 文件，不触碰向量库与模型。
"""
import os
from dataclasses import replace

import pytest

from app.wiki.router import WikiRouter, normalize
from app.wiki.seed import SEED_PAGES
from app.wiki.store import Page, Section, Span, WikiStore


@pytest.fixture()
def store(tmp_path):
    """用临时库初始化一份种子页面，避免污染 data/wiki.db。"""
    return WikiStore(str(tmp_path / "wiki.db"))


class TestWikiStore:
    """存储层:种子、读取、状态过滤"""

    def test_seed_pages_loaded(self, store):
        """首次初始化应写入内置页面。"""
        pages = store.list_pages("")
        assert len(pages) >= 4
        assert {p.id for p in pages} >= {
            "driver_access", "service_standard", "data_privacy", "emergency_handling",
        }

    def test_get_page_with_spans(self, store):
        """页面详情应带可回溯出处(文档名 + 页码 + 摘句)。"""
        page = store.get("data_privacy")
        assert page is not None
        spans = [sp for s in page.sections for sp in s.spans]
        assert spans and all(sp.doc_name and sp.page > 0 and sp.quote for sp in spans)

    def test_draft_page_excluded_from_routing_list(self, store):
        """draft 状态页面不应出现在问答可用的页面列表里。"""
        store.upsert(Page(id="tmp_draft", title="草稿页", aliases=["草稿别名"],
                          summary="x", sections=[Section(heading="草稿问题？", body="y")],
                          status="draft"))
        assert store.get("tmp_draft") is not None
        assert "tmp_draft" not in {p.id for p in store.list_pages("")}


class TestWikiRouter:
    """路由层:命中、小节级命中、法条查询放行给 RAG"""

    def test_alias_hit(self, store):
        """别名命中整页。"""
        hit = WikiRouter(store, "").route("跑网约车需要什么证件？")
        assert hit is not None
        assert hit.page.id == "driver_access"

    def test_section_hit_returns_single_section(self, store):
        """命中某个小节标题时只带回该节，节省 Prompt 预算。"""
        hit = WikiRouter(store, "").route("乘客投诉怎么处理")
        assert hit is not None
        assert len(hit.sections) == 1
        assert "投诉" in hit.sections[0].heading

    def test_law_clause_query_goes_to_rag(self, store):
        """明确指向法条的查询不拦截，交给 RAG 检索原文。"""
        assert WikiRouter(store, "").route("刑法第二百六十四条怎么规定") is None

    def test_emergency_page_routed(self, store):
        """应急处置页:事故、乘客突发疾病、遗落物品都应命中。"""
        router = WikiRouter(store, "")
        for query in ("出交通事故怎么处理", "乘客在车上晕倒怎么办", "乘客落东西了怎么还"):
            hit = router.route(query)
            assert hit is not None and hit.page.id == "emergency_handling", query

    def test_draft_page_readable_but_not_routable(self, store):
        """赔付页为草案:可读，但未经审定不得进入问答。"""
        assert store.get("complaint_compensation") is not None
        assert "complaint_compensation" not in {p.id for p in store.list_pages("")}
        assert WikiRouter(store, "").route("这种情况能赔多少") is None

    def test_unrelated_query_falls_through(self, store):
        """无关问题不应误命中。"""
        assert WikiRouter(store, "").route("今天杭州天气怎么样") is None

    def test_normalize(self):
        """归一化应去掉空白与中英文标点。"""
        assert normalize("乘客，个人信息？") == "乘客个人信息"


class TestReview:
    """审定流程:状态迁移、可路由性变化、审计与内容变更失效"""

    def test_approve_makes_draft_page_routable(self, store):
        """草案页审定通过后进入问答路由。"""
        assert WikiRouter(store, "").route("能赔多少") is None

        store.review("complaint_compensation", "approve", "张主管", "赔付标准已核对")

        page = store.get("complaint_compensation")
        assert page.status == "approved"
        assert page.approved_by == "张主管"
        assert WikiRouter(store, "").route("能赔多少") is not None

    def test_reject_removes_page_from_routing(self, store):
        """驳回后页面退出问答，但仍可读。"""
        store.review("driver_access", "reject", "张主管", "证件要求待更新")

        assert store.get("driver_access") is not None
        assert "driver_access" not in {p.id for p in store.list_pages("")}

    def test_review_is_logged(self, store):
        """每次审定写入审计日志，含审定人与意见。"""
        store.review("data_privacy", "approve", "李法务", "已核")

        latest = store.list_reviews()[0]
        assert latest["target_id"] == "data_privacy"
        assert latest["action"] == "approve"
        assert latest["reviewer"] == "李法务"

    def test_content_change_resets_approval(self, store):
        """内容变更后重新同步种子，页面回到草案，需要重新审定。"""
        store.review("complaint_compensation", "approve", "张主管", "")

        seed_page = next(p for p in SEED_PAGES if p.id == "complaint_compensation")
        changed = replace(seed_page, summary=seed_page.summary + "（本次修订）")
        store.upsert(changed)

        assert store.get("complaint_compensation").status == "draft"


class TestPageRender:
    """渲染层:上下文结构"""

    def test_render_contains_heading_and_source(self, store):
        page = store.get("driver_access")
        text = page.render(page.sections[:1])
        assert page.sections[0].heading in text
        assert "依据：" in text
