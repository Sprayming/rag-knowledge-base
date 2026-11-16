"""app/wiki —— Wiki 知识页层(检索前置的结构化入口)

与 RAG 的分工:Wiki 命中则用整页(或命中小节)作上下文，未命中再降级到
混合检索。两者共用 CitationTracker，引用格式保持一致。
"""

from app.wiki.router import WikiHit, WikiRouter
from app.wiki.store import Page, Section, Span, WikiStore, get_store

__all__ = ["Page", "Section", "Span", "WikiStore", "get_store", "WikiHit", "WikiRouter"]
