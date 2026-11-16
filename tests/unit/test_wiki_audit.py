"""
test_wiki_audit.py —— 文档上线前质检的单元测试

【测试覆盖范围】
- `repeated_lines`:重复短行被判为页眉页脚残留；唯一行与超长行不被误判；切片过少不启用。

【依赖】
- 仅测纯函数，不触碰向量库与模型。
"""
from app.wiki.audit import repeated_lines


def test_repeated_line_detected():
    docs = [f"正文内容{i}\n客服手册 v0.1" for i in range(1, 5)]
    assert repeated_lines(docs) == ["客服手册 v0.1"]


def test_unique_line_kept():
    docs = [f"正文内容{i}\n客服手册 v0.1" for i in range(1, 5)]
    found = repeated_lines(docs)
    assert all("正文内容" not in ln for ln in found)


def test_long_line_ignored():
    long_line = "这" * 60
    docs = [f"{long_line}\n正文{i}" for i in range(1, 5)]
    assert repeated_lines(docs) == []


def test_too_few_chunks():
    assert repeated_lines(["页脚\n正文", "页脚\n正文"]) == []
