"""
test_page_furniture.py —— 页眉页脚清洗的单元测试

【测试覆盖范围】
- detect_page_furniture:重复短行被识别为版式噪声，唯一行不被误判，页数过少时不启用。
- strip_page_furniture:剔除重复标题与页码行，保留正文与空行结构。

【依赖】
- 仅依赖 app.processing.multimodal_pipeline 中的纯函数，不加载模型。
"""
from app.processing.multimodal_pipeline import detect_page_furniture, strip_page_furniture


def _page(body: str, page_no: int) -> str:
    return f"{body}\n平台客服手册 v0.1（草案）\n第 {page_no} 页 / 共 3 页"


class TestDetectFurniture:
    """重复行识别"""

    def test_repeated_line_detected(self):
        pages = [_page(f"正文内容{i}", i) for i in range(1, 4)]
        furniture = detect_page_furniture(pages)
        assert "平台客服手册v0.1（草案）" in furniture

    def test_unique_line_kept(self):
        pages = [_page(f"正文内容{i}", i) for i in range(1, 4)]
        furniture = detect_page_furniture(pages)
        assert all("正文内容" not in key for key in furniture)

    def test_too_few_pages(self):
        assert detect_page_furniture(["标题\n正文", "标题\n正文"]) == set()


class TestStripFurniture:
    """页脚与页码剔除"""

    def test_strip_footer_and_page_number(self):
        pages = [_page(f"正文内容{i}", i) for i in range(1, 4)]
        furniture = detect_page_furniture(pages)
        out = strip_page_furniture(pages[1], furniture)
        assert "正文内容2" in out
        assert "平台客服手册" not in out
        assert "第 2 页" not in out

    def test_blank_lines_preserved(self):
        pages = [_page(f"第一段{i}\n\n第二段{i}", i) for i in range(1, 4)]
        furniture = detect_page_furniture(pages)
        out = strip_page_furniture(pages[0], furniture)
        assert "\n\n" in out
