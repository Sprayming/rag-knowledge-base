"""
test_smalltalk.py —— 寒暄短路与提示词口径的单元测试

【测试覆盖范围】
- `_is_smalltalk`:纯问候/致谢命中；带业务问题的长句不命中；空输入不命中。
- `_build_prompt`:系统提示词为内部知识库助手，且要求无相关内容时明确说明、不编造。

【依赖】
- 仅依赖 app.api.chat 的纯函数，不触发检索与模型调用。
"""
from app.api.chat import _build_prompt, _is_smalltalk


def test_greetings_hit():
    for text in ["你好", "您好！", "hi", "Hello", "在吗", "谢谢", "辛苦了"]:
        assert _is_smalltalk(text), text


def test_business_question_not_hit():
    for text in ["你好，司机证件过期怎么办", "乘客投诉绕路能赔多少", "谢谢你，顺便问下发票怎么开"]:
        assert not _is_smalltalk(text), text


def test_empty_not_hit():
    assert not _is_smalltalk("")
    assert not _is_smalltalk("   ")


def test_prompt_is_internal_kb_persona():
    prompt = _build_prompt("[1] 网约车运营规范 第1页\n司机不得拒载", "", [], "司机可以拒载吗")
    assert "企业内部知识库" in prompt
    assert "法律专家" not in prompt
    assert "资料中没有相关内容" in prompt


def test_prompt_handles_empty_context():
    prompt = _build_prompt("", "", [], "你好")
    assert "参考文本为空" in prompt
