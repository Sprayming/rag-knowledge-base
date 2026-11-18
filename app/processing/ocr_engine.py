
"""
app/processing/ocr_engine —— 多后端 OCR 引擎

【作用与功能】
本模块提供统一的 OCR 文字识别接口 `OCREngine`，对图片(PDF 中提取的图
片、扫描页等)执行文字识别。它按优先级自动探测可用后端(PaddleOCR /
pytesseract)，并兼容不同版本的 API 差异，使上层管线无需关心具体引擎。

【主要组成】
- `OCREngine`:OCR 引擎类，`recognize()` 为对外识别入口，
  `_init_backend()/_init_paddleocr()/_init_pytesseract()` 负责后端初始化，
  `_ensure_init()` 实现惰性、幂等的后端加载。

【适用场景】
- 场景1:处理管线对 PDF 每页图片调用 `recognize()` 提取图片文字。
- 场景2:仅有扫描层、无文本层的知识库文档，通过 OCR 获取可检索文字。

【依赖关系】
- 上游调用方:`app.processing.multimodal_pipeline`。
- 下游依赖:可选 `paddleocr`、可选 `pytesseract` + `pillow`。
"""
import os, base64
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OCREngine:
    """OCR 引擎 - 自动检测可用后端。

    构造时不立即加载任何后端(惰性初始化)，首次 `recognize()` 时按
    优先级探测并锁定一个可用后端；若无任何引擎可用则标记 "none" 并返回
    空串。后端名称与 OCR 句柄分别缓存在 `_backend` 与 `_ocr`。
    """

    def __init__(self, lang: str = "ch"):
        """初始化 OCR 引擎(仅记录配置，不加载后端)。

        参数:
            lang (str): 识别语言代码，默认 "ch"(中文)。注意 PaddleOCR
                初始化时内部硬编码中文模型，此处主要供 pytesseract 分支使用。
        """
        self.lang = lang
        # 懒加载:默认不初始化任何后端，首次 recognize() 时才按需加载，
        # 避免纯文字 PDF(无需 OCR)也白白占用 PaddleOCR 数百 MB 显存/内存。
        self._backend = None  # None=尚未初始化；"none"=无可用引擎
        self._ocr = None

    def _ensure_init(self):
        """首次需要时初始化 OCR 后端(幂等)。

        若 `_backend` 仍为 None(尚未尝试初始化)，则调用 `_init_backend()`；
        否则直接返回，保证后端只加载一次。这样可避免纯文字 PDF 也白白占用
        PaddleOCR 数百 MB 显存/内存。
        """
        if self._backend is not None:
            return
        self._init_backend()

    def _init_backend(self):
        """按优先级尝试初始化 OCR 后端。

        依次尝试 `paddleocr` → `pytesseract`；首个成功初始化的后端即被
        采用并置 `_backend`，随后返回。若全部失败，则置 `_backend="none"`，
        后续 `recognize()` 直接返回空串而不再重试。
        """
        backends = [
            ("paddleocr", self._init_paddleocr),
            ("pytesseract", self._init_pytesseract),
        ]
        for name, init_fn in backends:
            try:
                init_fn()
                logger.info(f"OCR 后端: {name}")
                return
            except Exception:
                continue
        logger.warning("OCR 后端: 无可用引擎，返回空文本")
        self._backend = "none"

    def _init_paddleocr(self):
        """初始化 PaddleOCR 后端(兼容 3.x 与 2.x API)。

        优先使用 PaddleOCR 3.x 的新参数(文档方向/扭转/文本行方向开关)；
        若构造时抛出 TypeError，则回退到 2.x 旧参数
        (use_angle_cls / use_gpu)。成功后置 `_backend="paddleocr"`。
        """
        from paddleocr import PaddleOCR
        try:
            # PaddleOCR 3.x API(2024+):用文档方向/扭转/文本行方向开关替代旧的
            # use_angle_cls / use_gpu；lang 仍用于选择中英文模型。
            # text_det_limit_side_len:限制检测阶段输入长边，扫描件大图在 CPU 上
            # 推理极慢，压到 736 可显著降低单页耗时且不影响召回行数。
            # use_textline_orientation=False:企业知识库正文几乎都是正向排版，
            # 关闭文字行方向分类可省掉一次模型推理。
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_det_limit_side_len=736,
                text_det_limit_type="max",
                lang="ch",
            )
        except TypeError:
            # 兼容旧版 2.x API
            self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False, use_gpu=False)
        self._backend = "paddleocr"

    def _init_pytesseract(self):
        """初始化 pytesseract 后端。

        仅做模块级导入并把 pytesseract 对象作为 OCR 句柄保存；成功后置
        `_backend="pytesseract"`。运行时以 `chi_sim+eng` 语言包识别。
        """
        import pytesseract
        self._ocr = pytesseract
        self._backend = "pytesseract"

    @staticmethod
    def _order_by_layout(texts: list, boxes, img_w: int, img_h: int) -> list:
        """按版面顺序重排 OCR 文本行。

        PaddleOCR 默认按检测框的纵向位置输出，遇到双栏排版的扫描件(如人大
        公报、期刊)会把左右栏的文字交错拼接，导致句子断裂、检索命中率下降。
        本方法按版面把行分成"页眉 / 正文 / 页脚"三段，正文再判断是否双栏:
        若中线两侧都有足够多的行则判为双栏，先读完左栏再读右栏，栏内按纵坐标
        排序；单栏文档则整体按纵坐标排序。

        参数:
            texts (list): 识别出的文本行。
            boxes: 与 texts 等长的检测框序列，每项为 [x0, y0, x1, y1]。
            img_w (int): 图片宽度；<=0 时退化为按 (y, x) 排序。
            img_h (int): 图片高度。

        返回:
            list: 按阅读顺序排列的文本行；解析失败时原样返回 texts。
        """
        try:
            items = []
            for b, t in zip(boxes, texts):
                x0, x1 = float(b[0]), float(b[2])
                y0, y1 = float(b[1]), float(b[3])
                items.append({"x0": x0, "x1": x1, "y": (y0 + y1) / 2, "t": str(t)})

            if img_w <= 0 or img_h <= 0:
                items.sort(key=lambda it: (it["y"], it["x0"]))
                return [it["t"] for it in items]

            top, bottom = img_h * 0.10, img_h * 0.90
            head = [it for it in items if it["y"] < top]
            foot = [it for it in items if it["y"] > bottom]
            body = [it for it in items if top <= it["y"] <= bottom]
            head.sort(key=lambda it: (it["y"], it["x0"]))
            foot.sort(key=lambda it: (it["y"], it["x0"]))

            # 双栏检测:以中线为界，两侧都有 25% 以上的行才判定为双栏，
            # 避免把正文居中标题误判成右栏。
            mid = img_w / 2
            left = [it for it in body if it["x0"] < mid]
            right = [it for it in body if it["x0"] >= mid]
            two_col = (
                len(body) >= 8
                and len(left) >= 0.25 * len(body)
                and len(right) >= 0.25 * len(body)
            )
            if two_col:
                left.sort(key=lambda it: (it["y"], it["x0"]))
                right.sort(key=lambda it: (it["y"], it["x0"]))
                ordered = left + right
            else:
                body.sort(key=lambda it: (it["y"], it["x0"]))
                ordered = body
            return [it["t"] for it in head + ordered + foot]
        except Exception:
            return list(texts)

    def recognize(self, image_bytes: bytes) -> str:
        """识别图片中的文字(对外主入口)。

        先确保后端已初始化；若无可用引擎则返回空串。随后按后端类型执行:
        PaddleOCR 需先落临时文件再调用 `predict()/ocr()`，并兼容 3.x 的
        `rec_texts` 与 2.x 的列表两种返回格式；pytesseract 则直接在内存中
        解码图片并识别。最终将各文本行用换行符拼接返回。

        参数:
            image_bytes (bytes): 图片的二进制内容。

        返回:
            str: 识别出的文字(多行以 "\\n" 连接)；无文字或无引擎时返回空串。

        异常:
            无:后端初始化与识别过程中的异常均被 `_init_backend` 内部吞掉；
            PaddleOCR 临时文件无论成败都会在 `finally` 中清理。
        """
        self._ensure_init()
        if self._backend == "none":
            return ""
        if self._backend == "paddleocr":
            import tempfile, io, gc
            img_w = img_h = 0
            # 归一化图片尺寸:扫描件单页常为 2480x3484 大图，CPU 推理内存压力大，
            # 长边压到 1600 以内既保证识别率又降低显存/内存峰值。
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img_w, img_h = im.size
                w, h = img_w, img_h
                max_side = 1600
                if max(w, h) > max_side:
                    scale = max_side / max(w, h)
                    im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                    img_w, img_h = im.size
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                image_bytes = buf.getvalue()
            except Exception:
                pass  # 缩放失败则用原图继续
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            try:
                # 兼容 PaddleOCR 3.x(推荐 predict())与 2.x(ocr())
                if hasattr(self._ocr, "predict"):
                    result = self._ocr.predict(tmp_path)
                else:
                    result = self._ocr.ocr(tmp_path)
                if not result:
                    return ""
                ocr_result = result[0]
                # PaddleOCR 3.x: OCRResult 是类字典对象，识别文本在 'rec_texts' 中
                if hasattr(ocr_result, "keys") and "rec_texts" in ocr_result:
                    texts = ocr_result.get("rec_texts") or []
                    boxes = ocr_result.get("rec_boxes")
                    if boxes is not None and len(boxes) == len(texts):
                        texts = self._order_by_layout(texts, boxes, img_w, img_h)
                    return "\n".join(str(t) for t in texts)
                # PaddleOCR 2.x: 列表格式 [ [bbox, (text, score)], ... ]
                texts = []
                for line in ocr_result:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        texts.append(line[1][0])
                return "\n".join(str(t) for t in texts)
            except Exception as e:
                # 单页识别失败不应中断整份文档的入库，记录后跳过该页图片。
                logger.warning(f"OCR 识别失败(跳过该页图片): {type(e).__name__}: {e}")
                return ""
            finally:
                os.unlink(tmp_path)
                # 及时回收推理中间对象，避免长文档批量 OCR 时内存持续累积。
                gc.collect()
        elif self._backend == "pytesseract":
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes))
            return self._ocr.image_to_string(img, lang="chi_sim+eng")
        return ""