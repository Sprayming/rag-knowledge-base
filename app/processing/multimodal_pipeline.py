
"""
app/processing/multimodal_pipeline —— 多模态文档处理管线

【作用与功能】
本模块是文档入库链路的编排核心:接收 PDF 或纯文本路径，依次完成
“图文提取 → OCR 文字识别 → Vision 图片描述 → 文本合并 → 分块”，
最终产出可向量化的 `MultimodalChunk` 列表。它把 `pdf_extractor`、
`ocr_engine` 与 `app.ingestion.vision_caption` 串成一条完整管线。

【主要组成】
- `MultimodalChunk`:多模态文本块数据类(文本 + 页码 + 关联图片引用)。
- `MultimodalPipeline`:管线主类，`process()` 对外入口，
  `_process_pdf()` 负责 PDF 的完整处理流程。

【适用场景】
- 场景1:将企业文档 PDF 转换为带图片引用与 OCR 文字的语义片段，供向量化入库。
- 场景2:非 PDF 文本文件直接读取为单块，走简化分支。

【依赖关系】
- 上游调用方:文档入库流程、检索编排层。
- 下游依赖:`app.processing.pdf_extractor.extract_pdf_pages`、
  `app.processing.ocr_engine.OCREngine`、`app.ingestion.vision_caption.VisionCaptioner`、
  `langchain_text_splitters.RecursiveCharacterTextSplitter`。
"""
import re
from pathlib import Path
from app.processing.pdf_extractor import extract_pdf_pages
from app.processing.ocr_engine import OCREngine
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 目录行的引导符：纯点/省略号/破折号，或带省略号再加页码
_TOC_LEADER_RE = re.compile(r"^[\s.．·…—\-]*$")
# 页码行：纯数字、第X页、第X页/共Y页 等格式
_PAGE_NO_RE = re.compile(
    r"^[\s\-—–_·.]*\d{1,4}[\s\-—–_·.]*$"
    r"|^第\s*\d{1,4}\s*页(\s*[/／]\s*共\s*\d{1,4}\s*页)?$"
)


def detect_page_furniture(page_texts: list, min_pages: int = 3,
                          ratio: float = 0.4, max_len: int = 80) -> set:
    """找出在多数页面重复出现的短行(页眉、页脚、水印)。

    统计每页的短行文本，出现在足够多页面上的即为版式噪声。
    页码行因每页不同无法按文本匹配，由 `strip_page_furniture` 用正则另行剔除。

    参数:
        page_texts (list[str]): 各页原始文本。
        min_pages (int): 至少出现在多少个页面上才算噪声。
        ratio (float): 同时要求出现在总页数的该比例以上。
        max_len (int): 只考虑长度不超过该值的行，避免误删正文段落。

    返回:
        set[str]: 归一化(去空白)后的噪声行文本集合。
    """
    if len(page_texts) < min_pages:
        return set()
    threshold = max(min_pages, int(len(page_texts) * ratio))

    counter: dict = {}
    for text in page_texts:
        seen = set()
        for line in text.split("\n"):
            s = line.strip()
            if not s or len(s) > max_len:
                continue
            key = re.sub(r"\s+", "", s)
            if key in seen:
                continue
            seen.add(key)
            counter[key] = counter.get(key, 0) + 1

    return {k for k, c in counter.items() if c >= threshold}


def strip_page_furniture(text: str, furniture: set) -> str:
    """从页面文本中剔除页眉页脚与页码行，保留原有空行结构。

    参数:
        text (str): 单页文本。
        furniture (set[str]): `detect_page_furniture` 的结果。

    返回:
        str: 清洗后的页面文本。
    """
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if re.sub(r"\s+", "", s) in furniture or _PAGE_NO_RE.match(s):
            continue
        kept.append(line)
    return "\n".join(kept)


def is_toc_noise(text: str) -> bool:
    """判断文本块是否为目录页片段。

    目录条目用省略号连接标题与页码，正文条款中极少出现连续省略号，
    因此用省略号密度 + 纯引导符行占比两个特征识别，避免误删正文。

    参数:
        text (str): 待判断的文本块。

    返回:
        bool: True 表示该块是目录噪声，应丢弃。
    """
    if text.count("…") >= 5:
        return True
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    dotted = sum(1 for ln in lines if _TOC_LEADER_RE.match(ln) or "…" in ln)
    return len(lines) >= 3 and dotted / len(lines) > 0.5


class MultimodalChunk:
    """多模态文本块 - 包含文本 + 关联图片。

    表示文档分块后的一个语义单元，除纯文本外还携带其来源页码与
    关联图片的索引/格式引用，便于后续“以文搜图”与上下文回溯。

    属性:
        text (str): 该文本块的正文内容(含 OCR 文字与图片描述)。
        page_number (int): 文本块所属的原文档页码(从 1 开始)。
        images (list): 关联图片引用列表，元素形如
            {"index": 页内图片序号, "ext": 图片扩展名}。
    """
    def __init__(self, text: str, page_number: int, images: list = None):
        """初始化多模态文本块。

        参数:
            text (str): 文本块正文。
            page_number (int): 来源页码(从 1 开始)。
            images (list, optional): 关联图片引用列表，为 None 时置为空列表。
        """
        self.text = text
        self.page_number = page_number
        self.images = images or []


class MultimodalPipeline:
    """多模态文档处理管线: PDF → 图文提取 → OCR → 分块 → 向量化。

    管线对外仅暴露 `process()` 方法:根据文件扩展名分发到 PDF 处理或
    纯文本处理分支。内部惰性创建 OCR 引擎与文本分块器，并在 PDF 分支中
    调用 VisionCaptioner 为图片生成语义描述。
    """

    def __init__(self):
        """初始化多模态处理管线。

        构造时即创建 OCR 引擎(内部为惰性后端加载，不会立刻占用资源)
        与递归字符分块器(针对中文优化的分隔符与 500 字符块大小)。
        """
        self._ocr = OCREngine()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", "，", " "],
        )

    def process(self, file_path: str) -> list[MultimodalChunk]:
        """处理文档: 提取图文 + OCR + 组装文本块。

        根据文件后缀分发处理分支:PDF 走完整的 `_process_pdf()` 流程；
        其余文本文件直接读取为单个多模态块(页码记为 1，无图片)。

        参数:
            file_path (str): 待处理文档的本地路径。

        返回:
            list[MultimodalChunk]: 分块后的多模态文本块列表。

        异常:
            无:非 PDF 文本读取失败时会以 Python 内置异常向上抛出；
            PDF 处理中各页 OCR/Vision 失败均被静默忽略，仅跳过空页。
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        else:
            # 非 PDF 文件: 直接读取文本
            text = Path(file_path).read_text(encoding="utf-8")
            return [MultimodalChunk(text=text, page_number=1)]

    def _process_pdf(self, pdf_path: str) -> list[MultimodalChunk]:
        """处理 PDF 文件(内部方法)。

        逐页执行:抽取页面文本与图片 → 对每张图片做 OCR 与 Vision 描述
        → 将页面文本、图片描述、OCR 文字合并为一个字符串 → 按块大小切分
        → 为每块附带整页图片引用，生成 `MultimodalChunk` 列表。

        参数:
            pdf_path (str): PDF 文件路径。

        返回:
            list[MultimodalChunk]: 该 PDF 分块后的多模态文本块列表。

        适用场景:
            - 由 `process()` 在检测到 `.pdf` 后缀时调用，通常不对外直接调用。
        """
        pages = extract_pdf_pages(pdf_path)
        # 先跨页统计重复出现的页眉/页脚，再逐页剔除，避免版式噪声进入切片
        furniture = detect_page_furniture([p["text"] for p in pages])
        chunks = []

        for page in pages:
            page_text = strip_page_furniture(page["text"], furniture)
            page_images = page["images"]
            ocr_texts = []

            # OCR 识别图片中的文字
            for img in page_images:
                ocr_result = self._ocr.recognize(img["bytes"])
                if ocr_result:
                    ocr_texts.append(ocr_result)

            # AI 图片描述 (VisionCaption)
            caption_texts = []
            try:
                from app.ingestion.vision_caption import VisionCaptioner
                import app.core.config as cfg
                captioner = VisionCaptioner(api_key=cfg.LLM_API_KEY, base_url=cfg.LLM_BASE_URL)
                for img in page_images:
                    try:
                        caption = captioner.caption(img["bytes"], img.get("ext", "png"))
                        if caption:
                            caption_texts.append("[图片描述] " + caption)
                    except Exception:
                        pass
            except Exception:
                pass

            # 页面文本 + OCR 文字合并
            combined = page_text
            if caption_texts:
                combined += "\n" + "\n".join(caption_texts)
            if ocr_texts:
                combined += "\n[图片文字]\n" + "\n".join(ocr_texts)

            if not combined.strip():
                continue

            # 分块
            split_texts = self._splitter.split_text(combined)
            for st in split_texts:
                # 目录页片段对检索只有噪声，入库前丢弃(正文条款不含连续省略号)
                if is_toc_noise(st):
                    continue
                # 仅为本块保留整页图片引用(不重复内嵌二进制)，供以文搜图回溯
                img_refs = [{"index": img["index"], "ext": img["ext"]} for img in page_images]
                chunks.append(MultimodalChunk(text=st, page_number=page["page_number"], images=img_refs))

        return chunks
