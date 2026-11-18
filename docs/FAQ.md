# 常见技术问题

## 面试常见问题

### Q1: BM25 + Dense + RRF 为什么不用纯语义？
语义检索对低频业务术语易漏召回。BM25 提供关键词匹配兜底，RRF 融合保证召回。

### Q2: Cross-Encoder vs Bi-Encoder？
Bi-Encoder 离线向量化适合召回，Cross-Encoder 交互式计算适合精排。管线先召回 50 条，Cross-Encoder 重排选 Top-5。

### Q3: 分块大小为什么 500？
基于制度条款长度统计，500 字符确保单块覆盖完整条款。

### Q4: RAGAS 指标？
faithfulness（忠实）+ answer_relevancy（切题）+ context_precision（精确）+ context_recall（召回）

### Q5: PDF 多模态解析？
`MultimodalPipeline` 对每页：① PyMuPDF 抽文字层；② 页面 / 内嵌图片交 `OCREngine`（PaddleOCR）识别文字；③ 图文统一分块 → BGE-M3 向量化。无文字层的扫描件靠 OCR 补齐（PaddleOCR 3.7，`predict()` + `rec_texts`），无需 pdf2image 中转。

### Q6: 记忆系统？
三层：短期（Redis 原文）-> 中期（LLM 摘要）-> 长期（ChromaDB 向量 + 遗忘曲线）。

### Q7: 为什么从 text2vec 换成豆包？
本地 Embedding 有部署/更新/硬件成本。豆包 API 零维护，支持中文知识库场景。

### Q8: 角色系统？
SQLite role 字段 + 前端 JS 校验 + 后端 API 二次校验防止越权。
