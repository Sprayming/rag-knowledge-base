#!/bin/bash
# 法律文书 RAG：macOS 一键启动（FastAPI 8000，前端由服务自身托管）
# 说明：
#   - Embedding 用本地 BGE-M3（model_cache/bge-m3），离线加载
#   - Redis 未安装时自动降级为内存记忆（日志会提示 Redis 连接失败，属正常）
#   - OCR（paddleocr）未安装时，扫描件退化为纯文字提取；需要 OCR 再单独建 .ocr_venv
#   - Reranker 模型未下载时自动跳过，RRF 结果直接作为 Top-K
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="/Users/sprayming/Documents/Trae-CN/temp_20261116_01"
LOG="$WORKSPACE/.tools/logs"
mkdir -p "$LOG"

PY="$ROOT/.venv/bin/python"
# 有 .ocr_venv 时优先用它（Python 3.12 + paddle 3.3.1 + paddleocr 3.7，含全部运行依赖），
# 与原作者 Windows 上的做法一致：整个应用跑在带 OCR 的解释器里，扫描件才能识别
if [ -x "$ROOT/.ocr_venv/bin/python" ]; then
  PY="$ROOT/.ocr_venv/bin/python"
  PY_DESC=".ocr_venv（含 OCR）"
else
  PY_DESC=".venv（无 OCR）"
fi
REDIS_BIN="$HOME/.local/opt/redis/bin/redis-server"
port_up() { lsof -ti "tcp:$1" >/dev/null 2>&1; }

[ -x "$PY" ] || { echo "缺少 .venv，先建环境"; exit 1; }

echo "==> 1/2 Redis (6379)"
if port_up 6379; then
  echo "    已在运行"
elif [ -x "$REDIS_BIN" ]; then
  "$REDIS_BIN" --port 6379 --bind 127.0.0.1 \
    --dir "$WORKSPACE/.tools/redis-data" --save "" --appendonly no \
    > "$LOG/redis.log" 2>&1 &
  sleep 1
  "$HOME/.local/opt/redis/bin/redis-cli" -p 6379 ping >/dev/null 2>&1 \
    && echo "    启动成功" || echo "    启动失败，服务会降级为内存记忆"
else
  echo "    未安装，服务将降级为内存记忆"
fi

echo "==> 2/2 法律文书 RAG (8000) — $PY_DESC"
if port_up 8000; then
  echo "    端口 8000 已被占用（小哲电商 / 司机卡券的 Agent 后端也用 8000），先停掉那个项目"
  exit 1
fi

cd "$ROOT" || exit 1
"$PY" run.py > "$LOG/rag-run.log" 2>&1 &
echo "    启动中，日志 $LOG/rag-run.log"
echo "    （首次启动要加载 BGE-M3，约 20-40 秒）"
sleep 4
echo
echo "Web 界面：http://localhost:8000"
echo "接口文档：http://localhost:8000/docs"
echo "演示账号：demo / demo123456"
echo "停止服务：./stop-mac.sh"
