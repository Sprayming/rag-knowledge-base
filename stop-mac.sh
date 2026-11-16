#!/bin/bash
# 停止法律文书 RAG 服务（含 Redis）
pids=$(lsof -ti "tcp:8000" 2>/dev/null)
if [ -n "$pids" ]; then
  echo "$pids" | xargs kill 2>/dev/null
  echo "已停止 RAG 服务 (8000)"
else
  echo "RAG 服务 (8000) 未在运行"
fi

CLI="$HOME/.local/opt/redis/bin/redis-cli"
if [ -x "$CLI" ] && "$CLI" -p 6379 ping >/dev/null 2>&1; then
  "$CLI" -p 6379 shutdown nosave >/dev/null 2>&1
  echo "已停止 Redis (6379)"
fi
