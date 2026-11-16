@echo off
title RAG Knowledge Base (FastAPI Local)
echo ========================================
echo   Starting RAG Knowledge Base (Local)
echo ========================================
echo.

cd /d "D:\Courses\rag-knowledge-base"

echo [1/2] Installing/updating dependencies...
pip install -r requirements-docker.txt >nul 2>&1

echo [2/2] Starting FastAPI backend...
echo   Open: http://localhost:8501
echo.
uvicorn app.main:app --host 127.0.0.1 --port 8501

pause