#!/bin/bash
# ============================================
# Search-R1 检索服务器启动脚本
# ============================================

set -e

# ========== 配置参数 ==========
INDEX_PATH="data/index/e5_Flat.index"
CORPUS_PATH="data/index/wiki-18.jsonl"
RETRIEVER_NAME="e5"
RETRIEVER_MODEL="intfloat/e5-base-v2"
TOPK=3
LOG_DIR="logs"

# ========== 颜色输出 ==========
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ========== 项目配置 ==========
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ========== 创建日志目录 ==========
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/retriever.log"

# ========== 打印配置信息 ==========
echo -e "${GREEN}🔍 Search-R1 检索服务器${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 配置信息:"
echo "   索引路径: $INDEX_PATH"
echo "   语料路径: $CORPUS_PATH"
echo "   检索模型: $RETRIEVER_MODEL"
echo "   Top-K: $TOPK"
echo ""
echo "📁 日志文件: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ========== 检查前置条件 ==========
echo "🔍 检查前置条件..."

# 检查索引文件
if [ ! -f "$INDEX_PATH" ]; then
    echo -e "${RED}❌ 索引文件不存在: $INDEX_PATH${NC}"
    echo "请先运行: python scripts/download_index.py"
    exit 1
fi

# 检查语料文件
if [ ! -f "$CORPUS_PATH" ]; then
    echo -e "${RED}❌ 语料文件不存在: $CORPUS_PATH${NC}"
    echo "请先运行: python scripts/download_index.py"
    exit 1
fi

echo -e "${GREEN}✅ 前置条件检查完成${NC}"
echo ""

# ========== 检查端口占用 ==========
echo "🔍 检查端口8000..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  端口8000已被占用${NC}"
    echo "请先关闭现有的检索服务器"
    echo "或者修改脚本使用其他端口"
    exit 1
fi
echo -e "${GREEN}✅ 端口8000可用${NC}"
echo ""

# ========== 启动检索服务器 ==========
echo -e "${GREEN}🚀 启动检索服务器...${NC}"
echo ""

# 启动服务器并记录日志
python search_r1/search/retrieval_server.py \
    --index_path "$INDEX_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --topk "$TOPK" \
    --faiss_gpu \
    2>&1 | tee "$LOG_FILE" &

# 获取进程ID
SERVER_PID=$!
echo "服务器进程ID: $SERVER_PID"
echo ""

# ========== 等待服务器启动 ==========
echo "⏳ 等待服务器启动..."
sleep 5

# ========== 验证服务器状态 ==========
echo ""
echo "🔍 验证服务器状态..."
if curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    echo -e "${GREEN}✅ 检索服务器启动成功！${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 服务器信息:"
    echo "   URL: http://127.0.0.1:8000/retrieve"
    echo "   PID: $SERVER_PID"
    echo "   日志: $LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 提示:"
    echo "   查看日志: tail -f $LOG_FILE"
    echo "   停止服务器: kill $SERVER_PID"
    echo ""
else
    echo -e "${RED}❌ 检索服务器启动失败${NC}"
    echo "请检查日志: tail $LOG_FILE"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
