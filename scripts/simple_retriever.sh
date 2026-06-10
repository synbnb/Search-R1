#!/bin/bash
# ============================================
# Search-R1 简化检索服务器启动脚本
# 使用在线搜索引擎，无需下载索引
# ============================================

set -e

# ========== 颜色输出 ==========
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ========== 项目配置 ==========
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ========== 配置信息 ==========
RETRIEVER_TYPE="online"  # 在线搜索引擎

echo -e "${GREEN}🔍 Search-R1 在线检索服务器${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 配置信息:"
echo "   检索器类型: 在线搜索引擎"
echo "   搜索引擎: 可配置 (Google/Bing/DuckDuckGo)"
echo "   无需下载索引文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ========== 检查API密钥 ==========
echo -e "${YELLOW}🔍 检查搜索引擎配置...${NC}"

if [ -z "$GOOGLE_API_KEY" ] && [ -z "$BING_API_KEY" ]; then
    echo -e "${RED}❌ 未配置搜索引擎API密钥${NC}"
    echo ""
    echo "💡 请设置以下环境变量之一："
    echo "   export GOOGLE_API_KEY='your-key'"
    echo "   export BING_API_KEY='your-key'"
    echo ""
    echo "📘 获取API密钥："
    echo "   Google: https://console.cloud.google.com/"
    echo "   Bing: https://www.microsoft.com/cognitiveservices"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ API密钥已配置${NC}"
echo ""

# ========== 启动在线检索服务器 ==========
echo -e "${GREEN}🚀 启动在线检索服务器...${NC}"
echo ""

# 检查是否有在线检索脚本
if [ -f "search_r1/search/serp_search_server.py" ]; then
    echo "使用SERP检索服务器..."
    python search_r1/search/serp_search_server.py &
    SERVER_PID=$!
else
    echo -e "${RED}❌ 在线检索脚本不存在${NC}"
    echo "请检查 search_r1/search/serp_search_server.py"
    exit 1
fi

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
    echo "   类型: 在线搜索引擎"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "💡 提示:"
    echo "   查看日志: tail -f /tmp/retriever_online.log"
    echo "   停止服务器: kill $SERVER_PID"
    echo ""
else
    echo -e "${RED}❌ 检索服务器启动失败${NC}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
