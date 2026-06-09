#!/bin/bash
# ============================================
# Search-R1 A6000 优化训练脚本
# ============================================

set -e  # 遇到错误立即退出

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# ========== 项目配置 ==========
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ========== 训练配置 ==========
export DATA_DIR='data/nq_search'
export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME='a6000-nq-search-r1-grpo-qwen2.5-7b'
export CONFIG_FILE='configs/a6000_grpo.yaml'

# ========== 输出配置 ==========
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

# ========== 打印配置信息 ==========
echo "🚀 Search-R1 A6000 优化训练"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 硬件配置:"
echo "   GPU: A6000 (48GB显存)"
echo ""
echo "🤖 模型配置:"
echo "   模型: $BASE_MODEL"
echo "   算法: GRPO"
echo ""
echo "📈 训练配置:"
echo "   batch_size: 16"
echo "   n_agent: 5"
echo "   训练步数: 500"
echo ""
echo "⏱️  预期性能:"
echo "   训练时间: 4-5小时"
echo "   显存占用: ~38-42GB"
echo ""
echo "📁 输出路径:"
echo "   日志: $LOG_FILE"
echo "   检查点: checkpoints/$EXPERIMENT_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ========== 检查前置条件 ==========
echo "🔍 检查前置条件..."

# 检查GPU
if ! nvidia-smi &> /dev/null; then
    echo "❌ GPU不可用"
    exit 1
fi

# 检查数据文件
if [ ! -f "$DATA_DIR/train.json" ]; then
    echo "❌ 数据文件不存在: $DATA_DIR/train.json"
    echo "请先运行: python scripts/data_prepare.py"
    exit 1
fi

# 检索检查索器
if ! curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    echo "⚠️  检索器未启动，训练可能失败"
    echo "建议先启动检索器"
    echo ""
fi

echo "✅ 前置条件检查完成"
echo ""

# ========== 开始训练 ==========
echo "🎯 开始训练..."
echo ""

# 使用配置文件启动训练
python3 -m verl.trainer.main_ppo --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# ========== 训练完成 ==========
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 训练完成！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 查看日志:"
echo "   cat $LOG_FILE"
echo ""
echo "📁 检查点:"
echo "   ls checkpoints/$EXPERIMENT_NAME"
echo ""
