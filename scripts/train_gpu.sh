#!/bin/bash
# ============================================
# Search-R1 灵活GPU训练脚本
# 支持自定义GPU选择和配置
# ============================================

set -e  # 遇到错误立即退出

# ========== 颜色定义 ==========
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ========== 默认配置 ==========
DEFAULT_GPU_ID=0
DEFAULT_CONFIG="configs/a6000_grpo.yaml"
DEFAULT_DATA_DIR="data/nq_search"
DEFAULT_MODEL="Qwen/Qwen2.5-7B"

# ========== 参数解析 ==========
GPU_ID="${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU_ID}"
CONFIG_FILE="$DEFAULT_CONFIG"
DATA_DIR="$DEFAULT_DATA_DIR"
BASE_MODEL="$DEFAULT_MODEL"

# ========== 帮助信息 ==========
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --gpu GPU_ID           指定GPU设备ID (默认: 0)"
    echo "  --config CONFIG_FILE   指定配置文件 (默认: configs/a6000_grpo.yaml)"
    echo "  --data DATA_DIR        指定数据目录 (默认: data/nq_search)"
    echo "  --model MODEL_PATH     指定模型路径 (默认: Qwen/Qwen2.5-7B)"
    echo "  --name EXPERIMENT_NAME 指定实验名称"
    echo "  --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 使用默认GPU (GPU 0)"
    echo "  $0"
    echo ""
    echo "  # 使用GPU 1"
    echo "  $0 --gpu 1"
    echo ""
    echo "  # 使用多个GPU (0,1,2)"
    echo "  $0 --gpu 0,1,2"
    echo ""
    echo "  # 使用PPO配置"
    echo "  $0 --config configs/a6000_ppo.yaml"
    echo ""
    echo "  # 使用特定模型"
    echo "  $0 --model meta-llama/Llama-3.2-3B"
    echo ""
    echo "  # 完整自定义"
    echo "  $0 --gpu 2 --config configs/a6000_ppo.yaml --name my-experiment"
    exit 0
}

# ========== 解析命令行参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data)
            DATA_DIR="$2"
            shift 2
            ;;
        --model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            ;;
    esac
done

# ========== 自动生成实验名称 ==========
if [ -z "$EXPERIMENT_NAME" ]; then
    # 从配置文件名提取算法类型
    if [[ "$CONFIG_FILE" == *"grpo"* ]]; then
        ALGORITHM="grpo"
    elif [[ "$CONFIG_FILE" == *"ppo"* ]]; then
        ALGORITHM="ppo"
    else
        ALGORITHM="custom"
    fi

    # 从模型路径提取模型名
    MODEL_NAME=$(basename "$BASE_MODEL")
    EXPERIMENT_NAME="gpu${GPU_ID}-${ALGORITHM}-${MODEL_NAME}"
fi

# ========== 环境配置 ==========
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

# ========== 项目配置 ==========
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ========== 输出配置 ==========
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}.log"

# ========== 打印配置信息 ==========
echo -e "${BLUE}🚀 Search-R1 灵活GPU训练${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}📊 硬件配置:${NC}"
echo "   GPU设备: $GPU_ID"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

echo -e "${GREEN}🤖 模型配置:${NC}"
echo "   模型: $BASE_MODEL"
echo "   配置文件: $CONFIG_FILE"
echo "   算法: $ALGORITHM"
echo ""

echo -e "${GREEN}📁 数据配置:${NC}"
echo "   数据目录: $DATA_DIR"
echo ""

echo -e "${GREEN}🎯 实验配置:${NC}"
echo "   实验名称: $EXPERIMENT_NAME"
echo "   日志文件: $LOG_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ========== 检查前置条件 ==========
echo -e "${YELLOW}🔍 检查前置条件...${NC}"

# 检查GPU是否可用
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ GPU不可用${NC}"
    exit 1
fi

# 检查指定的GPU是否存在
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
IFS=',' read -ra GPU_ARRAY <<< "$GPU_ID"
for gpu in "${GPU_ARRAY[@]}"; do
    if [ "$gpu" -ge "$GPU_COUNT" ]; then
        echo -e "${RED}❌ GPU $gpu 不存在 (系统只有 $GPU_COUNT 个GPU)${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ GPU检查通过${NC}"

# 检查数据文件
if [ ! -f "$DATA_DIR/train.json" ]; then
    echo -e "${RED}❌ 数据文件不存在: $DATA_DIR/train.json${NC}"
    echo "请先运行: python scripts/data_prepare.py --output_dir $DATA_DIR"
    exit 1
fi
echo -e "${GREEN}✅ 数据文件存在${NC}"

# 检索检查索器
if ! curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    echo -e "${YELLOW}⚠️  检索器未启动，训练可能失败${NC}"
    echo "建议先启动检索器: bash scripts/start_retriever.sh"
    echo ""
fi

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 配置文件存在${NC}"

echo ""

# ========== 显示GPU信息 ==========
echo -e "${BLUE}🎮 当前GPU状态:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -v gpu_id="$GPU_ID" 'BEGIN{split(gpu_id, gpus, ",");} {for(i in gpus){if($1==gpus[i]){print "   GPU " $0;}}}'
echo ""

# ========== 开始训练 ==========
echo -e "${GREEN}🎯 开始训练...${NC}"
echo ""

# 使用配置文件启动训练，支持命令行参数覆盖
python3 -m verl.trainer.main_ppo \
    --config "$CONFIG_FILE" \
    data.train_files="$DATA_DIR/train.json" \
    data.val_files="$DATA_DIR/test.json" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="checkpoints/$EXPERIMENT_NAME" \
    2>&1 | tee "$LOG_FILE"

# ========== 训练完成 ==========
EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}🎉 训练完成！${NC}"
    echo ""
    echo "📊 查看日志:"
    echo "   cat $LOG_FILE"
    echo ""
    echo "📁 检查点:"
    echo "   ls checkpoints/$EXPERIMENT_NAME"
else
    echo -e "${RED}❌ 训练失败 (退出码: $EXIT_CODE)${NC}"
    echo ""
    echo "📊 查看错误日志:"
    echo "   tail -100 $LOG_FILE"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $EXIT_CODE
