# Search-R1 最小化复现指南

## 📚 目录
- [1. 硬件要求](#1-硬件要求)
- [2. 环境准备](#2-环境准备)
- [3. 快速开始](#3-快速开始)
- [4. 最小化配置说明](#4-最小化配置说明)
- [5. 故障排查](#5-故障排查)

---

## 1. 硬件要求

### 1.1 推荐配置（最小化）

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **GPU** | 1× RTX 3090 (24GB) | 1× RTX 4090 (24GB) |
| **CPU** | 8核 | 16核 |
| **内存** | 32GB | 64GB |
| **硬盘** | 100GB SSD | 200GB SSD |

### 1.2 显存占用分析

基于Qwen2.5-3B模型：
```
最小化配置显存占用（单GPU RTX 3090 24GB）:

┌─────────────────────────────────────────────────┐
│  模型组件              │ 显存占用    │
├─────────────────────────────────────────────────┤
│  Qwen2.5-3B (FP16)     │ ~6GB       │
│  vLLM推理引擎          │ ~8GB       │
│  FSDP参数/梯度缓存     │ ~4GB       │
│  优化器状态            │ ~2GB       │
│  激活值/缓存           │ ~4GB       │
├─────────────────────────────────────────────────┤
│  总计                  │ ~24GB      │
└─────────────────────────────────────────────────┘
```

---

## 2. 环境准备

### 2.1 基础环境安装

```bash
# 创建conda环境
conda create -n searchr1_minimal python=3.10
conda activate searchr1_minimal

# 安装PyTorch (GPU版本)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装vLLM (小版本，显存占用更少)
pip install vllm==0.4.2

# 安装基础依赖
pip install transformers datasets
pip install ray==2.9.0
pip install hydra-core
pip install wandb
pip install accelerate
pip install flash-attn --no-build-isolation

# 安装veRL框架
cd /home/devuser/workspace/agentrl/Search-R1
pip install -e .
```

### 2.2 检索器环境（轻量级，无需GPU）

```bash
# 创建独立环境（可选，也可以在searchr1环境中安装）
conda create -n retriever_minimal python=3.10
conda activate retriever_minimal

# 安装基础依赖
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pyserini

# 注意：BM25检索器不需要GPU，也不需要faiss-gpu
# 只需要Pyserini用于稀疏检索
```

---

## 3. 快速开始

### 3.1 最小化复现方案概述

```
方案：单GPU + 小模型 + BM25检索器 + GRPO算法
模型：Qwen/Qwen2.5-3B (比Llama3.2-3B更小)
检索：BM25 (无需GPU)
算法：GRPO (无需Critic网络，省显存)
数据：NQ数据集（小规模测试）
训练：100步快速验证
```

### 3.2 步骤1：数据准备

```bash
# 创建数据目录
mkdir -p data/minimal
cd data/minimal

# 下载并处理NQ数据集（小规模）
python -c "
from datasets import load_dataset
import json

# 加载NQ数据集
dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

# 只取前100个样本用于快速测试
train_data = dataset['train'].select(range(100))
test_data = dataset['test'].select(range(10))

# 保存为简化格式
with open('train_simple.json', 'w') as f:
    for item in train_data:
        f.write(json.dumps({
            'question': item['question'],
            'golden_answers': item['golden_answers']
        }) + '\n')

with open('test_simple.json', 'w') as f:
    for item in test_data:
        f.write(json.dumps({
            'question': item['question'],
            'golden_answers': item['golden_answers']
        }) + '\n')

print('数据准备完成！')
print('训练集：100个样本')
print('测试集：10个样本')
"
```

### 3.3 步骤2：准备简化检索器

```bash
# 使用Wikipedia-18的小型索引（预先下载的）
# 如果没有索引，可以跳过检索器，使用简化版测试

cd /home/devuser/workspace/agentrl/Search-R1

# 创建简化检索器（无需真实索引）
python -c "
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    queries: list
    topk: int = 3
    return_scores: bool = False

# 模拟的简单检索结果（仅用于测试）
MOCK_RESULTS = {
    'default': [
        {
            'document': {
                'contents': '\"George Washington\"\n乔治·华盛顿是美国第一任总统，于1789年就职...'
            },
            'score': 0.95
        },
        {
            'document': {
                'contents': '\"美国总统\"\n美国总统列表包括华盛顿、亚当斯、杰斐逊...'
            },
            'score': 0.87
        },
        {
            'document': {
                'contents': '\"华盛顿就职\"\n华盛顿于1789年4月30日在纽约市就职...'
            },
            'score': 0.76
        }
    ]
}

@app.post('/retrieve')
def retrieve(request: QueryRequest):
    results = []
    for query in request.queries:
        # 返回模拟结果
        results.append([MOCK_RESULTS['default']] * request.topk)
    
    return {'result': results}

if __name__ == '__main__':
    print('启动简化检索器...')
    print('注意：这是模拟检索器，仅用于测试！')
    uvicorn.run(app, host='0.0.0.1', port=8000)
" &
```

### 3.4 步骤3：启动训练

```bash
# 新开一个终端，启动检索器
conda activate retriever_minimal
cd /home/devuser/workspace/agentrl/Search-R1

# 启动简化检索服务
python -c "
import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    queries: list
    topk: int = 3
    return_scores: bool = False

MOCK_RESULTS = {
    'default': [
        {
            'document': {'contents': '\"George Washington\"\n乔治·华盛顿是美国第一任总统，于1789年就职...'},
            'score': 0.95
        },
        {
            'document': {'contents': '\"美国总统\"\n美国总统列表包括华盛顿、亚当斯、杰斐逊...'},
            'score': 0.87
        },
        {
            'document': {'contents': '\"华盛顿就职\"\n华盛顿于1789年4月30日在纽约市就职...'},
            'score': 0.76
        }
    ]
}

@app.post('/retrieve')
def retrieve(request: QueryRequest):
    results = []
    for query in request.queries:
        results.append([MOCK_RESULTS['default']] * request.topk)
    return {'result': results}

if __name__ == '__main__':
    print('启动简化检索器在 http://127.0.0.1:8000')
    uvicorn.run(app, host='0.0.0.1', port=8000)
"
```

### 3.5 步骤4：创建最小化训练脚本

```bash
# 新开一个终端，启动训练
conda activate searchr1_minimal
cd /home/devuser/workspace/agentrl/Search-R1

# 创建最小化训练脚本
cat > train_minimal.sh << 'EOF'
#!/bin/bash

# ========== 最小化配置 ==========
export CUDA_VISIBLE_DEVICES=0  # 只使用1个GPU
export DATA_DIR='data/minimal'
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME='minimal-reproduction'

# ========== GRPO训练（无Critic，省显存）==========
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_simple.json \
    data.val_files=$DATA_DIR/test_simple.json \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    \
    algorithm.adv_estimator=grpo \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.project_name='Search-R1-Minimal' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=40 \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    \
    max_turns=1 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "训练完成！"
echo "检查日志: cat $EXPERIMENT_NAME.log"
EOF

chmod +x train_minimal.sh

# 运行训练
./train_minimal.sh
```

---

## 4. 最小化配置说明

### 4.1 关键参数解释

| 参数 | 原始值 | 最小化值 | 说明 |
|------|--------|----------|------|
| **GPU数量** | 8 | 1 | `CUDA_VISIBLE_DEVICES=0` |
| **batch_size** | 512 | 8 | 大幅减少显存占用 |
| **mini_batch_size** | 256 | 4 | 减少batch内计算量 |
| **micro_batch_size** | 64 | 2 | 减少梯度累积量 |
| **max_prompt_length** | 4096 | 1024 | 减少序列长度 |
| **max_response_length** | 500 | 256 | 减少生成长度 |
| **训练步数** | 1005 | 40 | 快速验证流程 |
| **数据样本数** | 数千 | 100/10 | 最小化数据规模 |

### 4.2 算法选择：GRPO vs PPO

**选择GRPO的原因：**

```
┌─────────────────────────────────────────────────┐
│  PPO (需要Critic网络)                            │
│  ├─ Actor模型: ~6GB                             │
│  ├─ Critic模型: ~6GB                             │
│  ├─ RefPolicy模型: ~6GB                         │
│  └─ 总计: ~18GB                                 │
├─────────────────────────────────────────────────┤
│  GRPO (不需要Critic网络)                          │
│  ├─ Actor模型: ~6GB                             │
│  ├─ RefPolicy模型: ~6GB                         │
│  └─ 总计: ~12GB                                 │
└─────────────────────────────────────────────────┘

节省显存: ~6GB
```

### 4.3 显存优化技巧

```python
# ===== 显存优化配置 =====

# 1. 参数卸载到CPU
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.grad_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

# 2. 减少vLLM显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.85  # 默认0.6，提高利用率

# 3. 梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing=true

# 4. 减少tensor并行
actor_rollout_ref.rollout.tensor_model_parallel_size=1

# 5. 减少batch size
data.train_batch_size=8  # 原始512
actor_rollout_ref.actor.ppo_mini_batch_size=4  # 原始256
```

### 4.4 数据量控制

```python
# ===== 最小化数据集 =====

训练集: 100个样本 (原始数千个)
测试集: 10个样本 (原始数百个)

# 训练步数计算:
total_training_steps = 40
train_batch_size = 8
n_agent (GRPO) = 3

每个iteration处理样本数 = 8 × 3 = 24
总训练样本数 = 40 × 24 = 960次模型调用

# 预计训练时间（单GPU RTX 3090）:
~2-3小时（取决于配置和数据）
```

---

## 5. 故障排查

### 5.1 常见错误及解决方案

#### 错误1: CUDA OOM (显存不足)

```bash
# 错误信息
RuntimeError: CUDA out of memory

# 解决方案
# 1. 进一步减少batch size
data.train_batch_size=4  # 从8减少到4

# 2. 减少序列长度
data.max_prompt_length=512  # 从1024减少到512

# 3. 使用更激进的卸载
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.grad_offload=true
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true

# 4. 降低vLLM显存占用
actor_rollout_ref.rollout.gpu_memory_utilization=0.9  # 提高到0.9
```

#### 错误2: 检索器连接失败

```bash
# 错误信息
requests.exceptions.ConnectionError: http://127.0.0.1:8000/retrieve

# 解决方案
# 1. 检查检索器是否启动
curl http://127.0.0.1:8000/retrieve

# 2. 如果没有启动，使用简化检索器
python simple_retriever.py

# 3. 或者跳过检索，使用本地mock
# 在train_minimal.sh中添加:
export retriever.url="mock"
```

#### 错误3: 模型下载失败

```bash
# 错误信息
OSError: Qwen/Qwen2.5-3B is not a local folder

# 解决方案
# 1. 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 或者使用更小的模型
export BASE_MODEL='TinyLlama/TinyLlama-3.1M'  # 只有3.1M参数

# 3. 或者先下载模型
huggingface-cli download Qwen/Qwen2.5-3B --local-dir ./models/Qwen2.5-3B
export BASE_MODEL='./models/Qwen2.5-3B'
```

#### 错误4: Ray初始化失败

```bash
# 错误信息
RuntimeError: Failed to start Ray

# 解决方案
# 1. 清理Ray缓存
ray stop
rm -rf /tmp/ray

# 2. 限制Ray资源
export RAY_TOKENIZER_PARALLELISM=false

# 3. 使用单机模式
python train_minimal.sh --num-nodes 1
```

### 5.2 监控训练进度

```bash
# ===== 查看训练日志 =====
tail -f minimal-reproduction.log

# ===== 查看显存使用 =====
watch -n 1 nvidia-smi

# ===== 查看训练进度 =====
# 日志中会显示:
# - ACTIVE_TRAJ_NUM: 活跃轨迹数量
# - loss: 训练损失
# - reward: 奖励值
# - 准确率等指标

# ===== 检查检查点 =====
ls -lh checkpoints/minimal-reproduction/
# 会看到:
# - actor/ (Actor模型)
# - checkpoint_*.pt (检查点文件)
```

### 5.3 性能基准

```
最小化配置的预期性能：

硬件: 单GPU RTX 3090 (24GB)
模型: Qwen2.5-3B
数据: 100训练样本，10测试样本
算法: GRPO

┌─────────────────────────────────────────────────┐
│  指标              │ 预期值        │
├─────────────────────────────────────────────────┤
│  训练时间          │ ~2-3小时      │
│  显存占用          │ ~20-22GB      │
│  初始准确率        │ 10-20%        │
│  最终准确率        │ 30-50%        │
│  搜索次数          │ 0.5-1.0次/问题  │
│  格式正确率        │ 60-80%        │
└─────────────────────────────────────────────────┘

注意：
- 由于数据量小，准确率提升有限
- 主要目的是验证流程，不是追求最佳性能
- 如需更好性能，增加数据和训练步数
```

---

## 6. 验证安装是否成功

### 6.1 环境测试

```bash
# ===== 测试1: PyTorch GPU =====
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

```bash
# ===== 测试2: vLLM =====
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-3B', trust_remote_code=True)
print('vLLM测试成功！')
"
```

```bash
# ===== 测试3: Ray =====
python -c "
import ray
ray.init()
print(f'Ray version: {ray.__version__}')
ray.shutdown()
print('Ray测试成功！')
"
```

### 6.2 数据流测试

```bash
# ===== 测试完整数据流 =====
python -c "
import requests
import json

# 1. 测试检索器
response = requests.post(
    'http://127.0.0.1:8000/retrieve',
    json={'queries': ['测试查询'], 'topk': 3, 'return_scores': True}
)
print('检索器测试：', response.status_code == 200)

# 2. 测试模型加载
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', trust_remote_code=True)
print('模型加载测试成功！')

print('所有组件测试通过！')
"
```

---

## 7. 进阶：真实检索器（可选）

如果想要使用真实的检索器而不是模拟版本：

### 7.1 下载小型索引

```bash
# 下载Wikipedia-18的BM25索引（较小版本）
cd /home/devuser/workspace/agentrl/Search-R1

mkdir -p data/index
cd data/index

# 下载预构建的BM25索引
wget https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index/resolve/main/bm25-00000-of-00001.bin
wget https://huggingface.co/datasets/PeterJinGo/wiki-18-bm25-index/resolve/main/bm25-00000-of-00001.bin.json

# 下载小规模语料（用于真实检索）
wget https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz
gunzip wiki-18.jsonl.gz
```

### 7.2 启动真实BM25检索器

```bash
# 启动真实的BM25检索器
conda activate retriever_minimal

python -m verl.trainer.main_generation \
    python search_r1/search/retrieval_server.py \
        --index_path ./data/index \
        --corpus_path ./data/index/wiki-18.jsonl \
        --retriever_name bm25 \
        --topk 3
```

---

## 8. 完整的一键脚本（最简化版）

```bash
#!/bin/bash
# ============================================
# Search-R1 最小化复现一键脚本
# ============================================

set -e  # 遇到错误立即退出

echo "🚀 开始Search-R1最小化复现..."

# ===== 步骤1: 环境检查 =====
echo "📋 检查环境..."
if ! command -v conda &> /dev/null; then
    echo "❌ 请先安装conda"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo "❌ 请确保GPU可用"
    exit 1
fi

echo "✅ 环境检查通过"

# ===== 步骤2: 创建最小化数据 =====
echo "📊 创建最小化数据集..."
mkdir -p data/minimal
cd data/minimal

python -c "
from datasets import load_dataset
import json

dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')
train_data = dataset['train'].select(range(100))
test_data = dataset['test'].select(range(10))

with open('train_simple.json', 'w') as f:
    for item in train_data:
        f.write(json.dumps({
            'question': item['question'],
            'golden_answers': item['golden_answers']
        }, ensure_ascii=False) + '\n')

with open('test_simple.json', 'w') as f:
    for item in test_data:
        f.write(json.dumps({
            'question': item['question'],
            'golden_answers': item['golden_answers']
        }, ensure_ascii=False) + '\n')

print('数据准备完成！100训练 + 10测试')
"

echo "✅ 数据准备完成"

# ===== 步骤3: 启动简化检索器 =====
echo "🔍 启动简化检索器..."
cd /home/devuser/workspace/agentrl/Search-R1

# 后台启动检索器
python -c "
import uvicorn
import json
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    queries: list
    topk: int = 3
    return_scores: bool = False

MOCK_RESULTS = {
    'default': [
        {'document': {'contents': '\"George Washington\"\n乔治·华盛顿是美国第一任总统...'}, 'score': 0.95},
        {'document': {'contents': '\"美国总统\"\n美国总统列表包括华盛顿...'}, 'score': 0.87},
        {'document': {'contents': '\"华盛顿就职\"\n华盛顿于1789年就职...'}, 'score': 0.76}
    ]
}

@app.post('/retrieve')
def retrieve(request: QueryRequest):
    results = []
    for query in request.queries:
        results.append([MOCK_RESULTS['default']] * request.topk)
    return {'result': results}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.1', port=8000)
" &
RETRIEVER_PID=$!

echo "✅ 检索器启动 (PID: $RETRIEVER_PID)"

# 等待检索器启动
sleep 10

# ===== 步骤4: 测试检索器 =====
echo "🧪 测试检索器..."
if curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    echo "✅ 检索器测试通过"
else
    echo "❌ 检索器启动失败"
    kill $RETRIEVER_PID
    exit 1
fi

# ===== 步骤5: 创建最小化训练脚本 =====
echo "📝 创建训练脚本..."
cat > train_minimal.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME='minimal-test'

python3 -m verl.trainer.main_ppo \
    data.train_files=data/minimal/train_simple.json \
    data.val_files=data/minimal/test_simple.json \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.max_start_length=512 \
    data.max_obs_length=256 \
    \
    algorithm.adv_estimator=grpo \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n_agent=3 \
    \
    trainer.n_gpus_per_node=1 \
    trainer.total_training_steps=40 \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    \
    max_turns=1 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3
EOF

chmod +x train_minimal.sh

echo "✅ 训练脚本创建完成"

# ===== 步骤6: 开始训练 =====
echo "🎯 开始最小化训练..."
echo "预计时间: 2-3小时"
echo "显存需求: ~20GB"
echo ""

./train_minimal.sh

# ===== 步骤7: 清理 =====
echo "🧹 清理后台进程..."
kill $RETRIEVER_PID

echo "🎉 最小化复现完成！"
echo "检查结果: cat minimal-test.log"
echo "检查点: ls checkpoints/minimal-test/"
```

---

## 9. 预期输出

### 9.1 训练日志示例

```
[RayPID=xxx] ip-xxx-xxx-xxx.compute.internal:12345
...
ACTIVE_TRAJ_NUM: [8, 5, 3, 2, 1, 0, 0, ...]
Epoch 0/2: 100%|█████████████████████| 40/40 [00:15<00:00,  2.50it/s]
...
Validation: accuracy=0.35
```

### 9.2 检查点文件

```
checkpoints/minimal-test/
├── actor/
│   ├── model.pt
│   └── optimizer.pt
└── checkpoint_20.pt
```

### 9.3 模型推理测试

```bash
# 使用训练好的模型进行推理
python infer.py

# 修改 infer.py 中的模型路径
model_id = "checkpoints/minimal-test/actor"
```

---

## 10. 总结

### 10.1 最小化配置总结

| 组件 | 选择 | 原因 |
|------|------|------|
| **模型** | Qwen2.5-3B | 比Llama3.2-3B更小 |
| **算法** | GRPO | 无需Critic，省6GB显存 |
| **检索器** | 简化模拟版 | 无需GPU，无需下载索引 |
| **数据** | 100训练/10测试 | 快速验证流程 |
| **训练** | 40步 | 2-3小时完成 |
| **硬件** | 1× RTX 3090 | 最小化GPU需求 |

### 10.2 成功标志

✅ 环境安装无错误  
✅ 检索器正常响应  
✅ 训练日志正常输出  
✅ 显存占用 < 24GB  
✅ 检查点正常保存  
✅ 模型可以推理  

### 10.3 下一步

如果最小化复现成功，可以尝试：

1. **增加数据量**: 使用完整的NQ数据集
2. **增加训练步数**: 提升模型性能
3. **使用真实检索器**: 下载Wikipedia索引
4. **更大模型**: 尝试7B参数的模型
5. **多GPU训练**: 提升训练速度

---

*本指南提供了Search-R1的最小化复现方案，适合资源受限的环境进行快速验证和测试。*
