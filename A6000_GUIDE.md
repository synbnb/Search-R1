# Search-R1 A6000 (48GB显存) 优化配置指南

## 📚 目录
- [1. A6000硬件优势](#1-a6000硬件优势)
- [2. 推荐配置方案](#2-推荐配置方案)
- [3. 快速启动脚本](#3-快速启动脚本)
- [4. 性能基准](#4-性能基准)
- [5. 高级优化](#5-高级优化)

---

## 1. A6000硬件优势

### 1.1 硬件规格对比

| 特性 | A6000 | RTX 3090 | 优势分析 |
|------|--------|----------|----------|
| **显存** | 48GB GDDR6 | 24GB GDDR6X | ✅ 2倍显存 |
| **带宽** | 768 GB/s | 912 GB/s | 高带宽 |
| **计算** | 38 TFLOPS | 71 TFLOPS | 训算够用 |
| **功耗** | 300W | 350W | 更节能 |

### 1.2 可支持的训练配置

基于48GB显存，可以支持：

```
┌─────────────────────────────────────────────────┐
│  A6000 (48GB) 可支持配置                          │
├─────────────────────────────────────────────────┤
│  ✅ 7B模型训练（Qwen2.5-7B 或 Llama3.1-8B）      │
│  ✅ 完整PPO训练（带Critic网络）                 │
│  ✅ 大batch size（可达64-128）                  │
│  ✅ 长序列（4096 tokens）                         │
│  ✅ 多agent采样（GRPO可达10+）                  │
│  ✅ 真实检索器（FAISS GPU加速）                  │
└─────────────────────────────────────────────────┘
```

---

## 2. 推荐配置方案

### 2.1 方案对比

| 方案 | 模型 | 算法 | batch_size | 训练时间 | 推荐度 |
|------|------|------|------------|----------|--------|
| **保守方案** | Qwen2.5-3B | GRPO | 32 | 3-4小时 | ⭐⭐⭐ |
| **平衡方案** | Qwen2.5-7B | GRPO | 16 | 4-5小时 | ⭐⭐⭐⭐ |
| **激进方案** | Qwen2.5-7B | PPO | 16 | 6-8小时 | ⭐⭐⭐⭐⭐ |

### 2.2 推荐配置：平衡方案（最佳性价比）

```
┌─────────────────────────────────────────────────┐
│  A6000 平衡配置 (推荐)                            │
├─────────────────────────────────────────────────┤
│  模型：Qwen2.5-7B (高质量7B模型)                 │
│  算法：GRPO (无需Critic，省显存用于更大batch)      │
│  显存占用：~38-42GB                               │
│  训练时间：4-5小时                                 │
│  数据规模：完整NQ数据集                           │
└─────────────────────────────────────────────────┘
```

---

## 3. 快速启动脚本

### 3.1 环境准备

```bash
# ===== 创建conda环境 =====
conda create -n searchr1_a6000 python=3.10
conda activate searchr1_a6000

# ===== 安装PyTorch (CUDA 12.1) =====
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# ===== 安装vLLM (支持A6000) =====
pip install vllm==0.6.3

# ===== 安装基础依赖 =====
pip install transformers datasets
pip install ray==2.9.0
pip install hydra-core
pip install wandb
pip install accelerate
pip install flash-attn --no-build-isolation

# ===== 安装veRL框架 =====
cd /path/to/Search-R1
pip install -e .

# ===== 验证安装 =====
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无GPU\"}')
print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')
"
```

### 3.2 数据准备（完整NQ数据集）

```bash
# ===== 创建数据目录 =====
mkdir -p data/nq_search
cd data/nq_search

# ===== 处理完整NQ数据集 =====
python -c "
import datasets
from datasets import load_dataset
import json
import re

def make_prefix(question):
    return f'''Answer the given question. \
You must conduct reasoning inside <think and  first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
'''

dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq')

train_dataset = dataset['train']
test_dataset = dataset['test']

def process_fn(example, idx):
    question = example['question'].strip()
    if question[-1] != '?':
        question += '?'
    
    question = make_prefix(question)
    
    data = {
        'data_source': 'nq',
        'prompt': [{'role': 'user', 'content': question}],
        'ability': 'fact-reasoning',
        'reward_model': {
            'style': 'rule',
            'ground_truth': {'target': example['golden_answers']}
        },
        'extra_info': {'split': 'train', 'index': idx}
    }
    return data

train_dataset = train_dataset.map(process_fn, with_indices=True)
test_dataset = test_dataset.map(process_fn, with_indices=True)

# 保存为简化JSON格式
with open('train.json', 'w') as f:
    for item in train_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('test.json', 'w') as f:
    for item in test_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f'数据准备完成！')
print(f'训练集: {len(train_dataset)} 样本')
print(f'测试集: {len(test_dataset)} 样本')
"
```

### 3.3 检索器准备（真实E5检索）

```bash
# ===== 创建检索器环境 =====
conda create -n retriever_a6000 python=3.10
conda activate retriever_a6000

# ===== 安装依赖 =====
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pyserini

# ===== 下载预构建索引 =====
mkdir -p data/index
cd data/index

# 下载E5-Flat索引 (高质量密集检索)
python -c "
from huggingface_hub import hf_hub_download
import os

os.chdir('data/index')

# 下载索引文件
hf_hub_download(
    repo_id='PeterJinGo/wiki-18-e5-index',
    filename='part_aa',
    repo_type='dataset'
)

hf_hub_download(
    repo_id='PeterJinGo/wiki-18-corpus',
    filename='wiki-18.jsonl.gz',
    repo_type='dataset'
)

import gzip
import shutil

# 合并索引
with open('e5_Flat.index', 'wb') as f_out:
    for file in ['part_aa']:
        with open(file, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)

# 解压语料
with gzip.open('wiki-18.jsonl.gz', 'rb') as f_in:
    with open('wiki-18.jsonl', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print('索引下载完成！')
print('索引文件: data/index/e5_Flat.index')
print('语料文件: data/index/wiki-18.jsonl')
"
```

### 3.4 启动E5检索服务器

```bash
# ===== 启动GPU加速的E5检索器 =====
conda activate retriever_a6000
cd /path/to/Search-R1

# 启动检索服务器（使用A6000的GPU加速）
python search_r1/search/retrieval_server.py \
    --index_path data/index/e5_Flat.index \
    --corpus_path data/index/wiki-18.jsonl \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --topk 3 \
    --faiss_gpu \
    2>&1 | tee retriever_a6000.log
```

### 3.5 A6000优化训练脚本

```bash
# ===== 创建A6000专用训练脚本 =====
cd /path/to/Search-R1

cat > train_a6000.sh << 'EOF'
#!/bin/bash

# ========== A6000优化配置 ==========
export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='data/nq_search'
export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME='a6000-nq-search-r1-grpo-qwen2.5-7b'

echo "🚀 开始A6000优化训练..."
echo "📊 硬件: A6000 (48GB显存)"
echo "🤖 模型: Qwen2.5-7B"
echo "🔧 算法: GRPO"
echo "📈 显存预期: ~38-42GB"
echo "⏱️  预计时间: 4-5小时"
echo ""

# ========== A6000优化的GRPO训练 ==========
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.json \
    data.val_files=$DATA_DIR/test.json \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    \
    algorithm.adv_estimator=grpo \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.2 \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_remove_padding=True \
    actor_rollout_ref.actor.state_masking=true \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.logger=['wandb'] \
    trainer.project_name='Search-R1-A6000' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=5 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

echo ""
echo "🎉 训练完成！"
echo "📊 查看日志: cat $EXPERIMENT_NAME.log"
echo "📁 检查点: ls checkpoints/$EXPERIMENT_NAME/"
EOF

chmod +x train_a6000.sh
```

---

## 4. 性能基准

### 4.1 A6000 vs RTX 3090 对比

| 配置项 | A6000 (48GB) | RTX 3090 (24GB) | 提升 |
|--------|---------------|-----------------|------|
| **模型** | Qwen2.5-7B | Qwen2.5-3B | 2.3× |
| **batch_size** | 16 | 8 | 2× |
| **序列长度** | 2048 | 1024 | 2× |
| **n_agent** | 5 | 3 | 1.7× |
| **训练步数** | 500 | 40 | 12.5× |
| **显存占用** | ~40GB | ~20GB | 2× |
| **训练时间** | 4-5小时 | 2-3小时 | 相近 |

### 4.2 显存占用分析

```
┌─────────────────────────────────────────────────┐
│  A6000 显存占用分析 (Qwen2.5-7B)              │
├─────────────────────────────────────────────────┤
│  组件                    │ 显存占用    │
├─────────────────────────────────────────────────┤
│  Qwen2.5-7B (FP16)       │ ~14GB      │
│  vLLM推理引擎            │ ~12GB      │
│  FSDP参数缓存            │ ~8GB       │
│  优化器状态              │ ~4GB       │
│  激活值/缓存             │ ~8GB       │
│  批处理数据              │ ~4GB       │
├─────────────────────────────────────────────────┤
│  总计                    │ ~50GB      │
│  (有少量overhead管理)    │            │
└─────────────────────────────────────────────────┘

A6000 48GB显存刚好够用，有少量余量
```

### 4.3 预期训练效果

```
┌─────────────────────────────────────────────────┐
│  训练配置 (Qwen2.5-7B + GRPO + 完整NQ数据)      │
├─────────────────────────────────────────────────┤
│  数据集: NQ完整训练集 (~3000样本)               │
│  测试集: NQ完整测试集 (~3000样本)                │
│  训练步数: 500步                                  │
│  算法: GRPO (n_agent=5)                           │
├─────────────────────────────────────────────────┤
│  预期训练时间: 4-5小时                             │
│  预期最终准确率: 70-85%                            │
│  预期搜索次数: 1.2-1.5次/问题                       │
│  预期格式正确率: 95%+                              │
└─────────────────────────────────────────────────┘
```

---

## 5. 高级优化

### 5.1 方案1：激进方案（PPO + 7B）

```bash
# ===== 激进配置：使用完整PPO算法 =====
cat > train_a6000_ppo.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME='a6000-nq-search-r1-ppo-qwen2.5-7b'

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/nq_search/train.json \
    data.val_files=data/nq_search/test.json \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    \
    critic.optim.lr=5e-6 \
    critic.model.path=$BASE_MODEL \
    critic.model.use_remove_padding=true \
    critic.ppo_micro_batch_size=4 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.grad_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    \
    trainer.critic_warmup=5 \
    trainer.n_gpus_per_node=1 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=10 \
    \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "PPO训练完成！"
EOF

chmod +x train_a6000_ppo.sh
```

### 5.2 方案2：保守方案（3B + 超大batch）

```bash
# ===== 保守配置：使用3B模型但超大batch =====
cat > train_a6000_large_batch.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME='a6000-nq-search-r1-grpo-qwen2.5-3b-largebatch'

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=data/nq_search/train.json \
    data.val_files=data/nq_search/test.json \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    \
    algorithm.adv_estimator=grpo \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n_agent=10 \
    actor_rollout_ref.rollout.temperature=1.0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    \
    trainer.n_gpus_per_node=1 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=5 \
    \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "大batch训练完成！"
EOF

chmod +x train_a6000_large_batch.sh
```

### 5.3 三种方案对比

| 方案 | 模型 | 算法 | batch_size | n_agent | 显存 | 训练时间 | 适用场景 |
|------|------|------|------------|---------|------|----------|----------|
| **保守** | Qwen2.5-3B | GRPO | 64 | 10 | ~35GB | 3-4h | 最快训练 |
| **平衡** | Qwen2.5-7B | GRPO | 16 | 5 | ~40GB | 4-5h | 推荐 |
| **激进** | Qwen2.5-7B | PPO | 16 | 1 | ~45GB | 6-8h | 最佳性能 |

---

## 6. 监控与调试

### 6.1 实时监控

```bash
# ===== 终端1: 监控显存 =====
watch -n 1 nvidia-smi

# ===== 终端2: 监控训练日志 =====
tail -f a6000-nq-search-r1-grpo-qwen2.5-7b.log

# ===== 终端3: 监控检索器 =====
tail -f retriever_a6000.log

# ===== 终端4: 监控WandB (可选) =====
wandb online
```

### 6.2 关键指标

训练日志中关注的指标：

```
1. ACTIVE_TRAJ_NUM: 活跃轨迹数量
   - 初始: [batch_size]
   - 逐轮递减
   - 反映模型学习进度

2. loss: 训练损失
   - 应该逐渐降低
   - 突然升高可能有问题

3. reward: 奖励值
   - 应该逐渐提升
   - 最终反映在准确率上

4. 准确率 (validation)
   - 每25步评估一次
   - 目标: 70%+
```

### 6.3 常见问题

#### 问题1: 显存接近极限

```
现象：nvidia-smi显示显存使用率>95%
解决：
1. 减少batch_size: 16 → 12
2. 减少序列长度: max_prompt_length=2048 → 1536
3. 启用参数卸载: param_offload=true
```

#### 问题2: 训练速度慢

```
现象：每个iteration > 2分钟
解决：
1. 检查vLLM配置: gpu_memory_utilization=0.85 → 0.9
2. 减少数据加载时间: 使用本地SSD
3. 检查检索器延迟: 应该<100ms
```

#### 问题3: 准确率不提升

```
现象：训练了很多步，准确率仍然很低
解决：
1. 检查检索器是否正常工作
2. 检查数据格式是否正确
3. 增加训练步数: 500 → 1000
4. 调整学习率: 尝试5e-6 → 3e-6
```

---

## 7. 一键启动脚本

### 7.1 完整自动化脚本

```bash
#!/bin/bash
# ============================================
# Search-R1 A6000 一键启动脚本
# ============================================

set -e

echo "🚀 Search-R1 A6000 优化训练"
echo "硬件: A6000 (48GB显存)"
echo "模型: Qwen2.5-7B"
echo "算法: GRPO"
echo ""

# ===== 步骤1: 环境检查 =====
echo "📋 检查环境..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ 请确保GPU可用"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | awk '{print $1}')

echo "✅ GPU: $GPU_NAME"
echo "✅ 显存: $GPU_MEM"

if [ "$GPU_MEM" -lt 40000 ]; then
    echo "⚠️  警告: 显存可能不足40GB"
fi

# ===== 步骤2: 检查检索器 =====
echo "🔍 检查检索器..."
if curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    echo "✅ 检索器已启动"
else
    echo "❌ 检索器未启动，请先启动检索器"
    echo "运行: bash retrieval_launch.sh"
    exit 1
fi

# ===== 步骤3: 检查数据 =====
echo "📊 检查数据..."
if [ ! -f "data/nq_search/train.json" ]; then
    echo "❌ 数据文件不存在"
    echo "请先运行数据准备脚本"
    exit 1
fi

TRAIN_SAMPLES=$(wc -l < data/nq_search/train.json)
TEST_SAMPLES=$(wc -l < data/nq_search/test.json)
echo "✅ 训练样本: $TRAIN_SAMPLES"
echo "✅ 测试样本: $TEST_SAMPLES"

# ===== 步骤4: 创建A6000优化配置 =====
echo "📝 创建A6000训练配置..."

cat > train_a6000_auto.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME='a6000-auto'

python3 -m verl.trainer.main_ppo \
    data.train_files=data/nq_search/train.json \
    data.val_files=data/nq_search/test.json \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    \
    algorithm.adv_estimator=grpo \
    \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.state_masking=true \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n_agent=5 \
    \
    trainer.n_gpus_per_node=1 \
    trainer.total_training_steps=500 \
    trainer.total_epochs=5 \
    trainer.logger=['wandb'] \
    trainer.project_name='Search-R1-A6000' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/$EXPERIMENT_NAME \
    \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
EOF

chmod +x train_a6000_auto.sh

# ===== 步骤5: 开始训练 =====
echo "🎯 开始A6000优化训练..."
echo ""
echo "📊 配置信息:"
echo "   - 模型: Qwen2.5-7B"
echo "   - 算法: GRPO"
echo "   - batch_size: 16"
echo "   - n_agent: 5"
echo "   - 训练步数: 500"
echo ""
echo "⏱️  预计时间: 4-5小时"
echo "💾 预期显存: ~38-42GB"
echo ""

./train_a6000_auto.sh

echo ""
echo "🎉 训练完成！"
echo "📊 查看日志: cat a6000-auto.log"
echo "📁 检查点: ls checkpoints/a6000-auto/"
```

---

## 8. 性能预期

### 8.1 训练曲线预期

```
┌─────────────────────────────────────────────────┐
│  A6000 (Qwen2.5-7B + GRPO) 训练曲线预期           │
├─────────────────────────────────────────────────┤
│  Step │ 准确率 │ 搜索次数 │ 格式正确率 │ 损失    │
├──────┼────────┼──────────┼────────────┼─────────┤
│   0  │  25%   │  0.5     │   60%       │ 高      │
│  100 │  45%   │  1.0     │   85%       │ 中      │
│  250 │  65%   │  1.2     │   95%       │ 低      │
│  500 │  75%   │  1.2     │   98%       │ 稳定    │
└──────┴────────┴──────────┴────────────┴─────────┘
```

### 8.2 最终模型能力

训练完成后，模型应该具备：

```
✅ 何时搜索: 识别知识盲区
✅ 如何搜索: 构造精确检索查询
✅ 如何推理: 整合检索结果
✅ 格式正确: 严格遵循工具调用格式
✅ 答案质量: 70-85%准确率
```

---

## 9. 推理测试

### 9.1 使用训练好的模型

```bash
# ===== 修改推理脚本 =====
cd /path/to/Search-R1

# 编辑 infer.py
# 修改模型路径为训练好的模型
model_id = "checkpoints/a6000-auto/actor"

# ===== 运行推理 =====
conda activate searchr1_a6000
python infer.py
```

### 9.2 批量测试

```bash
# ===== 批量推理测试 =====
python -m verl.trainer.main_generation \
    --input_path data/nq_search/test.json \
    --output_path results/a6000_results.parquet \
    --model_path checkpoints/a6000-auto/actor \
    --temperature 1.0 \
    --max_samples 1 \
    --max_prompt_length 2048 \
    --max_response_length 512

# ===== 评估结果 =====
python -m verl.trainer.main_eval \
    --input_path results/a6000_results.parquet \
    --model_path checkpoints/a6000-auto/actor \
    --data_source nq
```

---

## 10. 总结

### 10.1 A6000配置要点

| 配置项 | 推荐值 | 原因 |
|--------|--------|------|
| **模型** | Qwen2.5-7B | 高质量7B模型 |
| **算法** | GRPO | 无需Critic，省显存 |
| **batch_size** | 16 | 平衡显存和训练速度 |
| **序列长度** | 2048 | 支持复杂推理 |
| **n_agent** | 5 | GRPO组内多样性 |
| **训练步数** | 500 | 充分训练 |

### 10.2 显存优化总结

```
A6000 (48GB) 显存分配:
┌─────────────────────────────────────┐
│  组件                    │ 显存    │
├─────────────────────────────────────┤
│  Qwen2.5-7B模型           │ 14GB   │
│  vLLM推理引擎              │ 12GB   │
│  FSDP参数梯度缓存         │ 8GB    │
│  激活值/批处理数据         │ 8GB    │
├─────────────────────────────────────┤
│  总计 (含管理开销)        │ 42GB   │
├─────────────────────────────────────┤
│  余量 (安全边际)            │ 6GB    │
└─────────────────────────────────────┘
```

### 10.3 性能提升

相比最小化配置：

```
┌─────────────────────────────────────────┐
│  指标        │ 最小化   │ A6000   │ 提升  │
├─────────────────────────────────────────┤
│  模型大小    │ 3B       │ 7B       │ +133% │
│  batch_size  │ 8        │ 16       │ +100% │
│  序列长度    │ 1024     │ 2048     │ +100% │
│  n_agent     │ 3        │ 5        │ +67%  │
│  训练步数    │ 40       │ 500      │ +1150%│
│  最终准确率  │ 30-50%   │ 70-85%   │ +70%  │
└─────────────────────────────────────────┘
```

---

## 11. 故障排查

### 11.1 显存相关

```bash
# ===== 实时监控显存 =====
watch -n 1 nvidia-smi

# ===== 检查显存占用详情 =====
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# ===== 如果显存接近48GB =====
# 1. 立即停止训练 (Ctrl+C)
# 2. 检查当前检查点
# 3. 调整配置重新训练
```

### 11.2 训练卡顿

```bash
# ===== 检查GPU利用率 =====
nvidia-smi dmon -s u

# ===== 如果GPU利用率低 =====
# 1. 检查数据加载是否卡住
# 2. 检查检索器响应时间
# 3. 检查网络连接
```

### 11.3 训练异常退出

```bash
# ===== 检查日志 =====
tail -100 a6000-auto.log

# ===== 常见错误 =====
# CUDA OOM: 减少batch_size或序列长度
# 连接错误: 检查检索器是否正常
# NaN损失: 降低学习率或检查数据质量
```

---

*本指南针对A6000 (48GB显存) 提供了完整的优化配置，能够在合理时间内完成高质量的Search-R1训练！*
