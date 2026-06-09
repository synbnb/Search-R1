# Search-R1 完整复现指南

本指南提供Search-R1项目的完整复现说明，从环境准备到模型训练的完整流程。

## 📚 目录

- [1. 项目概述](#1-项目概述)
- [2. 硬件要求](#2-硬件要求)
- [3. 环境准备](#3-环境准备)
- [4. GPU选择配置](#4-gpu选择配置)
- [5. 数据准备](#5-数据准备)
- [6. 检索器准备](#6-检索器准备)
- [7. 训练配置](#7-训练配置)
- [8. 训练执行](#8-训练执行)
- [9. 监控与调试](#9-监控与调试)
- [10. 故障排查](#10-故障排查)

---

## 1. 项目概述

### 1.1 Search-R1简介

Search-R1是一个基于强化学习的搜索增强推理系统，通过训练模型学习何时搜索以及如何整合搜索结果来回答复杂问题。

**核心特性：**
- 🤖 多轮搜索-推理循环
- 🎯 基于GRPO/PPO的强化学习训练
- 🔍 可配置的检索系统
- 📊 支持多种模型规模

### 1.2 训练流程

```
┌─────────────────────────────────────────────────┐
│           Search-R1 训练流程                      │
├─────────────────────────────────────────────────┤
│  1. 环境验证 → 验证GPU和依赖                     │
│  2. 数据准备 → 下载和转换NQ数据集                 │
│  3. 索引下载 → 获取检索索引和语料                  │
│  4. 检索器启动 → 启动GPU加速检索服务               │
│  5. 模型训练 → 执行GRPO/PPO训练                   │
│  6. 模型评估 → 测试和结果分析                     │
└─────────────────────────────────────────────────┘
```

---

## 2. 硬件要求

### 2.1 GPU配置对比

| 配置方案 | 显存要求 | 推荐GPU | 训练时间 | 适用场景 |
|----------|----------|---------|----------|----------|
| **最小化** | 24GB | RTX 3090 | 2-3h | 快速验证 |
| **标准** | 48GB | A6000 | 4-5h | 推荐配置 |
| **高性能** | 48GB×N | 多个A6000 | 更快 | 大规模训练 |

### 2.2 软件要求

- Python 3.10+
- CUDA 12.1+
- PyTorch 2.4.0+
- vLLM 0.6.3+

---

## 3. 环境准备

### 3.1 快速环境验证

```bash
# 验证GPU环境
python scripts/verify_gpu.py
```

**预期输出：**
```
============================================================
Search-R1 GPU环境验证
============================================================

📦 检查PyTorch...
   PyTorch版本: 2.4.0
   CUDA可用: True

🎮 检查GPU...
   GPU数量: 1
   GPU 0: NVIDIA RTX A6000
   显存: 47.3 GB

✅ 环境验证完成！
```

### 3.2 完整环境安装

```bash
# 创建conda环境
conda create -n searchr1 python=3.10
conda activate searchr1

# 安装PyTorch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装vLLM
pip install vllm==0.6.3

# 安装基础依赖
pip install transformers datasets
pip install ray==2.9.0 hydra-core wandb accelerate
pip install flash-attn --no-build-isolation
pip install huggingface_hub

# 安装veRL框架
cd /path/to/Search-R1
pip install -e .
```

---

## 4. GPU选择配置

### 4.1 查看可用GPU

```bash
# 查看所有GPU信息
nvidia-smi

# 查看GPU数量
nvidia-smi --query-gpu=count --format=csv,noheader

# 查看详细GPU信息
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

### 4.2 GPU选择方法

#### 方法1: 命令行参数（推荐）

```bash
# 使用默认GPU (GPU 0)
bash scripts/train_gpu.sh

# 指定单个GPU
bash scripts/train_gpu.sh --gpu 1

# 使用多个GPU
bash scripts/train_gpu.sh --gpu 0,1,2

# 完整自定义
bash scripts/train_gpu.sh --gpu 2 --config configs/a6000_ppo.yaml --name my-experiment
```

#### 方法2: 环境变量

```bash
export CUDA_VISIBLE_DEVICES=1
bash scripts/train_gpu.sh
```

#### 方法3: 修改配置

编辑 `configs/gpu_config.yaml`：
```yaml
gpu_id: 1  # 使用GPU 1
```

### 4.3 GPU选择最佳实践

**选择空闲GPU：**
```bash
# 查看GPU使用情况
nvidia-smi

# 选择内存充足的GPU
bash scripts/train_gpu.sh --gpu 2
```

**自动选择最空闲GPU：**
```bash
# 查找内存最多的GPU
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    sort -t',' -k2 -rn | head -1 | cut -d',' -f1)

bash scripts/train_gpu.sh --gpu $FREE_GPU
```

---

## 5. 数据准备

### 5.1 NQ数据集准备

```bash
# 创建数据目录
mkdir -p data/nq_search

# 使用专业脚本准备数据
python scripts/data_prepare.py --output_dir data/nq_search
```

**输出结构：**
```
data/nq_search/
├── train.json    # ~3000个训练样本
└── test.json     # ~3000个测试样本
```

**数据格式：**
```json
{
  "data_source": "nq",
  "prompt": [{"role": "user", "content": "..."}],
  "ability": "fact-reasoning",
  "reward_model": {
    "style": "rule",
    "ground_truth": {"target": ["答案1", "答案2"]}
  },
  "extra_info": {"split": "train", "index": 0}
}
```

### 5.2 数据验证

```bash
# 查看数据统计
wc -l data/nq_search/train.json  # 训练样本数
wc -l data/nq_search/test.json   # 测试样本数

# 查看数据样例
head -1 data/nq_search/train.json | python -m json.tool
```

---

## 6. 检索器准备

### 6.1 下载检索索引

```bash
# 下载E5检索索引和Wikipedia语料
python scripts/download_index.py --output_dir data/index
```

**输出结构：**
```
data/index/
├── e5_Flat.index     # E5检索索引 (~1GB)
└── wiki-18.jsonl     # Wikipedia语料 (~500MB)
```

### 6.2 启动检索服务器

```bash
# 启动GPU加速的检索服务器
bash scripts/start_retriever.sh
```

**服务信息：**
- URL: `http://127.0.0.1:8000/retrieve`
- 日志: `logs/retriever.log`
- 支持GPU加速（FAISS GPU）

**验证检索器：**
```bash
curl http://127.0.0.1:8000/retrieve
```

---

## 7. 训练配置

### 7.1 配置方案选择

#### 方案A: 最小化配置（24GB显存）

**适用：** RTX 3090，快速验证

```bash
python scripts/train_gpu.sh \
    --gpu 0 \
    --model Qwen/Qwen2.5-3B \
    --config configs/minimal_config.yaml
```

**参数：**
- 模型: Qwen2.5-3B
- batch_size: 8
- n_agent: 3
- 训练步数: 40
- 显存占用: ~20GB

#### 方案B: 标准配置（48GB显存）⭐推荐

**适用：** A6000，平衡性能

```bash
bash scripts/train_gpu.sh \
    --gpu 0 \
    --config configs/a6000_grpo.yaml
```

**参数：**
- 模型: Qwen2.5-7B
- 算法: GRPO
- batch_size: 16
- n_agent: 5
- 训练步数: 500
- 显存占用: ~40GB

#### 方案C: 高性能配置（48GB×N）

**适用：** 多个A6000，最佳性能

```bash
bash scripts/train_gpu.sh \
    --gpu 0,1 \
    --config configs/a6000_ppo.yaml
```

**参数：**
- 模型: Qwen2.5-7B
- 算法: PPO
- batch_size: 32
- 训练步数: 500
- 显存占用: ~45GB×2

### 7.2 配置文件说明

**GRPO配置 (`configs/a6000_grpo.yaml`)：**
- 无需Critic网络，省显存
- 适合单GPU训练
- 训练时间4-5小时

**PPO配置 (`configs/a6000_ppo.yaml`)：**
- 完整PPO算法
- 更精确的价值估计
- 需要更多显存和训练时间

### 7.3 自定义配置

**修改训练步数：**
```bash
bash scripts/train_gpu.sh \
    --gpu 0 \
    --config configs/a6000_grpo.yaml \
    trainer.total_training_steps=1000
```

**修改batch大小：**
```bash
bash scripts/train_gpu.sh \
    --gpu 0 \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n_agent=10
```

---

## 8. 训练执行

### 8.1 完整训练流程

```bash
# 1. 环境验证
python scripts/verify_gpu.py

# 2. 数据准备
python scripts/data_prepare.py --output_dir data/nq_search

# 3. 索引下载
python scripts/download_index.py --output_dir data/index

# 4. 启动检索器
bash scripts/start_retriever.sh

# 5. 开始训练
bash scripts/train_gpu.sh --gpu 0 --config configs/a6000_grpo.yaml
```

### 8.2 一键启动（完整自动化）

```bash
#!/bin/bash
set -e

echo "🚀 Search-R1 一键启动"

# 步骤1: 环境验证
python scripts/verify_gpu.py || exit 1

# 步骤2: 数据准备
if [ ! -f "data/nq_search/train.json" ]; then
    python scripts/data_prepare.py --output_dir data/nq_search
fi

# 步骤3: 索引下载
if [ ! -f "data/index/e5_Flat.index" ]; then
    python scripts/download_index.py --output_dir data/index
fi

# 步骤4: 启动检索器
if ! curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    bash scripts/start_retriever.sh
    sleep 10
fi

# 步骤5: 开始训练
bash scripts/train_gpu.sh --gpu 0 --config configs/a6000_grpo.yaml

echo "🎉 训练完成！"
```

### 8.3 训练输出

**检查点保存：**
```
checkpoints/{experiment_name}/
├── actor/                    # 训练好的Actor模型
├── checkpoint_50/           # 第50步检查点
├── checkpoint_100/          # 第100步检查点
└── ...
```

**日志文件：**
```
logs/{experiment_name}.log    # 训练日志
logs/retriever.log            # 检索器日志
```

---

## 9. 监控与调试

### 9.1 实时监控

**终端1 - 监控GPU：**
```bash
watch -n 1 nvidia-smi
```

**终端2 - 监控训练日志：**
```bash
tail -f logs/{experiment_name}.log
```

**终端3 - 监控检索器：**
```bash
tail -f logs/retriever.log
```

**终端4 - WandB监控（可选）：**
```bash
wandb online
```

### 9.2 关键指标

**训练指标：**
```
1. ACTIVE_TRAJ_NUM: 活跃轨迹数量
   - 初始: [batch_size]
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

### 9.3 预期训练曲线

```
┌─────────────────────────────────────────────────┐
│  Search-R1 训练曲线预期                           │
├──────┼────────┼──────────┼────────────┼─────────┤
│ Step │ 准确率 │ 搜索次数 │ 格式正确率 │ 损失    │
├──────┼────────┼──────────┼────────────┼─────────┤
│  0   │  25%   │  0.5     │   60%      │ 高      │
│ 100  │  45%   │  1.0     │   85%      │ 中      │
│ 250  │  65%   │  1.2     │   95%      │ 低      │
│ 500  │  75%   │  1.2     │   98%      │ 稳定    │
└──────┴────────┴──────────┴────────────┴─────────┘
```

---

## 10. 故障排查

### 10.1 GPU相关问题

#### 问题1: GPU不存在

**错误：**
```
❌ GPU 5 不存在 (系统只有 5 个GPU)
```

**解决：**
```bash
# 查看可用GPU
nvidia-smi

# 使用有效的GPU ID (0-4)
bash scripts/train_gpu.sh --gpu 0
```

#### 问题2: 显存不足

**错误：**
```
CUDA out of memory
```

**解决：**
```bash
# 方案1: 使用更大显存的GPU
bash scripts/train_gpu.sh --gpu 2

# 方案2: 减少batch size
# 编辑配置文件，减少 train_batch_size

# 方案3: 使用更小的模型
bash scripts/train_gpu.sh --gpu 0 --model Qwen/Qwen2.5-3B
```

#### 问题3: GPU被占用

**解决：**
```bash
# 查看GPU使用情况
nvidia-smi

# 使用其他GPU
bash scripts/train_gpu.sh --gpu 1
```

### 10.2 训练问题

#### 问题1: 检索器连接失败

**错误：**
```
连接检索器失败
```

**解决：**
```bash
# 检查检索器状态
curl http://127.0.0.1:8000/retrieve

# 重新启动检索器
bash scripts/start_retriever.sh
```

#### 问题2: 数据文件缺失

**错误：**
```
❌ 数据文件不存在: data/nq_search/train.json
```

**解决：**
```bash
# 准备数据
python scripts/data_prepare.py --output_dir data/nq_search
```

#### 问题3: 准确率不提升

**解决：**
```bash
# 1. 检查检索器是否正常工作
# 2. 检查数据格式是否正确
# 3. 增加训练步数: 500 → 1000
# 4. 调整学习率
```

### 10.3 性能问题

#### 问题1: 训练速度慢

**解决：**
```bash
# 1. 检查vLLM配置: gpu_memory_utilization
# 2. 检查检索器延迟: 应该<100ms
# 3. 使用本地SSD存储数据
```

#### 问题2: 显存接近极限

**解决：**
```bash
# 1. 减少batch_size: 16 → 12
# 2. 减少序列长度: 2048 → 1536
# 3. 启用参数卸载: param_offload=true
```

---

## 11. 技术参考文档

**更多技术细节请参考：**

- [ARCHITECTURE.md](ARCHITECTURE.md) - 项目架构详解
- [CODE_FLOW_ANALYSIS.md](CODE_FLOW_ANALYSIS.md) - 代码流程分析
- [VERL_INTEGRATION.md](VERL_INTEGRATION.md) - 框架集成文档
- [scripts/README.md](scripts/README.md) - 脚本使用说明
- [configs/README.md](configs/README.md) - 配置文件说明

---

## 12. 总结

### 快速回顾

**最小化配置（24GB）：**
```bash
python scripts/verify_gpu.py
python scripts/data_prepare.py --output_dir data/nq_search
python scripts/download_index.py --output_dir data/index
bash scripts/start_retriever.sh
bash scripts/train_gpu.sh --gpu 0 --model Qwen/Qwen2.5-3B
```

**标准配置（48GB）：**
```bash
python scripts/verify_gpu.py
python scripts/data_prepare.py --output_dir data/nq_search
python scripts/download_index.py --output_dir data/index
bash scripts/start_retriever.sh
bash scripts/train_gpu.sh --gpu 0 --config configs/a6000_grpo.yaml
```

### 关键要点

1. **GPU选择** - 使用 `train_gpu.sh --gpu N` 灵活选择GPU
2. **配置文件** - 使用YAML配置管理训练参数
3. **数据准备** - 使用专业脚本自动处理数据
4. **监控训练** - 实时监控GPU和训练日志
5. **故障排查** - 参考第10节的详细解决方案

---

**祝您训练顺利！** 如有问题，请参考故障排查章节或技术参考文档。
