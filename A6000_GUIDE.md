# Search-R1 A6000 (48GB显存) 优化配置指南

> **专业重构版** - 使用模块化脚本和配置文件，提供更好的可维护性和专业性

## 🚀 快速开始（5分钟）

```bash
# 1. 验证环境
python scripts/verify_gpu.py

# 2. 准备数据
python scripts/data_prepare.py --output_dir data/nq_search

# 3. 下载索引
python scripts/download_index.py --output_dir data/index

# 4. 启动检索器
bash scripts/start_retriever.sh

# 5. 开始训练
bash scripts/train_a6000.sh
```

## 📚 目录
- [1. A6000硬件优势](#1-a6000硬件优势)
- [2. 推荐配置方案](#2-推荐配置方案)
- [3. 快速启动脚本](#3-快速启动脚本)
- [4. 性能基准](#4-性能基准)
- [5. 高级优化](#5-高级优化)
- [6. 监控与调试](#6-监控与调试)
- [7. 一键启动脚本](#7-一键启动脚本)
- [8. 性能预期](#8-性能预期)
- [9. 推理测试](#9-推理测试)
- [10. 总结](#10-总结)
- [11. 故障排查](#11-故障排查)

## 📝 新增章节说明

本次重构新增了以下专业化的内容结构：

- **3.1节**: 环境准备使用GPU验证脚本
- **3.2节**: 数据准备使用专业Python脚本
- **3.3节**: 索引下载使用专业下载脚本
- **3.4节**: 检索器启动使用专业Shell脚本
- **3.5节**: 训练使用YAML配置文件
- **10.1节**: 新增重构改进说明
- **10.2节**: 新增文件结构说明

---

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
pip install huggingface_hub

# ===== 安装veRL框架 =====
cd /path/to/Search-R1
pip install -e .

# ===== 验证安装 =====
python scripts/verify_gpu.py
```

### 3.2 数据准备（完整NQ数据集）

```bash
# ===== 创建数据目录 =====
mkdir -p data/nq_search

# ===== 使用专业数据准备脚本 =====
python scripts/data_prepare.py --output_dir data/nq_search
```

**数据准备脚本说明：**
- 📥 从HuggingFace Hub自动下载NQ数据集
- 🔄 转换为Search-R1训练格式
- 💾 保存为JSON Lines格式
- 📊 显示数据统计信息

**输出结构：**
```
data/nq_search/
├── train.json    # ~3000个训练样本
└── test.json     # ~3000个测试样本
```

### 3.3 检索器准备（真实E5检索）

```bash
# ===== 创建检索器环境 =====
conda create -n retriever_a6000 python=3.10
conda activate retriever_a6000

# ===== 安装依赖 =====
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pyserini

# ===== 使用专业索引下载脚本 =====
python scripts/download_index.py --output_dir data/index
```

**索引下载脚本说明：**
- 📥 从HuggingFace Hub下载预构建E5索引
- 📦 自动下载Wikipedia-18语料库
- 🔧 自动解压和文件整理
- ✅ 完整性检查

**输出结构：**
```
data/index/
├── e5_Flat.index     # E5检索索引 (~1GB)
└── wiki-18.jsonl     # Wikipedia语料 (~500MB)
```

### 3.4 启动E5检索服务器

```bash
# ===== 启动GPU加速的E5检索器 =====
conda activate retriever_a6000
cd /path/to/Search-R1

# ===== 使用专业检索服务器脚本 =====
bash scripts/start_retriever.sh
```

**检索服务器脚本说明：**
- 🔍 自动检查索引文件和端口占用
- 🚀 启动GPU加速的E5检索服务
- 📊 提供详细状态信息
- 📝 自动记录日志

**服务信息：**
- URL: `http://127.0.0.1:8000/retrieve`
- 日志: `logs/retriever.log`
- 支持GPU加速（FAISS GPU）

### 3.5 A6000优化训练脚本

```bash
# ===== 使用专业训练脚本 =====
cd /path/to/Search-R1
bash scripts/train_a6000.sh
```

**训练脚本说明：**
- 📝 使用YAML配置文件 (`configs/a6000_grpo.yaml`)
- 🔍 自动检查前置条件（GPU、数据、检索器）
- 📊 提供详细的配置信息和性能预期
- 📁 自动管理日志和检查点目录

**配置文件结构：**
```yaml
# configs/a6000_grpo.yaml
data:
  train_batch_size: 16
  max_prompt_length: 2048

algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  model:
    path: Qwen/Qwen2.5-7B
  rollout:
    n_agent: 5
    gpu_memory_utilization: 0.85

trainer:
  total_training_steps: 500
  experiment_name: a6000-nq-search-r1-grpo-qwen2.5-7b
```

**输出结构：**
```
logs/
└── a6000-nq-search-r1-grpo-qwen2.5-7b.log

checkpoints/
└── a6000-nq-search-r1-grpo-qwen2.5-7b/
    ├── actor/
    ├── checkpoint_50/
    ├── checkpoint_100/
    └── ...
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
python3 -m verl.trainer.main_ppo --config configs/a6000_ppo.yaml
```

**配置说明 (`configs/a6000_ppo.yaml`)：**
- 🤖 使用完整PPO算法（带Critic网络）
- 🎯 更精确的价值估计
- 💾 更高的显存占用 (~45GB)
- ⏱️ 更长的训练时间 (6-8小时)

### 5.2 方案2：保守方案（3B + 超大batch）

```bash
# ===== 保守配置：使用3B模型但超大batch =====
# 创建自定义配置文件或修改参数
python3 -m verl.trainer.main_ppo \
    --config configs/a6000_grpo.yaml \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    actor_rollout_ref.rollout.n_agent=10
```

**配置说明：**
- 🚀 更大的batch size (64)
- ⚡ 更多的GRPO agents (10)
- 💾 最低显存占用 (~35GB)
- ⏱️ 最快训练时间 (3-4小时)

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

echo "🚀 Search-R1 A6000 一键启动"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ===== 步骤1: 环境验证 =====
echo "📋 步骤1: 验证GPU环境..."
python scripts/verify_gpu.py || exit 1

# ===== 步骤2: 数据准备 =====
echo ""
echo "📊 步骤2: 准备数据集..."
if [ ! -f "data/nq_search/train.json" ]; then
    python scripts/data_prepare.py --output_dir data/nq_search
else
    echo "✅ 数据已存在，跳过准备"
fi

# ===== 步骤3: 索引下载 =====
echo ""
echo "🔍 步骤3: 准备检索索引..."
if [ ! -f "data/index/e5_Flat.index" ]; then
    python scripts/download_index.py --output_dir data/index
else
    echo "✅ 索引已存在，跳过下载"
fi

# ===== 步骤4: 启动检索器 =====
echo ""
echo "🚀 步骤4: 启动检索服务器..."
if ! curl -s http://127.0.0.1:8000/retrieve &> /dev/null; then
    bash scripts/start_retriever.sh
    echo "⏳ 等待检索器启动..."
    sleep 10
else
    echo "✅ 检索器已运行"
fi

# ===== 步骤5: 开始训练 =====
echo ""
echo "🎯 步骤5: 开始A6000优化训练..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash scripts/train_a6000.sh

echo ""
echo "🎉 训练完成！"
echo "📊 查看日志: ls logs/"
echo "📁 检查点: ls checkpoints/"
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

### 10.1 重构改进

**改进前：**
- ❌ 长内联Python命令（113-167行）
- ❌ 手动索引下载脚本（185-222行）
- ❌ 超长训练命令行（266-315行）
- ❌ 代码重复和维护困难

**改进后：**
- ✅ 专业Python脚本 (`scripts/`)
- ✅ YAML配置文件 (`configs/`)
- ✅ 模块化设计
- ✅ 易于维护和扩展

### 10.2 新增文件结构

```
Search-R1/
├── scripts/                          # 训练脚本目录
│   ├── README.md                     # 脚本使用说明
│   ├── data_prepare.py              # 数据准备脚本
│   ├── download_index.py            # 索引下载脚本
│   ├── verify_gpu.py                # GPU验证脚本
│   ├── train_a6000.sh               # A6000训练脚本
│   └── start_retriever.sh          # 检索器启动脚本
│
└── configs/                          # 配置文件目录
    ├── README.md                     # 配置文件说明
    ├── a6000_grpo.yaml              # 平衡方案配置
    └── a6000_ppo.yaml               # 激进方案配置
```

### 10.3 A6000配置要点

| 配置项 | 推荐值 | 原因 |
|--------|--------|------|
| **模型** | Qwen2.5-7B | 高质量7B模型 |
| **算法** | GRPO | 无需Critic，省显存 |
| **batch_size** | 16 | 平衡显存和训练速度 |
| **序列长度** | 2048 | 支持复杂推理 |
| **n_agent** | 5 | GRPO组内多样性 |
| **训练步数** | 500 | 充分训练 |
| **配置方式** | YAML文件 | 专业、易维护 |

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
