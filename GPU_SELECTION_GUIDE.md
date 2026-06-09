# GPU选择和配置指南

本指南说明如何在Search-R1训练中选择和配置GPU设备。

## 🎮 快速开始

### 方法1: 命令行参数（推荐）

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

### 方法2: 环境变量

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 运行训练脚本
bash scripts/train_gpu.sh
```

### 方法3: 修改配置文件

编辑 `configs/gpu_config.yaml`:
```yaml
# 使用GPU 1
gpu_id: 1

# 使用多个GPU
gpu_id: 0,1,2,3
```

## 🔍 GPU信息查看

### 查看可用GPU

```bash
# 查看所有GPU
nvidia-smi

# 查看GPU数量
nvidia-smi --query-gpu=count --format=csv,noheader

# 查看GPU详细信息
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
```

### 输出示例

```
index, name, memory.total [MiB], memory.free [MiB]
0, NVIDIA RTX A6000, 49140, 49140
1, NVIDIA RTX A6000, 49140, 49140
2, NVIDIA RTX A6000, 49140, 49140
3, NVIDIA RTX A6000, 49140, 49140
4, NVIDIA RTX A6000, 49140, 49140
```

## 📊 GPU配置策略

### 单GPU训练

**适用场景:**
- ✅ 显存充足 (≥48GB)
- ✅ 小到中等模型 (≤7B)
- ✅ GRPO算法

**配置示例:**
```bash
bash scripts/train_gpu.sh --gpu 0
```

**预期性能:**
- 训练时间: 4-5小时
- 显存占用: ~40GB

### 多GPU训练

**适用场景:**
- ✅ 需要更大batch size
- ✅ 需要更多agents (GRPO)
- ✅ 更大模型 (≥13B)

**配置示例:**
```bash
# 双GPU
bash scripts/train_gpu.sh --gpu 0,1

# 四GPU
bash scripts/train_gpu.sh --gpu 0,1,2,3
```

**预期性能提升:**
- 双GPU: ~1.8x 训练速度
- 四GPU: ~3.2x 训练速度

## 🛠️ 高级配置

### 自定义实验名称

```bash
bash scripts/train_gpu.sh \
    --gpu 1 \
    --name "gpu1-grpo-qwen2.5-7b"
```

### 使用不同配置文件

```bash
# GRPO配置
bash scripts/train_gpu.sh --gpu 0 --config configs/a6000_grpo.yaml

# PPO配置
bash scripts/train_gpu.sh --gpu 0 --config configs/a6000_ppo.yaml
```

### 使用不同模型

```bash
# 3B模型
bash scripts/train_gpu.sh --gpu 0 --model Qwen/Qwen2.5-3B

# Llama模型
bash scripts/train_gpu.sh --gpu 0 --model meta-llama/Llama-3.2-3B
```

## 🎯 GPU选择最佳实践

### 1. 检查GPU可用性

```bash
# 运行GPU验证脚本
python scripts/verify_gpu.py
```

### 2. 查看GPU使用情况

```bash
# 实时监控GPU状态
watch -n 1 nvidia-smi
```

### 3. 选择空闲GPU

```bash
# 查看GPU内存使用
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# 选择内存充足的GPU
# 例如: GPU 2 有充足内存
bash scripts/train_gpu.sh --gpu 2
```

### 4. 避免GPU冲突

```bash
# 检查GPU是否被占用
nvidia-smi

# 如果GPU被占用，选择其他GPU
bash scripts/train_gpu.sh --gpu 3
```

## 📋 常用命令参考

### 基础GPU选择

| 需求 | 命令 |
|------|------|
| 使用默认GPU | `bash scripts/train_gpu.sh` |
| 使用GPU 1 | `bash scripts/train_gpu.sh --gpu 1` |
| 使用多个GPU | `bash scripts/train_gpu.sh --gpu 0,1` |
| 使用特定GPU | `bash scripts/train_gpu.sh --gpu 2,3` |

### 高级配置

| 需求 | 命令 |
|------|------|
| PPO算法 | `bash scripts/train_gpu.sh --config configs/a6000_ppo.yaml` |
| 3B模型 | `bash scripts/train_gpu.sh --model Qwen/Qwen2.5-3B` |
| 自定义名称 | `bash scripts/train_gpu.sh --name my-experiment` |
| 完整配置 | `bash scripts/train_gpu.sh --gpu 1 --config configs/a6000_ppo.yaml --name gpu1-ppo` |

## 🚨 故障排查

### 问题1: GPU不存在

**错误信息:**
```
❌ GPU 5 不存在 (系统只有 5 个GPU)
```

**解决方案:**
```bash
# 查看可用GPU
nvidia-smi

# 使用有效的GPU ID (0-4)
bash scripts/train_gpu.sh --gpu 0
```

### 问题2: GPU内存不足

**错误信息:**
```
CUDA out of memory
```

**解决方案:**
```bash
# 1. 使用更大显存的GPU
bash scripts/train_gpu.sh --gpu 2

# 2. 减少batch size
# 编辑配置文件，减少 train_batch_size

# 3. 使用更小的模型
bash scripts/train_gpu.sh --gpu 0 --model Qwen/Qwen2.5-3B
```

### 问题3: GPU被占用

**错误信息:**
```
GPU 0 is being used by another process
```

**解决方案:**
```bash
# 1. 查看GPU使用情况
nvidia-smi

# 2. 使用其他GPU
bash scripts/train_gpu.sh --gpu 1

# 3. 等待GPU释放或结束占用进程
```

## 📚 相关文档

- [A6000_GUIDE.md](A6000_GUIDE.md) - A6000优化配置指南
- [scripts/README.md](scripts/README.md) - 脚本使用说明
- [configs/README.md](configs/README.md) - 配置文件说明

## 💡 提示和技巧

### 自动选择空闲GPU

```bash
# 查找内存最多的GPU
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    sort -t',' -k2 -rn | head -1 | cut -d',' -f1)

echo "最空闲的GPU: $FREE_GPU"
bash scripts/train_gpu.sh --gpu $FREE_GPU
```

### GPU性能监控

```bash
# 终端1: 启动训练
bash scripts/train_gpu.sh --gpu 0

# 终端2: 监控GPU
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv'
```

### 批量训练不同GPU

```bash
# 在不同GPU上启动多个训练
for gpu in 0 1 2 3; do
    bash scripts/train_gpu.sh --gpu $gpu --name "gpu${gpu}-experiment" &
done
```

---

通过本指南，您应该能够灵活选择和配置GPU设备进行Search-R1训练。
