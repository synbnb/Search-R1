# Search-R1 配置文件

本目录包含Search-R1训练的YAML配置文件，用于统一管理训练参数。

## 📁 配置文件列表

### `a6000_grpo.yaml` - A6000平衡方案（推荐）

**适用硬件：** NVIDIA A6000 (48GB显存)

**训练配置：**
- 模型: Qwen/Qwen2.5-7B
- 算法: GRPO (无需Critic网络)
- batch_size: 16
- n_agent: 5
- 训练步数: 500
- 显存占用: ~38-42GB
- 训练时间: 4-5小时

**使用方法：**
```bash
python3 -m verl.trainer.main_ppo --config configs/a6000_grpo.yaml
```

**适合场景：**
- 追求最佳性价比
- 需要高质量7B模型
- 显存充足但不想极致优化

---

### `a6000_ppo.yaml` - A6000激进方案

**适用硬件：** NVIDIA A6000 (48GB显存)

**训练配置：**
- 模型: Qwen/Qwen2.5-7B
- 算法: PPO (完整算法，带Critic网络)
- batch_size: 16
- n_agent: 1 (PPO单agent)
- 训练步数: 500
- 显存占用: ~45GB
- 训练时间: 6-8小时

**使用方法：**
```bash
python3 -m verl.trainer.main_ppo --config configs/a6000_ppo.yaml
```

**适合场景：**
- 追求最佳性能
- 有充足训练时间
- 需要精确的价值估计

---

## 🔧 配置文件结构

所有配置文件遵循统一的结构：

```yaml
# 数据配置
data:
  train_files: 数据路径
  train_batch_size: 批大小
  max_prompt_length: 最大prompt长度
  ...

# 算法配置
algorithm:
  adv_estimator: 算法类型 (grpo/gae)

# 模型配置
actor_rollout_ref:
  model:
    path: 模型路径
  rollout:
    n_agent: agent数量
    gpu_memory_utilization: GPU显存利用率
    ...

# 训练器配置
trainer:
  total_training_steps: 总训练步数
  experiment_name: 实验名称
  ...

# Search-R1特定配置
max_turns: 对话轮数
retriever:
  url: 检索器URL
  topk: 检索结果数量
```

---

## 📝 自定义配置

### 方法1: 修改现有配置

直接编辑配置文件：
```bash
vim configs/a6000_grpo.yaml
```

### 方法2: 命令行覆盖

使用命令行参数覆盖配置：
```bash
python3 -m verl.trainer.main_ppo \
    --config configs/a6000_grpo.yaml \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n_agent=10
```

### 方法3: 创建新配置

复制现有配置并修改：
```bash
cp configs/a6000_grpo.yaml configs/my_custom_config.yaml
# 编辑 my_custom_config.yaml
```

---

## 🎯 配置参数说明

### 关键参数对比

| 参数 | 保守值 | 平衡值 | 激进值 | 说明 |
|------|--------|--------|--------|------|
| `data.train_batch_size` | 32-64 | 16 | 8 | 批大小 |
| `actor_rollout_ref.rollout.n_agent` | 10 | 5 | 1 | GRPO agents |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.9 | 0.85 | 0.8 | GPU显存利用率 |
| `trainer.total_training_steps` | 300-500 | 500 | 500-1000 | 训练步数 |

### 显存优化参数

| 参数 | 启用 | 禁用 | 显存影响 |
|------|------|------|----------|
| `fsdp_config.param_offload` | true | false | 节省显存 |
| `fsdp_config.grad_offload` | true | false | 节省显存 |
| `fsdp_config.optimizer_offload` | true | false | 节省显存 |

---

## 🛠️ 故障排查

### 配置文件加载失败

确保在项目根目录执行：
```bash
cd /path/to/Search-R1
python3 -m verl.trainer.main_ppo --config configs/a6000_grpo.yaml
```

### 参数冲突

如果命令行参数和配置文件冲突，命令行参数优先：
```bash
# 配置文件中 batch_size=16，但命令行指定32
python3 -m verl.trainer.main_ppo \
    --config configs/a6000_grpo.yaml \
    data.train_batch_size=32  # 这个值会被使用
```

---

## 📚 相关文档

- [A6000_GUIDE.md](../A6000_GUIDE.md) - A6000优化配置指南
- [scripts/README.md](../scripts/README.md) - 脚本使用说明
- [ARCHITECTURE.md](../ARCHITECTURE.md) - 项目架构文档
