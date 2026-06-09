# Search-R1 脚本说明

本目录包含Search-R1项目的专业训练脚本，用于替代文档中的长命令行操作。

## 📁 脚本列表

### 数据准备脚本

#### `data_prepare.py`
**功能：** 准备Search-R1训练所需的NQ数据集

**用法：**
```bash
python scripts/data_prepare.py --output_dir data/nq_search
```

**参数：**
- `--output_dir`: 输出目录路径（默认: `data/nq_search`）
- `--dataset_name`: HuggingFace数据集名称（默认: `RUC-NLPIR/FlashRAG_datasets`）
- `--subset`: 数据集子集名称（默认: `nq`）

**输出：**
- `train.json`: 训练集 (~3000样本)
- `test.json`: 测试集 (~3000样本)

---

### 索引下载脚本

#### `download_index.py`
**功能：** 从HuggingFace Hub下载预构建的E5检索索引

**用法：**
```bash
python scripts/download_index.py --output_dir data/index
```

**参数：**
- `--output_dir`: 输出目录路径（默认: `data/index`）

**输出：**
- `e5_Flat.index`: E5检索索引 (~1GB)
- `wiki-18.jsonl`: Wikipedia语料 (~500MB)

---

### GPU验证脚本

#### `verify_gpu.py`
**功能：** 验证训练环境是否正确配置

**用法：**
```bash
python scripts/verify_gpu.py
```

**检查项目：**
- PyTorch和CUDA安装
- GPU可用性和显存容量
- vLLM安装
- 其他必需依赖包

---

### 训练脚本

#### `train_a6000.sh`
**功能：** A6000优化的GRPO训练脚本（使用GPU 0）

**用法：**
```bash
bash scripts/train_a6000.sh
```

#### `train_gpu.sh` ⭐ 推荐
**功能：** 灵活的GPU训练脚本，支持自定义GPU选择

**用法：**
```bash
# 使用默认GPU (GPU 0)
bash scripts/train_gpu.sh

# 指定GPU
bash scripts/train_gpu.sh --gpu 1

# 使用多个GPU
bash scripts/train_gpu.sh --gpu 0,1,2

# 完整自定义
bash scripts/train_gpu.sh --gpu 2 --config configs/a6000_ppo.yaml --name my-experiment
```

**参数说明：**
- `--gpu GPU_ID`: 指定GPU设备ID（支持多个，用逗号分隔）
- `--config CONFIG_FILE`: 指定配置文件
- `--data DATA_DIR`: 指定数据目录
- `--model MODEL_PATH`: 指定模型路径
- `--name EXPERIMENT_NAME`: 指定实验名称
- `--help`: 显示帮助信息

**特点：**
- 支持灵活的GPU选择
- 使用YAML配置文件
- 自动检查前置条件
- 详细的配置信息显示
- 彩色输出和错误提示
- 自动管理日志和检查点

**前置条件：**
- 数据已准备 (`data/nq_search/`)
- 检索器已启动 (`http://127.0.0.1:8000/retrieve`)

---

### 检索器脚本

#### `start_retriever.sh`
**功能：** 启动GPU加速的E5检索服务器

**用法：**
```bash
bash scripts/start_retriever.sh
```

**特点：**
- 自动检查索引文件和端口占用
- GPU加速（FAISS GPU）
- 详细的状态信息
- 自动日志记录

**服务信息：**
- URL: `http://127.0.0.1:8000/retrieve`
- 日志: `logs/retriever.log`

---

## 🔧 配置文件

配置文件位于 `../configs/` 目录：

### `configs/a6000_grpo.yaml`
A6000平衡方案配置（推荐）
- 模型: Qwen2.5-7B
- 算法: GRPO
- batch_size: 16
- n_agent: 5

### `configs/a6000_ppo.yaml`
A6000激进方案配置
- 模型: Qwen2.5-7B
- 算法: PPO (带Critic网络)
- batch_size: 16
- 显存占用更高 (~45GB)

---

## 📝 使用流程

### 完整训练流程：

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
bash scripts/train_a6000.sh
```

### 快速开始（一键启动）：

参考 `A6000_GUIDE.md` 中的第7节完整自动化脚本。

---

## 🛠️ 故障排查

### 脚本权限问题

如果遇到权限错误，请执行：
```bash
chmod +x scripts/*.sh
```

### Python依赖缺失

确保安装了所有依赖：
```bash
pip install torch transformers datasets ray hydra-core wandb accelerate huggingface_hub
pip install vllm==0.6.3
pip install flash-attn --no-build-isolation
```

### 检索器启动失败

检查端口8000是否被占用：
```bash
lsof -i :8000
```

---

## 📚 相关文档

- [REPRODUCTION_GUIDE.md](../REPRODUCTION_GUIDE.md) - 完整复现指南（推荐）
- [ARCHITECTURE.md](../ARCHITECTURE.md) - 项目架构文档
- [CODE_FLOW_ANALYSIS.md](../CODE_FLOW_ANALYSIS.md) - 代码流程分析
- [VERL_INTEGRATION.md](../VERL_INTEGRATION.md) - 框架集成文档
