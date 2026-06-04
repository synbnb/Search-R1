# Search-R1 项目架构文档

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 目录结构说明](#2-目录结构说明)
- [3. 核心模块详解](#3-核心模块详解)
- [4. 数据流程](#4-数据流程)
- [5. 训练流程](#5-训练流程)
- [6. 推理流程](#6-推理流程)
- [7. 配置说明](#7-配置说明)

---

## 1. 项目概述

Search-R1是一个强化学习框架，用于训练具有推理和搜索引擎调用能力的大语言模型。该项目基于veRL框架构建，扩展了DeepSeek-R1的思想，支持LLM学习在推理过程中调用搜索引擎获取外部知识。

### 1.1 核心特性
- **多种RL算法**：支持PPO、GRPO、Reinforce等强化学习算法
- **多种LLM支持**：支持Llama系列、Qwen2.5等模型
- **多种搜索引擎**：本地稀疏检索器(BM25)、本地密集检索器(E5)、在线搜索引擎
- **分布式训练**：支持多节点训练30B+大模型
- **工具调用能力**：训练LLM学习何时以及如何调用搜索引擎

### 1.2 技术栈
- **深度学习框架**：PyTorch, vLLM
- **分布式训练**：Ray, FSDP, Megatron-LM
- **检索引擎**：FAISS, Pyserini, E5嵌入模型
- **强化学习**：PPO, GRPO算法实现
- **API服务**：FastAPI, uvicorn

---

## 2. 目录结构说明

```
Search-R1/
├── docs/                          # 文档目录
│   ├── experiment_log.md         # 实验日志
│   ├── multinode.md              # 多节点训练说明
│   └── retriever.md              # 检索器配置说明
├── example/                       # 示例和脚本
│   ├── multinode/                # 多节点训练脚本
│   │   ├── train_ppo_multinode_32b.sh
│   │   ├── train_ppo_multinode_72b.sh
│   │   └── train_grpo_multinode_*.sh
│   ├── retriever/                # 检索器启动脚本
│   │   ├── retrieval_launch_bm25.sh      # BM25检索器
│   │   ├── retrieval_launch_ann.sh       # ANN密集检索器
│   │   ├── retrieval_launch_google.sh    # Google搜索API
│   │   ├── retrieval_launch_serpapi.sh   # SerpAPI集成
│   │   └── retrieval_launch_hierarchical.sh  # 分层检索
│   └── corpus.jsonl             # 示例语料库
├── public/                        # 公共资源（图片等）
├── scripts/                       # 工具脚本
│   ├── data_process/             # 数据处理脚本
│   │   ├── nq.py                 # Natural Questions数据处理
│   │   ├── nq_search.py          # NQ搜索格式处理
│   │   ├── nq_rag.py             # NQ RAG格式处理
│   │   ├── qa_search_test_merge.py
│   │   └── qa_search_train_merge.py
│   ├── download.py               # HuggingFace模型下载
│   ├── download.sh              # 下载脚本封装
│   ├── upload.py                 # 模型上传脚本
│   ├── upload.sh                 # 上传脚本封装
│   └── nq_hotpotqa/              # NQ+HotpotQA实验脚本
│       ├── data_process.sh
│       ├── evaluate.sh
│       └── v0.1|v0.2|v0.3/       # 不同版本的训练脚本
├── search_r1/                    # Search-R1核心模块
│   ├── __init__.py
│   ├── llm_agent/                # LLM智能体模块
│   │   ├── __init__.py
│   │   ├── generation.py         # 生成逻辑和工具调用
│   │   └── tensor_helper.py      # 张量操作辅助工具
│   └── search/                   # 搜索引擎模块
│       ├── __init__.py
│       ├── build_index.sh        # 索引构建脚本
│       ├── index_builder.py     # FAISS索引构建
│       ├── retrieval.py         # 检索核心实现
│       ├── retrieval_server.py  # FastAPI检索服务器
│       ├── retrieval_rerank_server.py  # 检索+重排序服务器
│       ├── retrieval_request.py  # 检索请求示例
│       ├── retrieval.sh          # 检索脚本
│       ├── google_search_server.py    # Google搜索API服务器
│       ├── serp_search_server.py      # SerpAPI服务器
│       └── rerank_server.py     # 重排序服务器
├── verl/                         # veRL强化学习框架
│   ├── models/                   # 模型实现
│   │   ├── transformers/         # HuggingFace Transformers集成
│   │   │   ├── llama.py         # Llama模型实现
│   │   │   ├── qwen2.py         # Qwen2模型实现
│   │   │   └── monkey_patch.py  # 模型补丁
│   │   ├── llama/               # Llama Megatron实现
│   │   │   └── megatron/        # Megatron-LM并行
│   │   ├── registry.py          # 模型注册表
│   │   └── weight_loader_registry.py  # 权重加载器注册
│   ├── trainer/                 # 训练器模块
│   │   ├── main_ppo.py          # PPO训练主入口
│   │   ├── main_ppo_format.py   # 格式化PPO训练
│   │   ├── main_generation.py   # 生成脚本
│   │   ├── main_eval.py         # 评估脚本
│   │   ├── ppo/                 # PPO算法实现
│   │   │   ├── core_algos.py    # 核心算法
│   │   │   └── ray_trainer.py   # Ray分布式训练器
│   │   └── fsdp_sft_trainer.py  # FSDP监督微调训练器
│   ├── workers/                  # 工作进程模块
│   │   ├── actor/               # Actor工作进程
│   │   │   ├── base.py          # Actor基类
│   │   │   ├── dp_actor.py      # 数据并行Actor
│   │   │   └── megatron_actor.py  # Megatron Actor
│   │   ├── critic/              # Critic工作进程
│   │   │   ├── base.py          # Critic基类
│   │   │   ├── dp_critic.py     # 数据并行Critic
│   │   │   └── megatron_critic.py  # Megatron Critic
│   │   ├── rollout/             # Rollout工作进程
│   │   │   ├── base.py          # Rollout基类
│   │   │   ├── vllm_rollout/    # vLLM Rollout实现
│   │   │   ├── hf_rollout.py    # HuggingFace Rollout
│   │   │   └── tokenizer.py     # 分词器
│   │   ├── reward_model/         # 奖励模型
│   │   │   ├── base.py          # 奖励模型基类
│   │   │   └── megatron/        # Megatron奖励模型
│   │   ├── sharding_manager/    # 分片管理器
│   │   │   ├── fsdp_vllm.py     # FSDP+vLLM分片
│   │   │   ├── fsdp_ulysses.py  # FSDP+Ulysses分片
│   │   │   └── megatron_vllm.py # Megatron+vLLM分片
│   │   ├── fsdp_workers.py      # FSDP工作进程
│   │   └── megatron_workers.py  # Megatron工作进程
│   ├── utils/                   # 工具模块
│   │   ├── reward_score/        # 奖励计算
│   │   │   ├── qa_em.py         # QA精确匹配奖励
│   │   │   ├── qa_em_format.py  # 格式化QA奖励
│   │   │   ├── countdown.py     # 倒计时奖励
│   │   │   ├── gsm8k.py         # GSM8K数学奖励
│   │   │   ├── math.py          # 数学问题奖励
│   │   │   └── multiply.py      # 乘法奖励
│   │   ├── dataset/             # 数据集处理
│   │   │   ├── rl_dataset.py    # RL数据集
│   │   │   └── rm_dataset.py    # 奖励模型数据集
│   │   ├── megatron/            # Megatron工具
│   │   │   ├── optimizer.py     # 优化器
│   │   │   ├── tensor_parallel.py  # 张量并行
│   │   │   ├── pipeline_parallel.py  # 流水线并行
│   │   │   ├── sequence_parallel.py  # 序列并行
│   │   │   └── memory.py        # 内存管理
│   │   ├── logger/              # 日志工具
│   │   ├── config.py            # 配置管理
│   │   ├── distributed.py       # 分布式工具
│   │   ├── tokenizer.py        # 分词器工具
│   │   └── tracking.py         # 追踪工具
│   ├── single_controller/       # 单控制器
│   │   ├── ray/                # Ray控制器
│   │   └── base/               # 基础控制器
│   ├── third_party/            # 第三方库
│   │   └── vllm/               # vLLM多版本支持
│   │       ├── vllm_v_0_3_1/
│   │       ├── vllm_v_0_4_2/
│   │       └── vllm_v_0_5_4/
│   └── protocol.py             # 协议定义
├── train_ppo.sh                # PPO训练启动脚本
├── train_grpo.sh               # GRPO训练启动脚本
├── infer.py                    # 推理脚本
├── retrieval_launch.sh         # 检索器启动脚本
├── setup.py                    # 安装配置
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
├── ARCHITECTURE.md             # 本文档
└── VERL_README.md              # veRL框架说明
```

---

## 3. 核心模块详解

### 3.1 search_r1/search - 搜索引擎模块

#### 核心文件

**index_builder.py - 索引构建器**
```python
class Index_Builder:
    """
    构建检索索引的工具类
    
    功能：
    1. BM25索引：使用Pyserini构建稀疏检索索引
    2. 密集索引：使用E5等嵌入模型构建FAISS向量索引
    3. 支持GPU加速（faiss_gpu）
    4. 支持多种索引类型（Flat, HNSW等）
    """
```

**retrieval.py - 检索核心实现**
```python
class BM25Retriever(BaseRetriever):
    """BM25稀疏检索器"""
    # 基于词频统计的精确匹配
    
class DenseRetriever(BaseRetriever):
    """密集检索器"""
    # 基于E5等嵌入模型的向量检索
    # 支持Flat和ANN索引
```

**retrieval_server.py - FastAPI服务器**
```python
@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    检索API端点
    
    输入：
    {
      "queries": ["查询1", "查询2"],
      "topk": 3,
      "return_scores": true
    }
    
    输出：
    {
      "result": [[文档1, 文档2, 文档3], ...]
    }
    """
```

#### 支持的检索器类型

| 检索器类型 | 实现类 | 特点 | 适用场景 |
|-----------|--------|------|----------|
| **BM25** | BM25Retriever | 词频匹配，无需GPU | 无嵌入模型领域 |
| **E5-Flat** | DenseRetriever+Flat | 精确向量匹配，需要GPU | 追求精度 |
| **E5-ANN** | DenseRetriever+ANN | 近似向量匹配，CPU友好 | 追求效率 |
| **Google** | google_search_server.py | 在线搜索API | 通用知识 |
| **SerpAPI** | serp_search_server.py | 聚合搜索API | 多搜索引擎 |

### 3.2 search_r1/llm_agent - LLM智能体模块

**generation.py - 生成和工具调用**
```python
"""
LLM生成逻辑，处理：
1. 搜索引擎调用工具
2. 推理-搜索循环
3. 响应格式化
"""
```

**tensor_helper.py - 张量操作辅助**
```python
"""
提供张量操作的辅助函数：
1. 张量切分和合并
2. 注意力掩码处理
3. 批处理操作
"""
```

### 3.3 verl/trainer - 训练器模块

**main_ppo.py - PPO训练主入口**
```python
"""
PPO训练流程：
1. 环境初始化（Ray集群）
2. 模型加载（Actor、Critic、Reference）
3. Worker创建（Rollout、Actor、Critic）
4. 数据加载（训练/验证集）
5. 训练循环：
   - Rollout生成
   - 奖励计算
   - PPO更新
   - 评估和保存
"""
```

**ray_trainer.py - Ray分布式训练器**
```python
class RayPPOTrainer:
    """
    Ray分布式PPO训练器
    
    管理：
    1. 资源池（CPU、GPU）
    2. Worker组（Actor、Rollout、Critic）
    3. 训练循环和同步
    4. 检查点保存和加载
    """
```

### 3.4 verl/workers - 工作进程模块

**Actor Worker**
```python
# 负责策略梯度更新和生成
# 位置：verl/workers/actor/
# 类型：FSDP或Megatron并行策略
```

**Rollout Worker**
```python
# 负责经验采样和生成
# 位置：verl/workers/rollout/
# 后端：vLLM（高效推理引擎）
```

**Critic Worker**
```python
# 负责价值函数估计
# 位置：verl/workers/critic/
# 用于优势函数计算
```

### 3.5 verl/utils/reward_score - 奖励计算模块

**qa_em.py - QA精确匹配奖励**
```python
def compute_score_em(solution_str, ground_truth, format_score):
    """
    计算QA任务的精确匹配奖励
    
    逻辑：
    1. 提取答案（从<answer></answer>标签）
    2. 与ground_truth比较
    3. 返回匹配分数（0或1）
    """
```

**countdown.py - 倒计时奖励**
```python
"""
限制搜索次数的奖励机制
超过max_turns会有惩罚
"""
```

---

## 4. 数据流程

### 4.1 完整训练数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌──────────────────────────────────────┐
         │  原始数据集 (HuggingFace)             │
         │  - NQ (Natural Questions)            │
         │  - HotpotQA                          │
         │  - TriviaQA                          │
         └──────────────────────────────────────┘
                              │
                              ▼
         ┌──────────────────────────────────────┐
         │  数据处理 (scripts/data_process/)    │
         │  - nq_search.py                      │
         │  - 添加搜索模板                      │
         │  - 格式化为Parquet                   │
         └──────────────────────────────────────┘
                              │
                              ▼
         ┌──────────────────────────────────────┐
         │  训练数据格式                         │
         │  {                                   │
         │    "prompt": [用户问题],             │
         │    "ability": "fact-reasoning",     │
         │    "reward_model": {                 │
         │      "style": "rule",                │
         │      "ground_truth": [答案列表]      │
         │    }                                  │
         │  }                                   │
         └──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        训练循环阶段                                │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────▼────────────────────┐
         │  1. Rollout生成                        │
         │  - Actor模型生成响应                    │
         │  - 检测<search>调用                    │
         │  - 调用检索API获取外部知识             │
         │  - 继续生成直到<answer>                │
         └────────────────────┬────────────────────┘
                              │
                              ▼
         ┌────────────────────▼────────────────────┐
         │  2. 奖励计算                            │
         │  - 提取模型答案                        │
         │  - 与ground_truth比较                  │
         │  - 计算EM（精确匹配）奖励              │
         │  - 格式正确性奖励                      │
         └────────────────────┬────────────────────┘
                              │
                              ▼
         ┌────────────────────▼────────────────────┐
         │  3. 优势函数计算                        │
         │  - Critic估计价值函数                   │
         │  - GAE计算优势                         │
         │  - KL散度惩罚                          │
         └────────────────────┬────────────────────┘
                              │
                              ▼
         ┌────────────────────▼────────────────────┐
         │  4. PPO更新                             │
         │  - Actor策略梯度更新                   │
         │  - Critic价值函数更新                   │
         │  - Reference模型KL约束                 │
         │  - 多轮迭代优化                        │
         └────────────────────┬────────────────────┘
                              │
                              ▼
         ┌────────────────────▼────────────────────┐
         │  5. 评估和保存                          │
         │  - 定期在测试集评估                    │
         │  - 保存检查点                          │
         │  - 记录WandB日志                       │
         └───────────────────────────────────────┘
```

### 4.2 单个样本的推理流程

```
用户问题: "Mike Barnett negotiated many contracts including which player..."
     │
     ▼
┌─────────────────────────────────────────┐
│  LLM生成推理                              │
│  "I need to find information about..."   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  决策：需要搜索                          │
│  生成: <search> Mike Barnett contracts   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  调用检索API                              │
│  POST /retrieve                           │
│  { "queries": ["Mike Barnett contracts"]}│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  检索器返回结果                           │
│  Doc 1: "Mike Barnett negotiated..."     │
│  Doc 2: "Andrei Kirilyuk..."            │
│  Doc 3: "NHL contract negotiations..."  │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  LLM继续推理（带检索结果）               │
│  <information>                            │
│  Doc 1: ...                               │
│  Doc 2: ...                               │
│  </information>                           │
│  "Based on the search results..."         │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  最终答案                                │
│  <answer> Andrei Kirilyuk </answer>     │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  奖励计算                                │
│  提取答案: "Andrei Kirilyuk"            │
│  与ground_truth比较                     │
│  奖励 = 1.0 (匹配成功)                  │
└─────────────────────────────────────────┘
```

### 4.3 检索器调用流程

```
训练/推理进程
     │
     │  HTTP POST
     ▼
┌─────────────────────────────────────────┐
│  FastAPI检索服务器                       │
│  (retrieval_server.py)                  │
│  端口: 8000                              │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  查询编码                                 │
│  - E5模型: "query: ..."                 │
│  - GPU加速推理                           │
│  - 返回768维向量                         │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  FAISS向量搜索                           │
│  - 内积相似度计算                       │
│  - Top-K检索                             │
│  - GPU/CPU加速                           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  文档加载                                 │
│  - 从语料库加载索引对应的文档            │
│  - 返回title和contents                   │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  结果返回                                 │
│  JSON格式包含文档和可选分数              │
└─────────────────────────────────────────┘
```

---

## 5. 训练流程

### 5.1 完整训练步骤

```bash
# 步骤1: 环境准备
conda create -n searchr1 python=3.9
conda activate searchr1
pip install torch==2.4.0
pip install vllm==0.6.3
pip install -e .
pip install flash-attn

# 步骤2: 检索器环境（可选）
conda create -n retriever python=3.10
conda activate retriever
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi

# 步骤3: 下载语料库和索引
python scripts/download.py --save_path ./data/index
cat ./data/index/part_* > ./data/index/e5_Flat.index
gzip -d ./data/index/wiki-18.jsonl.gz

# 步骤4: 处理数据集
python scripts/data_process/nq_search.py
# 输出: data/nq_search/train.parquet, test.parquet

# 步骤5: 启动检索服务器
conda activate retriever
bash retrieval_launch.sh
# 服务运行在 http://127.0.0.1:8000

# 步骤6: 启动RL训练
conda activate searchr1
bash train_ppo.sh
```

### 5.2 训练配置详解

**train_ppo.sh关键参数**

```bash
# 数据配置
data.train_files=$DATA_DIR/train.parquet
data.val_files=$DATA_DIR/test.parquet
data.max_prompt_length=4096          # 最大提示长度
data.max_response_length=500         # 最大响应长度
data.max_obs_length=500              # 最大观察（检索结果）长度

# 模型配置
actor_rollout_ref.model.path=$BASE_MODEL
actor_rollout_ref.rollout.name=vllm   # 使用vLLM推理引擎

# 学习率配置
actor_rollout_ref.actor.optim.lr=1e-6         # Actor学习率
critic.optim.lr=1e-5                          # Critic学习率

# PPO配置
actor_rollout_ref.actor.ppo_mini_batch_size=256
actor_rollout_ref.actor.ppo_micro_batch_size=64
actor_rollout_ref.rollout.temperature=1       # 采样温度
algorithm.kl_ctrl.kl_coef=0.001              # KL惩罚系数

# 检索器配置
retriever.url="http://127.0.0.1:8000/retrieve"
retriever.topk=3                            # 检索Top-K文档
max_turns=2                                 # 最大搜索轮数

# 训练配置
trainer.total_epochs=15
trainer.total_training_steps=1005
trainer.save_freq=100                      # 每100步保存
trainer.test_freq=50                       # 每50步评估
```

---

## 6. 推理流程

### 6.1 推理步骤

```bash
# 1. 启动检索服务器
conda activate retriever
bash retrieval_launch.sh

# 2. 运行推理
conda activate searchr1
python infer.py
```

### 6.2 推理代码流程

```python
# infer.py核心流程

# 1. 加载模型
model_id = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. 准备提示
prompt = f"""Answer the given question.
You must conduct reasoning inside <think and  first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>
Question: {question}"""

# 3. 生成循环
while True:
    # 生成文本
    outputs = model.generate(input_ids, max_new_tokens=1024)
    
    # 检查是否完成
    if outputs[-1] in eos_tokens:
        break
    
    # 检测搜索调用
    query = extract_query(outputs)
    if query:
        # 调用检索API
        search_results = search(query)
        # 添加到提示
        prompt += f"<information>{search_results}</information>"
```

---

## 7. 配置说明

### 7.1 检索器选择

| 场景 | 推荐检索器 | 启动脚本 |
|------|-----------|----------|
| 私有语料，无GPU | BM25 | `retrieval_launch_bm25.sh` |
| 私有语料，有GPU | E5-Flat | `retrieval_launch_ann.sh` (修改为Flat) |
| 通用知识 | SerpAPI | `retrieval_launch_serpapi.sh` |
| 特定在线搜索 | Google | `retrieval_launch_google.sh` |

### 7.2 并行策略选择

| 模型规模 | 推荐策略 | 配置 |
|---------|---------|------|
| ≤7B | FSDP | `actor_rollout_ref.actor.strategy=fsdp` |
| 7B-30B | Megatron-LM | `actor_rollout_ref.actor.strategy=megatron` |
| >30B | 多节点Megatron | 参考example/multinode/ |

### 7.3 RL算法选择

| 算法 | 特点 | 脚本 |
|------|------|------|
| **PPO** | 稳定，样本效率高 | `train_ppo.sh` |
| **GRPO** | 无Critic，更简单 | `train_grpo.sh` |
| **Reinforce** | 基础策略梯度 | 自定义配置 |

---

## 8. 关键技术点

### 8.1 搜索工具集成

**工具调用格式**
```
<search> 查询内容 </search>
```

**检索结果注入**
```
<information>
Doc 1 (Title: 标题) 文档内容
Doc 2 (Title: 标题) 文档内容
Doc 3 (Title: 标题) 文档内容
</information>
```

### 8.2 奖励函数设计

**QA任务奖励 = EM分数 + 格式分数**
```python
# EM（Exact Match）: 答案是否完全匹配
# 格式分: 是否正确使用<answer>标签
final_reward = em_score + format_score
```

### 8.3 多轮搜索机制

```
轮次1: 推理 → 搜索(查询1) → 推理(含结果)
轮次2: 推理 → 搜索(查询2) → 推理(含结果)
...
最终: <answer>最终答案</answer>
```

---

## 9. 故障排查

### 9.1 常见问题

**问题1: 检索器连接失败**
```bash
# 检查检索器是否运行
curl http://127.0.0.1:8000/retrieve

# 检查防火墙
sudo ufw allow 8000
```

**问题2: CUDA OOM**
```bash
# 减少batch size
data.train_batch_size=256  # 原值512

# 启用参数卸载
actor_rollout_ref.actor.fsdp_config.param_offload=true
actor_rollout_ref.actor.fsdp_config.grad_offload=true
```

**问题3: vLLM兼容性问题**
```bash
# 设置注意力后端
export VLLM_ATTENTION_BACKEND=XFORMERS
```

---

## 10. 扩展开发

### 10.1 添加新的检索器

1. 在`search_r1/search/`创建新的服务器脚本
2. 实现FastAPI的`/retrieve`端点
3. 在训练脚本中配置`retriever.url`

### 10.2 添加新的数据集

1. 在`scripts/data_process/`创建处理脚本
2. 实现标准格式:
```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "...",
    "reward_model": {
        "style": "rule",
        "ground_truth": [...]
    }
}
```
3. 在`verl/utils/reward_score/`添加奖励函数

---

## 11. 引用

如果您在研究中使用Search-R1，请引用：

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

---

## 12. 联系方式

- 项目主页: https://github.com/PeterGriffinJin/Search-R1
- 论文: https://arxiv.org/abs/2503.09516
- WandB日志: https://wandb.ai/peterjin/Search-R1-v0.2

---

*最后更新: 2026-06-04*
