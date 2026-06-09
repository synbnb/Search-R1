# Search-R1 与 verl 框架耦合详解

## 📚 目录
- [1. 耦合概述](#1-耦合概述)
- [2. 核心耦合点](#2-核心耦合点)
- [3. 详细代码分析](#3-详细代码分析)
- [4. 数据流对接](#4-数据流对接)
- [5. 自定义扩展](#5-自定义扩展)
- [6. 耦合架构图](#6-耦合架构图)

---

## 1. 耦合概述

### 1.1 耦合关系说明

Search-R1**基于verl框架构建**，将verl作为底层强化学习基础设施，Search-R1专注于搜索增强推理的特定逻辑。

**耦合模式：**
```
┌─────────────────────────────────────────────────────────────┐
│                    Search-R1 项目                            │
│  (搜索增强推理的特定逻辑: 搜索工具、检索服务、多轮交互)      │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ 使用/继承/扩展
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      verl 框架                               │
│  (通用RL训练基础设施: PPO、分布式训练、Worker管理)          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 耦合程度

| 组件 | 耦合程度 | 说明 |
|------|----------|------|
| **核心训练循环** | 强依赖 | 完全使用verl的PPO训练器 |
| **数据协议** | 强依赖 | 使用verl的DataProto标准 |
| **Worker管理** | 强依赖 | 使用verl的Ray分布式Workers |
| **工具函数** | 弱依赖 | 可选使用verl的工具 |
| **搜索逻辑** | 独立 | Search-R1自定义，不依赖verl |

---

## 2. 核心耦合点

### 2.1 导入关系总览

```python
# Search-R1 对 verl 的导入关系

# ===== 核心训练器 =====
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

# ===== Workers =====
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

# ===== 数据协议 =====
from verl import DataProto

# ===== 工具函数 =====
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking

# ===== 奖励函数 =====
from verl.utils.reward_score import qa_em

# ===== 模型相关 =====
from verl.models.registry import register_model_wrapper
from verl.models.transformers.monkey_patch import update_flash_attn
```

### 2.2 文件级耦合关系

```
Search-R1 核心文件                │  依赖的 verl 组件
─────────────────────────────────────────────────────────
verl/trainer/main_ppo.py         │  (Search-R1自定义，但使用verl组件)
├── 导入 RayPPOTrainer           │  ← verl/trainer/ppo/ray_trainer.py
├── 导入 ActorRolloutRefWorker   │  ← verl/workers/fsdp_workers.py
├── 导入 CriticWorker             │  ← verl/workers/fsdp_workers.py
├── 使用 RewardManager            │  ← 自定义，但使用verl的DataProto
└── 调用 trainer.fit()            │  ← verl的训练循环

search_r1/llm_agent/generation.py│  (Search-R1自定义多轮交互)
├── 使用 DataProto                 │  ← verl的数据协议
└── 调用 actor_rollout_wg          │  ← verl的Worker接口

search_r1/search/*.py             │  (完全独立，不依赖verl)
└── 检索服务、索引构建等            │  ← 独立实现

verl/workers/*.py                  │  (verl框架核心，Search-R1直接使用)
├── fsdp_workers.py                │  ← FSDP并行策略
├── megatron_workers.py            │  ← Megatron并行策略
└── rollout/*.py                   │  ← vLLM推理引擎

verl/utils/*.py                    │  (工具函数)
├── reward_score/qa_em.py          │  ← 奖励计算
├── hf_tokenizer                   │  ├── Tokenizer工具
├── tracking.py                    │  └── 实验追踪
└── dataset/rl_dataset.py          │  ← 数据集加载
```

---

## 3. 详细代码分析

### 3.1 主训练入口 (main_ppo.py)

#### 代码位置
`/home/devuser/workspace/agentrl/Search-R1/verl/trainer/main_ppo.py`

#### 耦合分析

```python
# ===== 强耦合点1: 导入verl核心组件 =====
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import qa_em
from verl import DataProto

# ===== 强耦合点2: 使用verl的配置系统 =====
@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """
    Hydra配置系统是verl的一部分
    配置文件: verl/trainer/config/ppo_trainer.yaml
    """
    
    # ===== 强耦合点3: Ray分布式初始化 =====
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true'}})
    
    # ===== 强耦合点4: Worker类选择 =====
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # 使用verl的FSDP Worker
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # 使用verl的Megatron Worker
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    
    # ===== 强耦合点5: 资源池管理 =====
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, 
        mapping=mapping
    )
    
    # ===== 强耦合点6: 创建PPO训练器 =====
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,          # Search-R1自定义
        val_reward_fn=val_reward_fn,    # Search-R1自定义
    )
    
    # ===== 强耦合点7: 执行训练 =====
    trainer.init_workers()
    trainer.fit()  # verl的训练循环
```

#### 关键对接点

| 对接点 | verl提供 | Search-R1提供 | 说明 |
|--------|----------|---------------|------|
| **训练器** | `RayPPOTrainer` | 配置、奖励函数 | Search-R1只需配置和自定义奖励 |
| **Workers** | `ActorRolloutRefWorker`, `CriticWorker` | 选择策略(FSDP/Megatron) | verl提供多种并行策略 |
| **资源管理** | `ResourcePoolManager`, `Role` | GPU资源分配 | verl管理分布式资源 |
| **数据协议** | `DataProto` | 奖励计算 | Search-R1计算奖励，返回DataProto格式 |

---

### 3.2 奖励函数 (RewardManager)

#### 代码位置
`/home/devuser/workspace/agentrl/Search-R1/verl/trainer/main_ppo.py` 第32-97行

#### 耦合分析

```python
class RewardManager():
    """
    Search-R1自定义的奖励管理器
    但完全使用verl的数据协议
    """
    
    def __init__(self, tokenizer, num_examine, format_score=0.):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
    
    def __call__(self, data: DataProto):
        """
        ===== 关键耦合点 =====
        输入: DataProto (verl的标准数据协议)
        输出: reward_tensor (verl期望的tensor格式)
        """
        
        # ===== 使用verl的DataProto访问 =====
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            # ===== DataProto的item访问 =====
            data_item = data[i]  # DataProtoItem (verl提供的数据结构)
            
            # ===== 访问tensor数据 =====
            prompt_ids = data_item.batch['prompts']
            response_ids = data_item.batch['responses']
            
            # ===== 访问非tensor数据 =====
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            
            # ===== 解码和奖励计算 (Search-R1自定义) =====
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # ===== 使用verl的奖励函数 =====
            from verl.utils.reward_score import qa_em
            score = qa_em.compute_score_em(
                solution_str=sequences_str, 
                ground_truth=ground_truth,
                format_score=self.format_score
            )
            
            # ===== 设置verl期望的奖励格式 =====
            reward_tensor[i, valid_response_length - 1] = score
        
        return reward_tensor
```

#### 数据协议对接

```
┌─────────────────────────────────────────────────────────────┐
│              verl.DataProto 数据结构                          │
├─────────────────────────────────────────────────────────────┤
│  batch (dict):                                               │
│    ├─ input_ids: Tensor [batch_size, seq_len]             │
│    ├─ attention_mask: Tensor [batch_size, seq_len]         │
│    ├─ position_ids: Tensor [batch_size, seq_len]           │
│    ├─ responses: Tensor [batch_size, response_len]         │
│    └─ old_log_probs: Tensor [batch_size, seq_len]         │
│                                                              │
│  non_tensor_batch (dict):                                    │
│    ├─ data_source: str                                      │
│    ├─ reward_model: dict                                    │
│    │   └─ ground_truth: List[str]                           │
│    └─ uid: int                                             │
│                                                              │
│  meta_info (dict):                                           │
│    └─ 各种元信息                                            │
└─────────────────────────────────────────────────────────────┘
              ↑ Search-R1必须遵循这个协议
              
Search-R1的RewardManager:
┌─────────────────────────────────────────────────────────────┐
│  输入: data (DataProto) ← 来自verl训练循环                   │
│  ↓                                                          │
│  访问: data_item.batch['responses']                       │
│  访问: data_item.non_tensor_batch['reward_model']           │
│  ↓                                                          │
│  计算: 调用qa_em.compute_score_em() ← verl提供的奖励函数      │
│  ↓                                                          │
│  输出: reward_tensor ← verl期望的格式 [batch_size, seq_len] │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.3 LLM生成管理器 (LLMGenerationManager)

#### 代码位置
`/home/devuser/workspace/agentrl/Search-R1/search_r1/llm_agent/generation.py`

#### 耦合分析

```python
from verl import DataProto

class LLMGenerationManager:
    """
    Search-R1的多轮交互管理器
    但必须对接verl的接口
    """
    
    def __init__(self, tokenizer, actor_rollout_wg, config):
        """
        ===== 关键耦合点 =====
        actor_rollout_wg: 来自verl的Worker
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg  # verl的Worker实例
        self.config = config
    
    def run_llm_loop(self, gen_batch, initial_input_ids):
        """
        ===== 关键耦合点 =====
        输入: gen_batch (DataProto) ← verl训练器传入
        输出: final_output (DataProto) → 返回给verl训练器
        """
        
        # ===== 使用verl的DataProto =====
        rollings = gen_batch  # 保持DataProto格式
        
        # ===== 调用verl Worker的生成方法 =====
        gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
        
        # ===== 处理响应 (Search-R1自定义) =====
        responses_ids, responses_str = self._postprocess_responses(
            gen_output.batch['responses']
        )
        
        # ===== 执行搜索和推理 (Search-R1自定义) =====
        next_obs, dones, valid_action, is_search = self.execute_predictions(
            responses_str, self.tokenizer.pad_token, active_mask
        )
        
        # ===== 返回verl期望的格式 =====
        return self._compose_final_output(
            original_left_side, 
            original_right_side, 
            meta_info
        )
```

#### Worker接口对接

```
┌─────────────────────────────────────────────────────────────┐
│         verl的ActorRolloutRefWorker接口                       │
├─────────────────────────────────────────────────────────────┤
│  class ActorRolloutRefWorker:                                │
│                                                              │
│      def generate_sequences(self, input_ids: DataProto):    │
│          """                                                 │
│          verl提供的生成接口                                    │
│          输入: DataProto (包含input_ids等)                    │
│          输出: DataProto (包含responses等)                    │
│          """                                                  │
│          # 使用vLLM进行快速推理                                │
│          # 返回生成的序列                                       │
└─────────────────────────────────────────────────────────────┘
                         ▲
                         │ Search-R1调用
                         │
┌─────────────────────────────────────────────────────────────┐
│  Search-R1的LLMGenerationManager                             │
├─────────────────────────────────────────────────────────────┤
│  def run_llm_loop():                                         │
│      # 准备DataProto格式                                      │
│      rollings = DataProto.from_dict({...})                  │
│                                                              │
│      # 调用verl Worker的生成方法                              │
│      gen_output = self.actor_rollout_wg.generate_sequences( │
│          rollings  # DataProto格式                            │
│      )                                                        │
│                                                              │
│      # 处理生成结果 (Search-R1自定义逻辑)                     │
│      responses = gen_output.batch['responses']               │
│      ... (多轮搜索、推理等)                                   │
│                                                              │
│      # 返回DataProto格式                                     │
│      return DataProto.from_dict({...})                      │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.4 数据集对接

#### 代码位置
`/home/devuser/workspace/agentrl/Search-R1/verl/utils/dataset/rl_dataset.py`

#### 转换逻辑

```python
# Search-R1数据处理 (scripts/data_process/nq_search.py)
def process_fn(example, idx):
    """
    将NQ数据集转换为verl期望的格式
    """
    data = {
        "data_source": "nq",
        "prompt": [{
            "role": "user",
            "content": question,  # 带搜索提示词
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": golden_answers
        }
    }
    return data

# 保存为parquet
train_dataset.to_parquet('train.parquet')

# ===== verl数据集加载 =====
from verl.utils.dataset.rl_dataset import RLHFDataset

dataset = RLHFDataset(
    data_paths=['train.parquet'],
    tokenizer=tokenizer,
    max_length=512,
    # verl会自动处理chat_template
)

# verl期望的输出格式:
# {
#   'input_ids': Tensor,
#   'attention_mask': Tensor,
#   'position_ids': Tensor,
#   'prompts': Tensor,           # ← 用于计算KL
#   'non_tensor_batch': {         # ← 用于计算奖励
#     'data_source': List[str],
#     'reward_model': {
#       'ground_truth': List[List[str]]
#     }
#   }
# }
```

---

## 4. 数据流对接

### 4.1 训练循环数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    verl训练循环 (RayPPOTrainer.fit())          │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ 使用verl的DataLoader
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  ① 数据加载 (verl.utils.dataset.RLHFDataset)                 │
│      读取parquet → tokenize → 生成DataProto                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ 调用Search-R1自定义
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  ② 多轮生成 (Search-R1: LLMGenerationManager.run_llm_loop)  │
│      输入DataProto → 多轮搜索交互 → 输出DataProto             │
│      内部调用: actor_rollout_wg.generate_sequences()         │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ 返回DataProto
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  ③ 奖励计算 (Search-R1: RewardManager)                      │
│      输入DataProto → 提取答案 → 计算EM奖励 → reward_tensor    │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ 使用verl的算法
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  ④ 优势计算 (verl.trainer.ppo.core_algos)                   │
│      计算GAE优势或GRPO优势                                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ 使用verl的Workers
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  ⑤ 模型更新 (verl.workers)                                   │
│      Actor更新 → Critic更新 (PPO)                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ 回到训练循环
                    └─── 重复②-⑤
```

### 4.2 关键数据结构转换

```python
# ===== 转换1: 数据集 → DataProto =====
# Search-R1生成 (scripts/data_process/):
data = {
    "prompt": [{"role": "user", "content": "问题..."}],
    "reward_model": {"ground_truth": ["答案"]}
}
# 保存为parquet

# verl加载 (RLHFDataset):
dataset = RLHFDataset(data_paths=['train.parquet'])
batch = dataset[0]  # 自动转换为DataProto

# ===== 转换2: DataProto → Worker生成 =====
# verl训练器传入:
batch = DataProto.from_dict({
    'input_ids': tensor,
    'attention_mask': tensor,
    'position_ids': tensor
})

# Search-R1接收:
def run_llm_loop(self, gen_batch: DataProto, initial_input_ids):
    rollings = gen_batch  # 保持DataProto格式
    
    # 调用verl Worker
    gen_output = self.actor_rollout_wg.generate_sequences(rollings)

# ===== 转换3: Worker输出 → DataProto =====
# verl Worker返回:
gen_output = DataProto.from_dict({
    'responses': tensor,
    'log_probs': tensor
})

# Search-R1处理:
responses = gen_output.batch['responses']
# ... 多轮搜索逻辑 ...

# 返回verl期望格式:
return DataProto.from_dict({
    'input_ids': 完整对话,
    'responses': 响应部分,
    'info_mask': 信息掩码
})

# ===== 转换4: DataProto → 奖励 =====
# verl训练器传入:
data = DataProto.from_dict({
    'responses': tensor,
    'non_tensor_batch': {
        'reward_model': {'ground_truth': ['答案']}
    }
})

# Search-R1计算:
def __call__(self, data: DataProto):
    for i in range(len(data)):
        item = data[i]
        ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
        score = compute_score_em(...)
        
    # 返回verl期望格式:
    return reward_tensor  # [batch_size, seq_len]
```

---

## 5. 自定义扩展

### 5.1 Search-R1的自定义部分

```python
# ===== 完全独立，不依赖verl =====

# 1. 检索服务 (search_r1/search/*.py)
class RetrievalServer:
    """完全独立的FastAPI服务"""
    @app.post("/retrieve")
    def retrieve_endpoint(request):
        # 检索逻辑
        pass

# 2. 索引构建 (search_r1/search/index_builder.py)
class Index_Builder:
    """构建FAISS索引"""
    def build_dense_index(self):
        # 索引构建逻辑
        pass

# 3. 工具调用解析 (search_r1/llm_agent/generation.py)
def execute_predictions(self, predictions, ...):
    """解析<search>和<answer>标签"""
    # 正则匹配
    # 调用检索API
    # 格式化结果
    pass

# 4. 张量操作 (search_r1/llm_agent/tensor_helper.py)
class TensorHelper:
    """处理token级别的张量操作"""
    def concatenate_with_padding(self, tensors):
        # 拼接和padding逻辑
        pass
```

### 5.2 与verl的集成点

```python
# ===== 集成点1: 训练配置 =====
# train_ppo.sh
python -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    retriever.url="http://127.0.0.1:8000/retrieve" \  # Search-R1参数
    max_turns=2 \                                            # Search-R1参数
    ...

# ===== 集成点2: Worker初始化 =====
# verl/workers/fsdp_workers.py (Search-R1不修改)
class ActorRolloutRefWorker:
    def init_model(self):
        # verl的模型加载逻辑
        pass
    
    def generate_sequences(self, inputs):
        # verl的vLLM生成逻辑
        # Search-R1通过调用此方法获得生成结果
        pass

# ===== 集成点3: 训练循环注入 =====
# Search-R1通过LLMGenerationManager注入自定义逻辑
# 但仍需遵循verl的接口规范

# ===== 集成点4: 配置系统 =====
# verl使用Hydra配置系统
# Search-R1在配置文件中添加自定义参数:
# verl/trainer/config/ppo_trainer.yaml
max_turns: 2
retriever:
    url: "http://127.0.0.1:8000/retrieve"
    topk: 3
```

---

## 6. 耦合架构图

### 6.1 层级架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Search-R1 应用层                           │
│  (搜索增强推理的特定逻辑)                                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LLMGenerationManager                              │  │
│  │  - run_llm_loop() (多轮搜索循环)                    │  │
│  │  - execute_predictions() (工具调用解析)              │  │
│  │  - batch_search() (检索API调用)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RewardManager                                      │  │
│  │  - __call__() (QA奖励计算)                           │  │
│  │  - 使用verl.utils.reward_score.qa_em                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  检索服务 (RetrievalServer)                         │  │
│  │  - 完全独立，不依赖verl                             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │ 使用/扩展
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     verl 框架层                               │
│  (通用RL训练基础设施)                                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │  RayPPOTrainer (训练循环)                            │  │
│  │  - fit() (主训练循环)                                │  │
│  │  - 调用LLMGenerationManager.run_llm_loop()             │  │
│  │  - 调用RewardManager.__call__()                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Workers (模型计算)                                  │  │
│  │  - ActorRolloutRefWorker (Actor + Rollout + Ref)     │  │
│  │  - CriticWorker (Critic)                              │  │
│  │  - generate_sequences() (vLLM推理)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  核心算法 (core_algos)                               │  │
│  │  - compute_gae_advantage_return() (GAE)             │  │
│  │  - compute_grpo_outcome_advantage() (GRPO)           │  │
│  │  - compute_policy_loss() (PPO Clipped Objective)      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  数据协议 (DataProto)                                │  │
│  │  - 标准化的数据交换格式                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │ 基于使用
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    底层依赖                                  │
├─────────────────────────────────────────────────────────────┤
│  Ray (分布式执行)                                           │
│  vLLM (快速推理)                                             │
│  FSDP/Megatron (模型并行)                                    │
│  PyTorch (深度学习框架)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 数据流架构

```
┌─────────────────────────────────────────────────────────────┐
│                     数据准备阶段                              │
└─────────────────────────────────────────────────────────────┘

原始QA数据集 (NQ, HotpotQA, etc.)
        │
        ▼ Search-R1处理
scripts/data_process/nq_search.py
        │
        ▼ 生成parquet (遵循verl格式)
{
    "prompt": [{"role": "user", "content": "..."}],
    "reward_model": {"ground_truth": ["..."]}
}
        │
        ▼ verl加载
verl.utils.dataset.RLHFDataset
        │
        ▼ 输出DataProto
DataProto {
    batch: {input_ids, attention_mask, position_ids},
    non_tensor_batch: {data_source, reward_model}
}

┌─────────────────────────────────────────────────────────────┐
│                     训练执行阶段                              │
└─────────────────────────────────────────────────────────────┘

RayPPOTrainer.fit()
        │
        ├─→ DataLoader → DataProto (batch)
        │
        ├─→ LLMGenerationManager.run_llm_loop()
        │       │
        │       ├─→ 接收DataProto
        │       │
        │       ├─→ actor_rollout_wg.generate_sequences()
        │       │       │ (verl Worker)
        │       │       └─→ 返回DataProto
        │       │
        │       ├─→ execute_predictions() (Search-R1)
        │       │       └─→ 调用检索API
        │       │
        │       └─→ 返回DataProto
        │
        ├─→ RewardManager.__call__()
        │       │
        │       ├─→ 接收DataProto
        │       │
        │       ├─→ qa_em.compute_score_em()
        │       │       │ (verl奖励函数)
        │       │       └─→ 返回分数
        │       │
        │       └─→ 返回reward_tensor
        │
        ├─→ compute_advantage() (verl算法)
        │       └─→ 返回advantages
        │
        └─→ Actor/Critic.update() (verl Workers)
                └─→ 更新模型参数

┌─────────────────────────────────────────────────────────────┐
│                     评估阶段                                  │
└─────────────────────────────────────────────────────────────┘

RayPPOTrainer.validate()
        │
        ├─→ LLMGenerationManager.run_llm_loop()
        │       │ (do_sample=False, 贪婪解码)
        │       └─→ 返回DataProto
        │
        ├─→ RewardManager.__call__()
        │       └─→ 计算奖励
        │
        └─→ 统计准确率等指标
```

---

## 7. 关键耦合点总结

### 7.1 强耦合点 (必须遵循verl规范)

| 耦合点 | verl要求 | Search-R1实现 |
|--------|----------|---------------|
| **数据格式** | `DataProto` | 必须遵循DataProto协议 |
| **Worker接口** | `generate_sequences()` | 调用时传入/返回DataProto |
| **奖励格式** | `reward_tensor[batch, seq_len]` | 最后位置给奖励 |
| **配置系统** | Hydra YAML | 必须符合verl配置结构 |
| **资源管理** | `ResourcePoolManager` | 必须注册Worker角色 |

### 7.2 弱耦合点 (可自定义)

| 组件 | verl提供 | Search-R1自定义 |
|------|----------|----------------|
| **奖励计算** | 基础EM奖励 | 可扩展多维度奖励 |
| **生成逻辑** | 简单生成 | 多轮搜索-推理循环 |
| **工具调用** | 无 | 检索工具、解析逻辑 |
| **数据处理** | 基础处理 | 特定数据集处理 |

### 7.3 完全独立部分

| 组件 | 说明 |
|------|------|
| **检索服务** | 独立的FastAPI服务 |
| **索引构建** | 独立的FAISS索引系统 |
| **检索API** | 独立的HTTP调用逻辑 |
| **工具解析** | 独立的标签解析逻辑 |

---

## 8. 代码修改指南

### 8.1 如何添加自定义检索器

```python
# ===== 步骤1: 创建检索服务 (独立) =====
# my_search_r1/search/my_retriever.py
from fastapi import FastAPI
import requests

app = FastAPI()

@app.post("/retrieve")
def my_retrieve(request):
    # 自定义检索逻辑
    results = my_search_function(request.queries)
    return {"result": results}

# ===== 步骤2: 配置检索服务地址 =====
# train_ppo.sh
retriever.url="http://my-service:8000/retrieve"

# ===== 步骤3: 无需修改verl代码 =====
# Search-R1会自动调用新的检索服务
```

### 8.2 如何添加自定义奖励

```python
# ===== 步骤1: 创建自定义奖励函数 =====
# my_search_r1/reward/my_reward.py
def compute_my_reward(solution_str, ground_truth):
    # 自定义奖励逻辑
    return my_score

# ===== 步骤2: 修改RewardManager =====
# verl/trainer/main_ppo.py
class RewardManager():
    def __call__(self, data: DataProto):
        # 使用自定义奖励函数
        score = compute_my_reward(sequences_str, ground_truth)
        reward_tensor[i, valid_response_length - 1] = score
        return reward_tensor

# ===== 步骤3: 无需修改verl框架 =====
# 只要保持DataProto接口不变
```

### 8.3 如何添加新的数据集

```python
# ===== 步骤1: 处理数据 =====
# scripts/data_process/my_dataset.py
def process_fn(example, idx):
    data = {
        "data_source": "my_dataset",
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"ground_truth": answers}
    }
    return data

# ===== 步骤2: 保存parquet =====
dataset.to_parquet('my_data.parquet')

# ===== 步骤3: 配置训练 =====
# train_ppo.sh
data.train_files=$DATA_DIR/my_data.parquet

# ===== 步骤4: 添加奖励函数支持 =====
# verl/trainer/main_ppo.py
def _select_rm_score_fn(data_source):
    if data_source == 'my_dataset':
        return compute_my_reward
    ...
```

---

## 9. 耦合的优势与限制

### 9.1 优势

```
┌─────────────────────────────────────────────────────────────┐
│  使用verl框架的好处                                           │
├─────────────────────────────────────────────────────────────┤
│  1. 分布式训练基础设施                                      │
│     - Ray集群管理                                            │
│     - GPU资源池分配                                          │
│     - 多节点训练支持                                        │
│                                                              │
│  2. 多种并行策略                                            │
│     - FSDP (适合中小模型)                                   │
│     - Megatron (适合大模型)                                 │
│     - 即插即用                                               │
│                                                              │
│  3. 高效推理引擎                                            │
│     - vLLM集成                                              │
│     - 张量并行/流水线并行                                   │
│     - 显存优化                                               │
│                                                              │
│  4. 成熟的RL算法                                            │
│     - PPO完整实现                                           │
│     - GRPO变体                                              │
│     - GAE优势估计                                           │
│                                                              │
│  5. 标准化数据流                                            │
│     - DataProto统一接口                                     │
│     - 易于调试和扩展                                         │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 限制

```
┌─────────────────────────────────────────────────────────────┐
│  verl框架的限制                                               │
├─────────────────────────────────────────────────────────────┤
│  1. 必须遵循DataProto协议                                   │
│     - 数据格式固定                                           │
│     - 扩展需要遵循接口规范                                    │
│                                                              │
│  2. Worker接口限制                                           │
│     - generate_sequences()签名固定                          │
│     - 不能随意修改                                            │
│                                                              │
│  3. 配置系统约束                                             │
│     - 必须使用Hydra配置                                      │
│     - 配置结构需要遵循verl规范                                │
│                                                              │
│  4. 与verl版本绑定                                          │
│     - 需要兼容特定verl版本                                   │
│     - 升级可能需要适配                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. 总结

### 10.1 耦合特点

**Search-R1与verl的耦合是"分层解耦"的设计：**

```
高层 (Search-R1)      │  自定义逻辑 (搜索、推理、工具调用)
中层 (verl框架)        │  通用基础设施 (训练、分布式、并行)
底层 (基础库)          │  Ray、vLLM、PyTorch等
```

### 10.2 关键设计原则

1. **接口标准化**: 通过DataProto实现统一数据交换
2. **模块化设计**: 核心逻辑与框架分离
3. **可扩展性**: 预留自定义接口（奖励、检索、数据集）
4. **向后兼容**: 保持verl核心接口不变

### 10.3 耦合程度评估

| 维度 | 耦合度 | 说明 |
|------|--------|------|
| **代码复用** | 高 | 大量使用verl的Worker、训练器、算法 |
| **接口依赖** | 中 | 必须遵循DataProto和Worker接口 |
| **逻辑自由** | 高 | Search-R1的核心逻辑完全独立 |
| **扩展性** | 高 | 可以灵活添加检索器、奖励函数 |

**结论**: Search-R1与verl的耦合是**合理的分工合作**，verl提供通用RL训练能力，Search-R1专注于搜索增强推理的特定需求。

---

*本文档详细说明了Search-R1与verl框架的耦合关系，希望能帮助理解整个系统的架构设计！*
