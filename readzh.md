# Search-R1: 使用强化学习训练你的大语言模型进行推理和调用搜索引擎

<div align="center">
  <img src="https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/public/logo.png" alt="logo" width="300"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/Paper1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/Paper2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/Resources-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/Tweet-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/Logs-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>


**Search-R1** 是一个强化学习框架，专为训练**推理与搜索交织的大语言模型**而设计——这些语言模型能够以协调的方式进行推理和工具调用（例如，调用搜索引擎）。

该框架构建于 [veRL](https://github.com/volcengine/verl) 之上，Search-R1 通过引入交织的搜索引擎访问能力扩展了 **DeepSeek-R1(-Zero)** 的理念，并提供了一个完全开源的强化学习训练流水线。它是 **OpenAI DeepResearch** 的开源替代方案，推动了工具增强型大语言模型推理的研究与发展。

我们支持不同的强化学习方法（例如，PPO、GRPO、reinforce），不同的大语言模型（例如，llama3、Qwen2.5 等）以及不同的搜索引擎（例如，本地稀疏/密集检索器和在线搜索引擎）。

论文：[链接1](https://arxiv.org/pdf/2503.09516)、[链接2](https://arxiv.org/abs/2505.15117)；模型和数据：[链接](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5)；Twitter 讨论：[链接](https://x.com/BowenJin13/status/1895544294473109889)；完整实验日志：[初步](https://wandb.ai/peterjin/Search-R1-open)、[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)、[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)。关于这些日志和方法的详细信息可以在[这里](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md)找到。


![single-turn](public/main.png)

## 新闻

- [2025.10] Search-R1 被 Thinking Machines Lab 的首个产品 [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook) 采用！详情：[文档](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/tool_use/search)
- [2025.7] Search-R1 获得 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 支持！详细说明：[代码](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/search)、[文档](https://novasky-ai.notion.site/skyrl-searchr1)
- [2025.6] Search-R1 现已集成到最新版本的 veRL 中，可以利用其最新功能！详细说明：[veRL](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)、[英文文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md)、[中文文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md)
- [2025.5] 第二篇进行详细实证研究的[论文](https://arxiv.org/abs/2505.15117)已发布，附带日志：[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)
- [2025.4] 我们支持 30B+ 大语言模型的[多节点](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/multinode.md)训练！
- [2025.4] 我们支持[不同的搜索引擎](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)，包括本地稀疏检索器、带有 ANN 索引的本地密集检索器和在线搜索引擎！
- [2025.3] 第一篇 Search-R1 [论文](https://arxiv.org/pdf/2503.09516)已发布，附带日志：[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)
- [2025.2] 我们开源了 Search-R1 代码库，附带[初步结果](https://wandb.ai/peterjin/Search-R1-open)

## 链接

- [安装](#installation)
- [快速开始](#quick-start)
- [初步结果](#preliminary-results)
- [推理](#inference)
- [使用你自己的数据集](#use-your-own-dataset)
- [使用你自己的搜索引擎](#use-your-own-search-engine)
- [功能特性](#features)
- [致谢](#acknowledge)
- [引用](#citations)

## 安装

### Search-r1 环境
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# 安装 torch [或者你可以跳过这一步，让 vllm 为你安装正确的版本]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 安装 vllm
pip3 install vllm==0.6.3 # 或者你可以安装 0.5.4、0.4.2 和 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### 检索器环境（可选）
如果你想使用本地检索器作为搜索引擎，可以按照以下方式安装环境。（我们建议使用单独的环境。）
```bash
conda create -n retriever python=3.10
conda activate retriever

# 我们建议使用 conda 安装 torch 以使用 faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## 安装 gpu 版本的 faiss 以保证高效的 RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API 函数
pip install uvicorn fastapi
```


## 快速开始

在 NQ 数据集上训练一个推理+搜索的大语言模型，使用 e5 作为检索器，维基百科作为语料库。

(1) 下载索引和语料库。
```bash
save_path=/保存/的/路径
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 处理 NQ 数据集。
```bash
python scripts/data_process/nq_search.py
```

(3) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) 运行 RL 训练（PPO），使用 Llama-3.2-3b-base。
```bash
conda activate searchr1
bash train_ppo.sh
```

## 初步结果

(1) 基础模型（llama3.2-3b-base）学会了调用搜索引擎并获得了性能提升。

![llama-3b](public/llama32-3b.png)


(2) 基础模型（Qwen2.5-7b-base）能够通过强化学习学会进行多轮搜索引擎调用和推理。

![multi-turn](public/multi-turn.png)

## 推理
#### 你可以使用训练好的 Search-R1 模型来回答你自己的问题。
(1) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) 运行推理。
```bash
conda activate searchr1
python infer.py
```
你可以修改第 7 行的 ```question``` 为你感兴趣的问题。

## 使用你自己的数据集

### 问答数据
对于每个问答样本，它应该是一个包含以下所需内容的字典：

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

你可以参考 ```scripts/data_process/nq_search.py``` 中的具体数据处理示例。

### 语料库

建议将语料库制作成 jsonl 文件，其中每一行（一个包含 "id" 键和 "contents" 键的字典）对应一个段落。你可以参考 ```example/corpus.jsonl``` 中的示例。

"id" 键对应段落 id，"contents" 键对应段落内容（'"' + 标题 + '"\n' + 文本）。
例如：
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 – July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**为你的语料库建立索引（可选）。**
如果你想使用本地检索器作为搜索引擎，可以通过以下方式为你的语料库建立索引：
```
bash search_r1/search/build_index.sh
```
你可以将 ```retriever_name``` 和 ```retriever_model``` 更改为你感兴趣的开箱即用的检索器。

## 使用你自己的搜索引擎

我们的代码库支持本地稀疏检索器（例如，BM25）、本地密集检索器（支持 GPU 的平面索引和 CPU 的 ANN 索引）以及在线搜索引擎（例如，Google、Bing 等）。更多详细信息可以在[这里](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md)找到。

主要理念是将本地或远程搜索引擎服务器与主 RL 训练流水线分开启动。

大语言模型可以通过调用搜索 API（例如，"http://127.0.0.1:8000/retrieve"）来调用搜索引擎。

你可以参考 ```search_r1/search/retriever_server.py``` 中启动本地检索器服务器的示例。

## 功能特性
- 支持本地稀疏检索器（例如，BM25）。 ✔️
- 支持本地密集检索器（平面索引和 ANN 索引） ✔️
- 支持 Google 搜索 / Bing 搜索 / Brave 搜索 API 等。 ✔️
- 支持开箱即用的神经重排序模型。 ✔️
- 支持不同的 RL 方法（例如，PPO、GRPO、reinforce）。 ✔️
- 支持不同的大语言模型（例如，llama3、Qwen2.5 等）。 ✔️

## 致谢

Search-R1 的概念受到 [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 和 [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main) 的启发。
其实现构建于 [veRL](https://github.com/volcengine/verl) 和 [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main) 之上。
我们真诚感谢这些团队为开源研究和开发所做的贡献。

## 由 Search-R1 驱动或受其启发的优秀工作

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)：通过真实环境中的强化学习扩展深度研究。[![[code]](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)
- [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)：激励大视觉模型进行搜索。[![[code]](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [OTC](https://arxiv.org/pdf/2504.14870)：通过强化学习实现最优工具调用。
- [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch)：在不进行搜索的情况下激励大语言模型的搜索能力。[![[code]](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)
- [IKEA](https://github.com/hzy312/knowledge-r1)：用于高效自适应搜索智能体的强化内外知识协同推理。[![[code]](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)
- [Scent of Knowledge](https://arxiv.org/abs/2505.09316)：通过信息觅食优化搜索增强推理。
- [AutoRefine](https://www.arxiv.org/pdf/2505.11277)：思考过程中的搜索与优化。[![[code]](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)
- [O^2-Searcher](https://arxiv.org/pdf/2505.16582)：基于搜索的开放域开放式问答智能体模型。[![[code]](https://img.shields.io/github/stars/Acade-Mate/O2-Searcher)](https://github.com/Acade-Mate/O2-Searcher)
- [MaskSearch](https://arxiv.org/pdf/2505.20285)：增强智能体搜索能力的通用预训练框架。[![[code]](https://img.shields.io/github/stars/Alibaba-NLP/MaskSearch)](https://github.com/Alibaba-NLP/MaskSearch)
- [VRAG-RL](https://arxiv.org/abs/2505.22019)：基于视觉感知的富视觉信息理解 RAG。[![[code]](https://img.shields.io/github/stars/Alibaba-NLP/VRAG)](https://github.com/Alibaba-NLP/VRAG)
- [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668)：通过 SFT 和 RL 训练大语言模型用代码进行推理。[![[code]](https://img.shields.io/github/stars/yongchao98/R1-Code-Interpreter)](https://github.com/yongchao98/R1-Code-Interpreter)
- [R-Search](https://arxiv.org/abs/2506.04185)：通过多奖励强化学习增强大语言模型推理。[![[code]](https://img.shields.io/github/stars/QingFei1/R-Search)](https://github.com/QingFei1/R-Search)
- [StepSearch](https://arxiv.org/pdf/2505.15107)：通过逐步近端策略优化点燃大语言模型的搜索能力。[![[code]](https://img.shields.io/github/stars/Zillwang/StepSearch)](https://github.com/Zillwang/StepSearch)
- [SimpleTIR](https://simpletir.notion.site/report)：多轮工具集成推理的稳定端到端强化学习。[![[code]](https://img.shields.io/github/stars/ltzheng/SimpleTIR)](https://github.com/ltzheng/SimpleTIR)
- [Router-R1](https://arxiv.org/pdf/2506.09033)：通过强化学习教授大语言模型多轮路由和聚合。[![[code]](https://img.shields.io/github/stars/ulab-uiuc/Router-R1)](https://github.com/ulab-uiuc/Router-R1)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/)：用于大语言模型的模块化全栈 RL 库。[![[code]](https://img.shields.io/github/stars/NovaSky-AI/SkyRL)](https://github.com/NovaSky-AI/SkyRL)
- [ASearcher](https://arxiv.org/abs/2508.07976)：搜索智能体的大规模 RL。[![[code]](https://img.shields.io/github/stars/inclusionAI/ASearcher)](https://github.com/inclusionAI/ASearcher)
- [ParallelSearch](https://www.arxiv.org/abs/2508.09303)：通过 RL 分解查询并并行搜索子查询。[![[code]](https://img.shields.io/github/stars/Tree-Shu-Zhao/ParallelSearch)](https://github.com/Tree-Shu-Zhao/ParallelSearch)
- [AutoTIR](https://arxiv.org/pdf/2507.21836)：通过强化学习实现自主工具集成推理。[![[code]](https://img.shields.io/github/stars/weiyifan1023/AutoTIR)](https://github.com/weiyifan1023/AutoTIR)
- [verl-tool](https://arxiv.org/pdf/2509.01055)：支持多样化工具使用的 verl 版本。[![[code]](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)](https://github.com/TIGER-AI-Lab/verl-tool)
- [Tree-GRPO](https://arxiv.org/abs/2509.21240)：大语言模型智能体强化学习的树搜索。[![[code]](https://img.shields.io/github/stars/AMAP-ML/Tree-GRPO)](https://github.com/AMAP-ML/Tree-GRPO)
- [EviNote-RAG](https://arxiv.org/abs/2509.00877)：通过答案支持性证据笔记增强 RAG 模型。[![[code]](https://img.shields.io/github/stars/Da1yuqin/EviNoteRAG)](https://github.com/Da1yuqin/EviNoteRAG)
- [GlobalRAG](https://arxiv.org/pdf/2510.20548v1)：GlobalRAG：通过强化学习增强多跳问答中的全局推理。[![[code]](https://img.shields.io/github/stars/CarnegieBin/GlobalRAG)](https://github.com/CarnegieBin/GlobalRAG)




## 引用

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```
