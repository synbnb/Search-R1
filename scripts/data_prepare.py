#!/usr/bin/env python3
"""
Search-R1 数据准备脚本

功能:
- 从FlashRAG数据集加载NQ (Natural Questions) 数据
- 转换为Search-R1训练所需的格式
- 保存为JSON Lines格式供训练使用

用法:
    python scripts/data_prepare.py --output_dir data/nq_search
"""

import argparse
import json
import os
from datasets import load_dataset
from typing import Dict, Any


def make_search_r1_prompt(question: str) -> str:
    """
    构造Search-R1训练所需的prompt格式

    Args:
        question: 原始问题

    Returns:
        格式化后的prompt字符串
    """
    # 确保问题以问号结尾
    if not question.endswith('?'):
        question += '?'

    prompt_template = (
        f'Answer the given question. '
        f'You must conduct reasoning inside <think and  first every time you get new information. '
        f'After reasoning, if you find you lack some knowledge, you can call a search engine by '
        f'<search> query </search> and it will return the top searched results between <information> '
        f'and </information>. You can search as many times as your want. If you find no further '
        f'external knowledge needed, you can directly provide the answer inside <answer> and '
        f'</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. '
        f'Question: {question}'
    )

    return prompt_template


def process_nq_sample(example: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    """
    处理单个NQ样本，转换为Search-R1格式

    Args:
        example: 单个数据样本
        idx: 样本索引
        split: 数据集划分 ('train' 或 'test')

    Returns:
        处理后的数据样本
    """
    # 清理问题文本
    question = example['question'].strip()

    # 构造Search-R1格式的prompt
    formatted_prompt = make_search_r1_prompt(question)

    # 构造最终数据结构
    processed_data = {
        'data_source': 'nq',
        'prompt': [{'role': 'user', 'content': formatted_prompt}],
        'ability': 'fact-reasoning',
        'reward_model': {
            'style': 'rule',
            'ground_truth': {'target': example['golden_answers']}
        },
        'extra_info': {
            'split': split,
            'index': idx,
            'original_question': question
        }
    }

    return processed_data


def prepare_nq_dataset(output_dir: str, dataset_name: str = 'RUC-NLPIR/FlashRAG_datasets',
                       subset: str = 'nq') -> Dict[str, int]:
    """
    准备完整的NQ数据集

    Args:
        output_dir: 输出目录路径
        dataset_name: HuggingFace数据集名称
        subset: 数据集子集名称

    Returns:
        数据集统计信息
    """
    print(f"📥 加载数据集: {dataset_name} ({subset})")

    # 加载完整数据集
    dataset = load_dataset(dataset_name, subset)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    print(f"✅ 训练集样本数: {len(train_dataset)}")
    print(f"✅ 测试集样本数: {len(test_dataset)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理训练集
    print("🔄 处理训练集...")
    train_dataset = train_dataset.map(
        lambda example, idx: process_nq_sample(example, idx, 'train'),
        with_indices=True,
        remove_columns=train_dataset.column_names
    )

    # 保存训练集为JSON Lines格式
    train_output_path = os.path.join(output_dir, 'train.json')
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"💾 训练集已保存: {train_output_path}")

    # 处理测试集
    print("🔄 处理测试集...")
    test_dataset = test_dataset.map(
        lambda example, idx: process_nq_sample(example, idx, 'test'),
        with_indices=True,
        remove_columns=test_dataset.column_names
    )

    # 保存测试集为JSON Lines格式
    test_output_path = os.path.join(output_dir, 'test.json')
    with open(test_output_path, 'w', encoding='utf-8') as f:
        for item in test_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"💾 测试集已保存: {test_output_path}")

    # 返回统计信息
    stats = {
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'train_path': train_output_path,
        'test_path': test_output_path
    }

    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Search-R1数据准备脚本')
    parser.add_argument('--output_dir', type=str, default='data/nq_search',
                        help='输出目录路径 (默认: data/nq_search)')
    parser.add_argument('--dataset_name', type=str, default='RUC-NLPIR/FlashRAG_datasets',
                        help='HuggingFace数据集名称')
    parser.add_argument('--subset', type=str, default='nq',
                        help='数据集子集名称')

    args = parser.parse_args()

    print("=" * 60)
    print("Search-R1 数据准备脚本")
    print("=" * 60)
    print()

    try:
        # 准备数据集
        stats = prepare_nq_dataset(
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            subset=args.subset
        )

        print()
        print("=" * 60)
        print("✅ 数据准备完成！")
        print("=" * 60)
        print(f"📊 训练集: {stats['train_samples']} 样本")
        print(f"📊 测试集: {stats['test_samples']} 样本")
        print(f"📁 训练集路径: {stats['train_path']}")
        print(f"📁 测试集路径: {stats['test_path']}")
        print()

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
