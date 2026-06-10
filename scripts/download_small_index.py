#!/usr/bin/env python3
"""
Search-R1 小型检索索引下载脚本
支持BM25等小型索引，下载快速
"""

import argparse
import os
import subprocess
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ 缺少依赖: huggingface_hub")
    print("请运行: pip install huggingface_hub")
    exit(1)


def download_bm25_index(output_dir: str = "data/index") -> str:
    """
    下载BM25稀疏检索索引（小而快）

    Returns:
        索引文件路径
    """
    print("📥 下载BM25稀疏索引...")
    print("   大小: ~1GB (vs E5-Flat 40GB)")
    print("   速度: 快速检索")
    print("   精度: 中等（适合快速验证）")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # BM25索引配置
    bm25_repo = "PeterJinGo/wiki-18-bm25-index"
    index_file = "bm25_index.tar.gz"

    try:
        print(f"   下载: {index_file}")
        downloaded_path = hf_hub_download(
            repo_id=bm25_repo,
            filename=index_file,
            repo_type="dataset",
            local_dir=output_dir
        )

        print(f"   解压索引...")
        # 解压tar.gz文件
        subprocess.run(
            ["tar", "-xzf", downloaded_path, "-C", output_dir],
            check=True
        )

        # 清理压缩文件
        os.remove(downloaded_path)

        print(f"✅ BM25索引下载完成")
        return os.path.join(output_dir, "bm25_index")

    except Exception as e:
        print(f"❌ BM25索引下载失败: {e}")
        return None


def download_online_retriever_config(output_dir: str = "data/index") -> str:
    """
    生成在线搜索引擎配置文件（无需下载索引）

    Returns:
        配置文件路径
    """
    print("🌐 配置在线搜索引擎...")
    print("   大小: 0GB (无需下载索引)")
    print("   速度: 依赖网络")
    print("   精度: 高（使用实时搜索）")
    print()

    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "online_retriever.json")
    config = {
        "retriever_type": "online",
        "search_engine": "bing",  # 或 google, duckduckgo
        "max_results": 3,
        "timeout": 10
    }

    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ 在线检索配置完成: {config_path}")
    return config_path


def download_wiki_corpus(output_dir: str = "data/index") -> str:
    """
    下载Wikipedia语料库（共享资源）

    Returns:
        语料库文件路径
    """
    print("📥 下载Wikipedia语料库...")
    print("   大小: ~14GB")

    os.makedirs(output_dir, exist_ok=True)

    corpus_repo = "PeterJinGo/wiki-18-corpus"
    corpus_file = "wiki-18.jsonl"

    try:
        print(f"   下载: {corpus_file}")
        downloaded_path = hf_hub_download(
            repo_id=corpus_repo,
            filename=corpus_file,
            repo_type="dataset",
            local_dir=output_dir
        )

        print(f"✅ 语料库下载完成: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        print(f"❌ 语料库下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Search-R1小型索引下载脚本')
    parser.add_argument('--output_dir', type=str, default='data/index',
                        help='输出目录路径 (默认: data/index)')
    parser.add_argument('--type', type=str, default='bm25',
                        choices=['bm25', 'online', 'corpus'],
                        help='索引类型: bm25(~1GB), online(0GB), corpus(14GB)')

    args = parser.parse_args()

    print("=" * 60)
    print("Search-R1 小型检索索引下载")
    print("=" * 60)
    print()

    try:
        if args.type == 'bm25':
            result = download_bm25_index(args.output_dir)
        elif args.type == 'online':
            result = download_online_retriever_config(args.output_dir)
        elif args.type == 'corpus':
            result = download_wiki_corpus(args.output_dir)
        else:
            print(f"❌ 未知的索引类型: {args.type}")
            return 1

        if result:
            print()
            print("=" * 60)
            print("✅ 下载完成！")
            print("=" * 60)
            print(f"📁 文件路径: {result}")
            print()
            print("💡 使用提示:")
            print("   修改检索服务器启动命令，指定新索引路径")
            if args.type == 'bm25':
                print("   --index_path {result}/bm25_index".format(result=result))
            elif args.type == 'online':
                print("   --retriever_name online")
            print()

            return 0
        else:
            return 1

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
