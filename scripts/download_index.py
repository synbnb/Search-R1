#!/usr/bin/env python3
"""
Search-R1 检索索引下载脚本

功能:
- 从HuggingFace Hub下载预构建的E5检索索引
- 下载Wikipedia-18语料库
- 自动解压和合并索引文件

用法:
    python scripts/download_index.py --output_dir data/index
"""

import argparse
import gzip
import os
import shutil
from typing import Dict, Optional

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ 缺少依赖: huggingface_hub")
    print("请运行: pip install huggingface_hub")
    exit(1)


class IndexDownloader:
    """检索索引下载器"""

    # E5-Flat索引配置
    E5_INDEX_CONFIG = {
        'repo_id': 'PeterJinGo/wiki-18-e5-index',
        'files': ['part_aa'],
        'output_name': 'e5_Flat.index'
    }

    # Wikipedia语料库配置
    WIKI_CORPUS_CONFIG = {
        'repo_id': 'PeterJinGo/wiki-18-corpus',
        'filename': 'wiki-18.jsonl.gz',
        'output_name': 'wiki-18.jsonl'
    }

    def __init__(self, output_dir: str):
        """
        初始化下载器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_index(self) -> str:
        """
        下载E5-Flat索引

        Returns:
            索引文件路径
        """
        print("📥 下载E5-Flat索引...")

        config = self.E5_INDEX_CONFIG
        output_path = os.path.join(self.output_dir, config['output_name'])

        # 合并所有索引部分
        with open(output_path, 'wb') as f_out:
            for file_name in config['files']:
                print(f"   下载: {file_name}")
                downloaded_path = hf_hub_download(
                    repo_id=config['repo_id'],
                    filename=file_name,
                    repo_type='dataset'
                )

                # 复制到输出文件
                with open(downloaded_path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)

                # 清理临时文件
                os.remove(downloaded_path)

        print(f"✅ 索引下载完成: {output_path}")
        return output_path

    def download_corpus(self) -> str:
        """
        下载Wikipedia语料库

        Returns:
            语料库文件路径
        """
        print("📥 下载Wikipedia-18语料库...")

        config = self.WIKI_CORPUS_CONFIG
        gz_path = os.path.join(self.output_dir, config['filename'])
        output_path = os.path.join(self.output_dir, config['output_name'])

        # 下载压缩文件
        print(f"   下载: {config['filename']}")
        hf_hub_download(
            repo_id=config['repo_id'],
            filename=config['filename'],
            repo_type='dataset',
            local_dir=self.output_dir
        )

        print(f"   解压: {config['filename']} -> {config['output_name']}")

        # 解压文件
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 清理压缩文件
        os.remove(gz_path)

        print(f"✅ 语料库下载完成: {output_path}")
        return output_path

    def download_all(self) -> Dict[str, str]:
        """
        下载所有必需文件

        Returns:
            文件路径字典
        """
        print("=" * 60)
        print("Search-R1 检索索引下载")
        print("=" * 60)
        print()

        try:
            # 下载索引
            index_path = self.download_index()
            print()

            # 下载语料库
            corpus_path = self.download_corpus()
            print()

            # 返回文件路径
            result = {
                'index': index_path,
                'corpus': corpus_path,
                'output_dir': self.output_dir
            }

            print("=" * 60)
            print("✅ 所有文件下载完成！")
            print("=" * 60)
            print(f"📁 索引文件: {result['index']}")
            print(f"📁 语料文件: {result['corpus']}")
            print()

            return result

        except Exception as e:
            print(f"❌ 下载失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Search-R1检索索引下载脚本')
    parser.add_argument('--output_dir', type=str, default='data/index',
                        help='输出目录路径 (默认: data/index)')

    args = parser.parse_args()

    try:
        # 创建下载器并下载
        downloader = IndexDownloader(args.output_dir)
        downloader.download_all()

        return 0

    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
