#!/usr/bin/env python3
"""
Search-R1 GPU环境验证脚本

功能:
- 验证CUDA和PyTorch安装
- 检查GPU可用性和显存容量
- 验证vLLM安装

用法:
    python scripts/verify_gpu.py
"""

import sys


def verify_environment():
    """验证训练环境"""
    print("=" * 60)
    print("Search-R1 GPU环境验证")
    print("=" * 60)
    print()

    try:
        # 检查PyTorch
        print("📦 检查PyTorch...")
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")

        if not torch.cuda.is_available():
            print("❌ CUDA不可用，请检查PyTorch安装")
            return False

        print(f"   CUDA版本: {torch.version.cuda}")
        print()

        # 检查GPU
        print("🎮 检查GPU...")
        num_gpus = torch.cuda.device_count()
        print(f"   GPU数量: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name}")
            print(f"   显存: {gpu_memory:.1f} GB")

            # 检查显存是否足够
            if gpu_memory < 40:
                print(f"   ⚠️  警告: 显存可能不足40GB")

        print()

        # 检查vLLM
        print("🚀 检查vLLM...")
        try:
            import vllm
            print(f"   vLLM版本: {vllm.__version__}")
            print("   ✅ vLLM已正确安装")
        except ImportError:
            print("   ❌ vLLM未安装")
            print("   请运行: pip install vllm==0.6.3")
            return False

        print()

        # 检查其他依赖
        print("📚 检查其他依赖...")
        dependencies = [
            ('transformers', 'transformers'),
            ('datasets', 'datasets'),
            ('ray', 'ray'),
            ('hydra', 'hydra.core'),
            ('wandb', 'wandb'),
            ('accelerate', 'accelerate'),
        ]

        missing = []
        for package_name, import_name in dependencies:
            try:
                __import__(import_name)
                print(f"   ✅ {package_name}")
            except ImportError:
                print(f"   ❌ {package_name}")
                missing.append(package_name)

        print()

        if missing:
            print(f"❌ 缺少依赖: {', '.join(missing)}")
            print("请安装缺失的包")
            return False

        print("=" * 60)
        print("✅ 环境验证完成！所有依赖已正确安装")
        print("=" * 60)
        print()

        return True

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    success = verify_environment()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
