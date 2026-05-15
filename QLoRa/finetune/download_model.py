"""
模型下载脚本 — AutoDL / 云 GPU 专用
=====================================
用法：
    # 下载 TinyLlama (1.1B, ~2GB) — 快速验证用
    python download_model.py

    # 下载指定模型到指定路径
    python download_model.py --model Qwen/Qwen2.5-7B-Instruct --dir /root/autodl-tmp/Qwen2.5-7B-Instruct

    # 下载 Qwen2.5-3B (中等)
    python download_model.py --model Qwen/Qwen2.5-3B-Instruct

AutoDL 路径说明:
    系统盘 /root/   → 30GB, 按量计费, 实例释放后**数据丢失**
    数据盘 /root/autodl-tmp/ → 免费 50GB, 实例**关机不释放** (可用 conda 扩容)
    模型应下载到数据盘: /root/autodl-tmp/
"""

import os
import argparse

# 使用 HF 镜像站，国内下载更快
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download


# ============================================================
# 预设模型列表（根据你的 GPU 配置推荐）
# ============================================================
MODEL_PRESETS = {
    # 小模型 — 快速验证流程 (2GB，6GB 显存就能跑)
    "tiny": {
        "repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "default_dir": "./TinyLlama-1.1B-Chat-v1.0",
        "size": "~2GB",
    },
    # 中等 — 性价比最高 (6GB，适合做最终方案)
    "3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "default_dir": "/root/autodl-tmp/Qwen2.5-3B-Instruct",
        "size": "~6GB",
    },
    # 7B — 最强推理能力 (15GB，你的 4090D 24GB 跑 QLoRA 绰绰有余)
    "7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "default_dir": "/root/autodl-tmp/Qwen2.5-7B-Instruct",
        "size": "~15GB",
    },
}


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace 模型到本地")
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace 模型 repo_id，如 Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="下载目标路径（绝对路径）, 推荐 /root/autodl-tmp/<模型名>"
    )
    parser.add_argument(
        "--preset", type=str, default=None, choices=list(MODEL_PRESETS.keys()),
        help=f"使用预设: {list(MODEL_PRESETS.keys())}"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有预设模型"
    )
    args = parser.parse_args()

    if args.list:
        print("预设模型列表:")
        for name, info in MODEL_PRESETS.items():
            print(f"  --preset {name:6s}  {info['repo']:50s}  {info['size']:>6s}")
        return

    # 确定下载目标
    if args.preset:
        preset = MODEL_PRESETS[args.preset]
        repo_id = args.model or preset["repo"]
        local_dir = args.dir or preset["default_dir"]
    elif args.model:
        repo_id = args.model
        local_dir = args.dir or f"/root/autodl-tmp/{repo_id.split('/')[-1]}"
    else:
        # 默认行为：下载 TinyLlama 用于快速验证
        preset = MODEL_PRESETS["tiny"]
        repo_id = preset["repo"]
        local_dir = args.dir or preset["default_dir"]

    print(f"模型: {repo_id}")
    print(f"目标路径: {local_dir}")
    print()

    # 检查磁盘空间（目标目录所在的磁盘）
    parent_dir = os.path.dirname(os.path.abspath(local_dir))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # 下载
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,          # 断点续传
        max_workers=4,                 # 4 线程并行下载
    )

    print(f"\n下载完成! 模型路径: {os.path.abspath(local_dir)}")
    print(f"\n在 notebook 中设置 MODEL_PATH = \"{os.path.abspath(local_dir)}\"")


if __name__ == "__main__":
    main()
