#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 JSON 文件并打印内容")
    parser.add_argument(
        "--json_path",
        type=str,
        default="/mnt/workspace/users/daiyixiang/JoyRA-RL/outputs_value_robotwin_plus_offline_with_neg/advantages_all_train_labeled.json",
        help="JSON 文件路径",
    )
    parser.add_argument("--max_chars", type=int, default=5000, help="最多打印多少字符")
    args = parser.parse_args()

    p = Path(args.json_path)
    if not p.exists():
        print(f"[Error] 文件不存在: {p}")
        return

    text = p.read_text(encoding="utf-8")
    print(f"[Info] 路径: {p}")
    print(f"[Info] 文件大小(字节): {p.stat().st_size}")
    print(f"[Info] 文本长度(字符): {len(text)}")

    if len(text.strip()) == 0:
        print("[Info] 文件内容为空。")
        return

    print("\n===== 原始内容预览 =====")
    preview = text[: max(0, args.max_chars)]
    print(preview)
    if len(text) > len(preview):
        print(f"\n... (已截断, 总长度={len(text)} 字符)")

    print("\n===== JSON 解析结果 =====")
    try:
        obj = json.loads(text)
    except Exception as e:
        print(f"[Error] 不是合法 JSON: {e}")
        return

    print(f"[Info] 顶层类型: {type(obj).__name__}")
    if isinstance(obj, dict):
        print(f"[Info] 顶层 keys: {list(obj.keys())[:30]}")
    elif isinstance(obj, list):
        print(f"[Info] 顶层长度: {len(obj)}")
        if obj:
            print(f"[Info] 第一个元素类型: {type(obj[0]).__name__}")
            print(f"[Info] 第一个元素: {obj[0]}")


if __name__ == "__main__":
    main()
