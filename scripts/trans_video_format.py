#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple, List

from tqdm import tqdm

# 你这个数据集基本都是 mp4，这里固定只处理 mp4，扫描快很多
VIDEO_GLOB = "*.mp4"


def run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def ffprobe_vcodec(path: Path) -> Optional[str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    rc, out = run(cmd)
    if rc != 0:
        return None
    out = out.strip().splitlines()
    return out[0].strip() if out else None


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def fast_copy(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return "skip_exists"
    ensure_parent(dst)
    shutil.copy2(src, dst)
    return "copied"


def transcode_to_h264(
    src: Path,
    dst: Path,
    crf: int,
    preset: str,
    ffmpeg_threads: int,
    overwrite: bool,
) -> Tuple[bool, str]:
    ensure_parent(dst)
    if dst.exists() and not overwrite:
        return True, "skip_exists"

    ow = "-y" if overwrite else "-n"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        ow,
        "-i",
        str(src),
        "-map",
        "0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-threads",
        str(ffmpeg_threads),

     # 👇 关键就在这一行
        "-an",     # ❗ 禁用音频（没有所谓 cmd_no_audio 命令）
        "-c:s",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    rc, out = run(cmd)
    if rc == 0:
        return True, "ok_no_audio"

    return False, out[-2000:]



def collect_mp4(root: Path) -> List[Path]:
    """
    比 pathlib.rglob('*') + is_file(stat) 快很多，特别是在大目录/NFS 上。
    """
    root_s = str(root)
    results: List[Path] = []

    # 先粗略统计文件数量（只为了让 tqdm 有总数；如果你觉得慢可以删掉这个统计）
    total = 0
    for _, _, files in os.walk(root_s):
        for fn in files:
            if fn.lower().endswith(".mp4"):
                total += 1

    # 再正式收集
    with tqdm(total=total, desc="Scanning .mp4", ncols=100) as pbar:
        for dirpath, _, files in os.walk(root_s):
            for fn in files:
                if fn.lower().endswith(".mp4"):
                    results.append(Path(dirpath) / fn)
                    pbar.update(1)

    return results


def decord_check(output_root: Path, n: int) -> None:
    try:
        from decord import VideoReader
    except Exception as e:
        print(f"[check] decord not available, skip ({e})")
        return

    # 只扫 mp4，别用 rglob('*') + stat
    files = collect_mp4(output_root)
    if not files:
        print("[check] no output videos found")
        return

    n = min(n, len(files))
    ok = 0

    # 随机抽样更合理，但为可复现这里取前 n 个；你想随机我也能改
    for p in files[:n]:
        try:
            vr = VideoReader(str(p))
            _ = len(vr)
            _ = vr[0].asnumpy()
            ok += 1
        except Exception as e:
            print(f"[check][FAIL] {p}: {e}")
    print(f"[check] decord read ok: {ok}/{n}")


def main():
    ap = argparse.ArgumentParser("LeRobot video transcode (AV1 -> H264, fast scan + tqdm)")
    ap.add_argument("--input", required=True, help="Input videos dir, e.g. .../AgiBotWorld-Beta-LeRobot/videos")
    ap.add_argument("--output", required=True, help="Output videos dir, e.g. .../AgiBotWorld-Beta-LeRobot/videos_h264")
    ap.add_argument("--overwrite", action="store_true", default=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--ffmpeg-threads", type=int, default=1)
    ap.add_argument("--crf", type=int, default=25)
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument("--check-decord", action="store_true", default=True)
    ap.add_argument("--check-n", type=int, default=16)
    args = ap.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()

    videos = collect_mp4(in_root)
    tqdm.write(f"Found {len(videos)} .mp4 videos under {in_root}")

    def worker(src: Path):
        rel = src.relative_to(in_root)
        dst = (out_root / rel).with_suffix(".mp4")

        # 已存在且不覆盖就跳过
        if dst.exists() and not args.overwrite:
            return src, True, "skip_exists"

        vcodec = ffprobe_vcodec(src)
        if vcodec is None:
            return src, False, "ffprobe_failed"

        # 非 AV1：直接 copy（保留目录结构）
        if vcodec != "av1":
            try:
                return src, True, fast_copy(src, dst, args.overwrite)
            except Exception as e:
                return src, False, f"copy_failed: {e}"

        ok, msg = transcode_to_h264(
            src,
            dst,
            crf=args.crf,
            preset=args.preset,
            ffmpeg_threads=args.ffmpeg_threads,
            overwrite=args.overwrite,
        )
        return src, ok, msg

    ok_cnt, fail_cnt = 0, 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(worker, v) for v in videos]

        with tqdm(total=len(futs), desc="Transcoding", ncols=100) as pbar:
            for fut in as_completed(futs):
                src, ok, msg = fut.result()
                tag = "OK" if ok else "FAIL"
                tqdm.write(f"{tag} {src.relative_to(in_root)} :: {msg}")
                ok_cnt += int(ok)
                fail_cnt += int(not ok)
                pbar.update(1)

    print(f"Done. ok={ok_cnt}, fail={fail_cnt}")
    print(f"Output root: {out_root}")

    if args.check_decord:
        decord_check(out_root, args.check_n)


if __name__ == "__main__":
    main()
