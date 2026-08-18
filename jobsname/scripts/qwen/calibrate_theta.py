#!/usr/bin/env python3
"""Qwen3-Embedding-4B θ1 校准：拿 530701 电子商务算 job-seed 余弦分布。

用法:
    source ~/workspace/venv/bin/activate
    python calibrate_theta.py --range 2026(01-06)
    python calibrate_theta.py --range 2026(01-06) --major 530701
"""

import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np

MODELS_DIR   = Path("/home/cl-715/workspace/models")
SEEDS_ROOT   = MODELS_DIR                          # 种子: models/典型岗位名称_gt100_{range}.jsonl
MAJORS_DATA_ROOT = MODELS_DIR / "majors_data"
EMBED_DIR    = MODELS_DIR / "Qwen3-Embedding-4B"
EMBED_BATCH  = 128

CALIB_MAJOR = "530701"   # 默认校准专业（电子商务，样本大、种子典型）


def load_seeds(seeds_file):
    seeds = {}
    with open(seeds_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            seeds[r["专业代码"]] = {"major_name": r["专业名称"], "seeds": r["典型岗位名称"]}
    return seeds


def main():
    ap = argparse.ArgumentParser(description="θ1 校准（Qwen3-Embedding-4B，按月份范围）")
    ap.add_argument("--range", required=True,
                    help="月份范围标识，如 2026(01-06)；种子默认 典型岗位名称_gt100_{range}.jsonl")
    ap.add_argument("--major", default=CALIB_MAJOR, help=f"校准专业代码（默认 {CALIB_MAJOR}）")
    ap.add_argument("--seeds", help="种子文件路径；默认 models/典型岗位名称_gt100_{range}.jsonl")
    args = ap.parse_args()

    rng = args.range
    calib_major = args.major

    # 种子文件
    seeds_file = Path(args.seeds) if args.seeds else SEEDS_ROOT / f"典型岗位名称_gt100_{rng}.jsonl"
    if not seeds_file.exists():
        print(f"[error] 种子文件不存在: {seeds_file}")
        print(f"        可用 --seeds 指定，或先构建种子（见 jobsname/种子构建说明.md）")
        return

    # 数据目录：majors_data/{range}/majors_data_dwd_all/
    dwd_dir = MAJORS_DATA_ROOT / rng / "majors_data_dwd_all"
    if not dwd_dir.exists():
        print(f"[error] dwd 目录不存在: {dwd_dir}")
        print(f"        请先跑 Stage 3 (build_by_major.py --output-range {rng})")
        return

    seeds_data = load_seeds(seeds_file)
    if calib_major not in seeds_data:
        print(f"[error] 校准专业 {calib_major} 不在种子文件里")
        print(f"        种子里有的专业: {sorted(seeds_data.keys())[:5]}...")
        return

    info = seeds_data[calib_major]
    major_name = info["major_name"]
    print(f"校准专业: {calib_major} {major_name}")
    print(f"种子文件: {seeds_file}")
    print(f"数据目录: {dwd_dir}")

    # 找数据文件
    data_file = dwd_dir / f"{calib_major}_{major_name}.jsonl"
    if not data_file.exists():
        for f in dwd_dir.glob(f"{calib_major}_*.jsonl"):
            data_file = f
            break
    if not data_file.exists():
        print(f"[error] 专业数据文件不存在: {dwd_dir}/{calib_major}_*.jsonl")
        return

    # 读数据
    names = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                n = (r.get("原始岗位名称") or "").strip()
                if n: names.append(n)
            except: continue

    name_freq = Counter(names)
    unique = list(name_freq.keys())
    seeds = info["seeds"]
    print(f"  岗位: {len(names):,} 条, 去重 {len(unique):,} 个, seed {len(seeds)} 个")

    # Qwen 编码
    import torch
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(str(EMBED_DIR), device="cuda" if torch.cuda.is_available() else "cpu",
                                   model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {})

    print(f"编码 {len(seeds)+len(unique)} 条...", end=" ", flush=True)
    t0 = time.time()
    all_texts = seeds + unique
    emb = embedder.encode(all_texts, normalize_embeddings=True, batch_size=EMBED_BATCH,
                          show_progress_bar=False, truncate_dim=1024)  # Qwen MRL 缩维
    print(f"{time.time()-t0:.0f}s")

    seed_emb = emb[:len(seeds)]
    name_emb = emb[len(seeds):]
    sim = name_emb @ seed_emb.T
    max_sim = sim.max(axis=1)

    # 分桶
    buckets = [(0.95,1),(0.90,0.95),(0.85,0.90),(0.80,0.85),(0.75,0.80),
               (0.70,0.75),(0.65,0.70),(0.60,0.65),(0.55,0.60),(0.50,0.55),(0.40,0.50),(0.00,0.40)]

    print(f"\n{'='*60}\n  job → 最近 seed 余弦分布\n{'='*60}")
    print(f"  {'区间':12s} {'job数':>8s} {'占比':>7s}")
    for lo, hi in buckets:
        n = int(((max_sim >= lo) & (max_sim < hi)).sum())
        pct = n/len(unique)*100
        bar = "█"*int(pct/2) + "░"*(50-int(pct/2))
        print(f"  [{lo:.2f}-{hi:.2f})  {n:>8,}  {pct:>6.1f}%  {bar}")

    print(f"\n  中位: {float(np.median(max_sim)):.3f}  p25: {float(np.percentile(max_sim,25)):.3f}  p75: {float(np.percentile(max_sim,75)):.3f}")
    print(f"\n各 θ 覆盖:")
    for lo, _ in buckets:
        n = int((max_sim >= lo).sum())
        print(f"  θ={lo:.2f}  seed桶: {n:>6,} ({n/len(unique)*100:5.1f}%)  其他: {len(unique)-n:>6,} ({(len(unique)-n)/len(unique)*100:5.1f}%)")


if __name__ == "__main__":
    main()
