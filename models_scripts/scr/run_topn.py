#!/usr/bin/env python3
"""
TopN: 从缓存向量产出 top20 带分候选（eval 基准）+ margin 截断（生产）
======================================================================
- 读 run_embed.py 落盘的 vectors.npy（不重跑模型），与 744 专业索引重搜，
  输出每岗位 top20 (code, name, cosine原始分)。
- eval 阶段: 全量 top20 落盘，margin 调整全是后处理，零 GPU 成本。
- 生产阶段: --margin 时按 sim ≥ top1−margin 截断（封顶 TOP_N_CAP），
  输出长表（岗位×专业一行），后续与规则集做并集入 top 表。

前置: run_embed.py --range {range} 已跑完（cache/jobs_{suffix}/single/vectors.npy 存在）。

用法:
    python run_topn.py --range "2026(01-06)_专科"                # top20 全量落盘
    python run_topn.py --range "2026(01-06)_专科" --margin 0.05  # 另出 margin 截断长表
    python run_topn.py --range test --input xxx/ --limit 1000    # 测试

输出（outputs_topn_{suffix}/）:
    {range}_top20.jsonl   每行: {job_id, title, top: [{code,name,score}×20]}
    {range}_topn.jsonl    (--margin 时) 长表: {job_id, title, code, name, score}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config import *
from tqdm import tqdm


def iter_valid_records(in_files, limit=None):
    """与 run_embed 完全同序的岗位迭代器（保证和 vectors.npy 行对齐）。

    range 单文件模式不做学历二次过滤（预筛过）；--input 目录模式同 run_embed
    的读法：extract_fields → build_job_text 非空即有效。
    """
    n = 0
    for jp in in_files:
        with open(jp, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f):
                if limit and lineno >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not args_range_mode() and not is_zhuanke(rec):
                    continue
                fields = extract_fields(rec)
                if build_job_text(fields):
                    n += 1
                    yield fields


def args_range_mode():
    return globals().get("_RANGE_MODE", True)


def main():
    global _RANGE_MODE
    p = argparse.ArgumentParser(description="TopN: top20 带分落盘 + margin 截断")
    p.add_argument("--range", required=True, help="范围标识，如 2026(01-06)_专科")
    p.add_argument("--input", help="输入目录/文件（测试用，覆盖 --range 默认输入）")
    p.add_argument("--limit", type=int, help="限量（只处理前 N 条有效岗位）")
    p.add_argument("--topn", type=int, default=TOP_N_CAND, help=f"落盘候选数（默认 {TOP_N_CAND}）")
    p.add_argument("--margin", type=float, help="另出 margin 截断长表（如 0.05）")
    args = p.parse_args()

    rng = args.range
    suffix = rng.removesuffix("岗位")
    _RANGE_MODE = not args.input

    # 输入文件
    if args.input:
        ip = Path(args.input)
        in_files = sorted(ip.glob("*.jsonl")) if ip.is_dir() else [ip]
    else:
        fp, exists = resolve_range_file(rng)
        if not exists:
            print(f"[error] 输入不存在: {fp}")
            return
        in_files = [fp]

    # 缓存向量（与 run_embed 同目录约定）
    vec_dir = BASE_DIR / "cache" / f"jobs_{suffix}" / "single"
    vecs_path = vec_dir / "vectors.npy"
    if not vecs_path.exists():
        print(f"[error] 向量缓存不存在: {vecs_path}\n        先跑 run_embed.py --range {rng}")
        return
    vecs = np.memmap(vecs_path, dtype="float32", mode="r").reshape(-1, EMBED_DIM)

    majors = load_majors(MAJORS_FILE)
    print(f"[load] {len(majors)} 专业, 向量 {vecs.shape[0]:,} 条")

    # 专业索引（复用 run_embed 的缓存；不存在则报错引导先跑 run_embed）
    index_path = CACHE_MAJORS_DIR / "majors_744.index"
    if not index_path.exists():
        print(f"[error] 专业索引不存在: {index_path}\n        先跑 run_embed.py（会构建索引）")
        return
    import faiss
    index = faiss.read_index(str(index_path))

    # 行对齐校验：输入有效岗位数必须 == 向量行数
    out_dir = BASE_DIR / f"outputs_topn_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    top20_path = out_dir / f"{rng}_top20.jsonl"

    # margin 长表输出
    margin_fh = None
    margin_path = None
    if args.margin is not None:
        margin_path = out_dir / f"{rng}_topn.jsonl"
        margin_fh = open(margin_path, "w", encoding="utf-8")

    t0 = time.time()
    n_jobs = 0
    n_sel = 0  # margin 入选专业总数（统计 N 分布）
    BATCH = 8192

    recs = iter_valid_records(in_files, args.limit)
    if args.limit:
        recs_limit = args.limit
    else:
        recs_limit = vecs.shape[0]

    pbar = tqdm(total=min(recs_limit, vecs.shape[0]), unit="条",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    buf_recs = []

    def flush(batch_recs, batch_vecs):
        nonlocal n_jobs, n_sel
        scores, cand_idx = index.search(batch_vecs, min(args.topn, len(majors)))
        for bi, fields in enumerate(batch_recs):
            top = [{"code": majors[j]["major_code"],
                    "name": majors[j]["major_name"],
                    "score": round(float(scores[bi][k]), 4)}
                   for k, j in enumerate(cand_idx[bi])]
            row = {"job_id": fields["岗位唯一标识"],
                   "title": fields["原始岗位名称"],
                   "top": top}
            top20_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_jobs += 1
            if margin_fh:
                top1 = top[0]["score"]
                sel = [t for t in top if t["score"] >= top1 - args.margin][:TOP_N_CAP]
                for t in sel:
                    margin_fh.write(json.dumps({"job_id": row["job_id"],
                                                "title": row["title"],
                                                "code": t["code"],
                                                "name": t["name"],
                                                "score": t["score"]},
                                               ensure_ascii=False) + "\n")
                    n_sel += 1
        pbar.update(len(batch_recs))

    with open(top20_path, "w", encoding="utf-8") as top20_fh:
        for fields in recs:
            if n_jobs >= vecs.shape[0]:
                print(f"[warn] 岗位数超过向量行数，截断到 {vecs.shape[0]}")
                break
            buf_recs.append(fields)
            if len(buf_recs) >= BATCH:
                flush(buf_recs, np.array(vecs[n_jobs:n_jobs + len(buf_recs)], dtype="float32"))
                buf_recs = []
        if buf_recs:
            flush(buf_recs, np.array(vecs[n_jobs:n_jobs + len(buf_recs)], dtype="float32"))
    pbar.close()
    if margin_fh:
        margin_fh.close()

    print(f"\n{'='*60}")
    print(f"[完成] {time.time()-t0:.0f}s  岗位 {n_jobs:,}")
    print(f"  top20 → {top20_path}")
    if margin_path:
        print(f"  margin={args.margin} 长表 → {margin_path}  (入选专业 {n_sel:,}, 均 {n_sel/max(n_jobs,1):.2f} 个/岗)")


if __name__ == "__main__":
    main()
