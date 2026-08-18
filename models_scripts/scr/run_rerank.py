#!/usr/bin/env python3
"""
Phase 2: Qwen3-0.6B Reranker 精排 Top3
========================================
- 读 cache/jobs/{源}/vectors.npy + candidates.npy
- Qwen3-Reranker-0.6B CrossEncoder（max_len=384, batch=64）
- 按数据源大小升序处理，progress.json 断点续跑
- 输出 outputs_json/{源}.jsonl

用法:
    python run_rerank.py                          # 全量（升序，跳过已完成）
    python run_rerank.py --source 华为招聘         # 单个源
    python run_rerank.py --limit 100              # 每源限条数
    python run_rerank.py --range 2026(01-06)_专科  # 预筛单文件模式（见下）

专科预筛单文件模式（--range）:
    读 cache/jobs_{suffix}/single/ 的向量缓存（Stage1 产出）
    读 data_dwd/{range}岗位.jsonl（全量）取岗位文本（不再跑 is_zhuanke）
    输出 outputs_json_{suffix}/{range}.jsonl（全量语义 Top，与旧多源结果隔离）

    新逻辑（2026-08-14）：语义对【全量】岗位精排。规则匹配独立产出 _rule.jsonl，
    在 merge_results.py 阶段做「规则置顶 + 语义补满 Top3 + 去重」合并。
    语义不再因规则命中而跳过任何岗位。
"""

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
from sentence_transformers import CrossEncoder

# ══════════════════════════════════════════════════════════════
def save_progress(prog: dict, path=None):
    p = path or PROGRESS_FILE
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PROGRESS_FILE)


def process_source(src_name, model, majors, args, prog):
    suffix = f"_p{args.offset}" if args.offset else ""
    if args.range:
        rsuffix = args.range.removesuffix("岗位")
        job_dir = BASE_DIR / "cache" / f"jobs_{rsuffix}" / "single"
        out_dir = BASE_DIR / f"outputs_json_{rsuffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 新逻辑（2026-08-14）：语义对【全量】岗位精排，输出 {range}.jsonl（全量语义 Top）。
        # 规则命中与否不再影响语义范围；规则置顶合并由 merge_results.py 负责。
        out_name = f"{args.range}.jsonl"
        jsonl_path = out_dir / out_name
        prog_key = out_name + suffix
    else:
        job_dir = CACHE_JOBS_DIR / src_name
        jsonl_path = OUTPUT_JSON_DIR / f"{src_name}{suffix}.jsonl"
        prog_key = f"{src_name}{suffix}"
    vecs_path = job_dir / "vectors.npy"
    cands_path = job_dir / "candidates.npy"

    if not vecs_path.exists():
        print(f"[skip] {src_name}: 无缓存")
        return

    total_cached = vecs_path.stat().st_size // (EMBED_DIM * 4)
    offset = args.offset
    n = min(total_cached - offset, args.limit or (total_cached - offset))
    if n <= 0:
        print(f"[skip] {src_name}: offset={offset} >= total={total_cached}")
        return

    # 断点续跑
    done = 0
    if args.resume and jsonl_path.exists():
        done = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
        if done >= n:
            print(f"[skip] {src_name}: 已完成 ({done}/{n})")
            prog[prog_key] = done
            return
        print(f"[resume] {src_name}: {done}/{n}")

    if not args.resume:
        for p in [jsonl_path]:
            if p.exists(): p.unlink()
        done = 0

    # 读缓存向量（只读 offset 后的部分）
    vecs = np.memmap(str(vecs_path), dtype="float32", mode="r", shape=(total_cached, EMBED_DIM))
    cands = np.memmap(str(cands_path), dtype="int64", mode="r", shape=(total_cached, TOP_K_STAGE1))

    # 读原始岗位文本（跳前 offset 条）—— 语义读全量，不再区分规则命中与否
    if args.range:
        job_file, _ = resolve_range_file(args.range)
    else:
        job_file = JOBS_DIR / f"{src_name}.jsonl"
    job_list = []
    skipped = 0
    with open(job_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(job_list) >= n:
                break
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except: continue
            keep = True if args.range else is_zhuanke(rec)   # 单文件已预筛
            if keep:
                if skipped < offset:
                    skipped += 1
                    continue
                fields = extract_fields(rec)
                job_list.append((fields, build_job_text(fields)))

    assert len(job_list) == n, f"岗位数 {len(job_list)} != 缓存条数 {n}"

    # 组装所有 pair：按行平铺 (0,ci0),(0,ci1),...,(1,ci0),...
    all_pairs = []
    for i in range(n):
        jt = job_list[i][1]
        for ci in range(TOP_K_STAGE1):
            mi = int(cands[i][ci])
            all_pairs.append((jt, majors[mi]["embedding_text"]))

    now_str = time.strftime("%H:%M:%S")
    print(f"\n{'='*60}\n[{now_str}] 开始: {src_name}  {n:,} 条  {len(all_pairs):,} pair ({done} 已完成)\n{'='*60}")

    # ── Reranker 推理（tqdm 进度条）──
    t0 = time.time()
    raw = []
    total_pair = len(all_pairs)
    pbar = tqdm(total=total_pair, initial=done * TOP_K_STAGE1,
                desc=f"[rerank {src_name}]", unit="pair",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for start in range(done * TOP_K_STAGE1, total_pair, RERANK_BATCH):
        end = min(start + RERANK_BATCH, total_pair)
        chunk = all_pairs[start:end]
        r = model.predict(chunk, batch_size=RERANK_BATCH, show_progress_bar=False)
        raw.extend(np.atleast_1d(r).astype("float32").tolist())
        pbar.update(end - start)
    pbar.close()

    t_rerank = time.time() - t0
    print(f"  Rerank 完成: {t_rerank:.0f}s  ({len(all_pairs)/max(t_rerank,1e-6):.0f} pair/s)")

    scores = (1.0 / (1.0 + np.exp(-np.array(raw, dtype="float32")))).tolist()

    # ── 写 JSONL ──
    t_write = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        for i in range(done, n):
            fields, _ = job_list[i]
            base = i * TOP_K_STAGE1
            local = [(scores[base + ci], ci) for ci in range(TOP_K_STAGE1)]
            local.sort(key=lambda x: -x[0])
            top3 = [(majors[int(cands[i][ci])]["major_code"],
                     majors[int(cands[i][ci])]["major_name"],
                     round(local[ri][0], 4))
                    for ri, (_, ci) in enumerate(local[:TOP_K_STAGE2])]
            jf.write(json.dumps(build_row(fields, top3), ensure_ascii=False) + "\n")

            if (i - done + 1) % FLUSH_EVERY == 0:
                jf.flush()
                prog[prog_key] = i + 1
                save_progress(prog)
                elapsed = max(time.time() - t0, 1e-6)
                spd = (i - done + 1) / elapsed
                eta_min = (n - i - 1) / max(spd, 1e-6) / 60
                pair_done = i * TOP_K_STAGE1 + min(TOP_K_STAGE1, (i - done) * TOP_K_STAGE1)
                now_str = time.strftime("%H:%M:%S")
                print(f"  [{now_str}] {src_name}  {i+1:,}/{n:,} ({spd:.1f}条/s)  ETA:{eta_min:.0f}min  Rerank:{pair_done:,}/{total_pair:,}")

    prog[prog_key] = n
    save_progress(prog)
    elapsed = max(time.time() - t0, 1e-6)
    print(f"[{src_name}] ✅ {n}条  {elapsed:.0f}s  {n/elapsed*60:.1f}条/分")



# ══════════════════════════════════════════════════════════════
def main():
    global PROGRESS_FILE
    p = argparse.ArgumentParser(description="Phase 2: Qwen3-0.6B Reranker Top3")
    p.add_argument("--source")
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0, help="从第 N 条开始（分卡用）")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--range",
                   help="单文件模式：读 cache/jobs_{suffix}/single/ 的缓存 + data_dwd/{range}岗位.jsonl（全量），"
                        "输出 outputs_json_{suffix}/{range}.jsonl（全量语义 Top）。如 --range 2026(01-06)_专科")
    args = p.parse_args()
    args.resume = not args.no_resume

    majors = load_majors(MAJORS_FILE)
    print(f"[load] {len(majors)} 专业")

    if args.range:
        rsuffix = args.range.removesuffix("岗位")
        # 进度文件隔离，避免污染旧多源模式进度
        PROGRESS_FILE = BASE_DIR / f"progress_{rsuffix}.json"
        sources = ["single"]
        print(f"[range] 单文件模式：{args.range} → outputs_json_{rsuffix}/{args.range}.jsonl")
    else:
        OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_XLSX_DIR.mkdir(parents=True, exist_ok=True)

        # 升序排数据源
        src_sizes = []
        for d in sorted(CACHE_JOBS_DIR.iterdir()):
            if not d.is_dir(): continue
            vp = d / "vectors.npy"
            if vp.exists():
                src_sizes.append((vp.stat().st_size // (EMBED_DIM * 4), d.name))
        src_sizes.sort()

        sources = [args.source] if args.source else [s for _, s in src_sizes]

        # 数据源概览
        prog_preview = {}
        if args.resume and PROGRESS_FILE.exists():
            prog_preview = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        print(f"\n数据源（升序）:")
        for n, s in src_sizes:
            mark = "✅" if prog_preview.get(s, 0) >= n else "⬜"
            print(f"  {mark} {s:20s} {n:>10,} 条")

    # 进度
    prog = {}
    if args.resume and PROGRESS_FILE.exists():
        prog = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))

    # 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[model] Qwen3-Reranker-0.6B  device={device}  max_len={RERANK_MAX_LEN}  batch={RERANK_BATCH}")
    model = CrossEncoder(
        str(MODELS_DIR / "Qwen3-Reranker-0.6B"),
        device=device, trust_remote_code=True,
        max_length=RERANK_MAX_LEN,
    )

    t_all = time.time()
    for src in sources:
        process_source(src, model, majors, args, prog)

    print(f"\n[完成] {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
