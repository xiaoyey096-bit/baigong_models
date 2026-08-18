#!/usr/bin/env python3
"""
Phase 1: Embedding 编码 → 只存缓存向量，不输出匹配结果
=========================================================
- 构建/加载 744 专业 FAISS 索引 → cache/majors/majors_744.index
- 编码岗位 → cache/jobs/{源}/vectors.npy + candidates.npy
- 不生成 outputs_json / outputs_xlsx（留给 Phase 2）

用法:
    python run_embed.py                                 # 全量多源（data/new/）
    python run_embed.py --source 华为招聘                # 单个源
    python run_embed.py --limit 1000
    python run_embed.py --list-sources
    python run_embed.py --range 2026(01-06)_专科         # 预筛单文件模式（见下）

专科预筛单文件模式（--range）:
    输入读全量 models/data_dwd/{range}岗位.jsonl（回退 usedto_top3_data/）
    （已按 jobid 清单预筛过专科，不再跑 is_zhuanke）
    缓存写 cache/jobs_{range_suffix}/single/  （与旧多源结果隔离）

    新逻辑（2026-08-14）：语义对【全量】岗位编码，规则命中与否不再影响语义范围。
    规则匹配（run_rule_match）独立产出 _rule.jsonl，在 merge_results 阶段做置顶合并。

缓存结构:
    cache/majors/majors_744.index
    cache/jobs/{源}/          （多源模式）
    cache/jobs_{suffix}/single/  （--range 单文件模式；suffix = range 去"岗位"后缀）
        ├── vectors.npy       # N × 1024 × float32
        └── candidates.npy    # N × 10  × int64 (FAISS Top10 专业编号)
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

from config import *
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    sys.exit("CUDA 不可用")


class Embedder:
    def __init__(self, model_name, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name, device=device,
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.model.max_seq_length = EMBED_MAX_LEN
        try:
            self.model.prompts = {"query": EMBED_QUERY_INST, "passage": ""}
        except Exception:
            pass

    def encode(self, texts, is_query=False):
        kwargs = dict(normalize_embeddings=True, batch_size=EMBED_BATCH, show_progress_bar=False)
        # Qwen 系列支持 prompt_name + truncate_dim，BGE 不
        try:
            return self.model.encode(texts, prompt_name="query" if is_query else "passage",
                                     truncate_dim=EMBED_DIM, **kwargs).astype("float32")
        except (TypeError, ValueError):
            return self.model.encode(texts, **kwargs).astype("float32")


# ══════════════════════════════════════════════════════════════
def build_or_load_majors_index(embedder, majors):
    index_path = CACHE_MAJORS_DIR / "majors_744.index"
    n = len(majors)

    if index_path.exists():
        import faiss
        idx = faiss.read_index(str(index_path))
        if idx.ntotal == n and idx.d == EMBED_DIM:
            print(f"[majors] 命中缓存 ({idx.ntotal} 条 {idx.d}d)")
            return idx

    print(f"[majors] 编码 {n} 个专业向量...")
    texts = [m["embedding_text"] for m in majors]
    vecs = embedder.encode(texts, is_query=False)

    import faiss
    idx = faiss.IndexFlatIP(EMBED_DIM)
    idx.add(vecs)
    CACHE_MAJORS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(index_path))
    print(f"[majors] 已保存 → {index_path}")
    return idx


# ══════════════════════════════════════════════════════════════
def process_source(src_name, embedder, index, majors, args):
    if args.range:
        # 单文件模式：读全量 data_dwd/{range}岗位.jsonl（回退 usedto_top3_data/）
        # 新逻辑（2026-08-14）：语义对【全量】岗位编码，不再只编码规则未命中部分。
        # 规则命中/未命中已不再影响语义范围 —— 规则只在 merge_results 阶段做置顶。
        suffix = args.range.removesuffix("岗位")
        job_file, _ = resolve_range_file(args.range)
        job_dir = BASE_DIR / "cache" / f"jobs_{suffix}" / "single"
    else:
        suffix = None
        job_file = JOBS_DIR / f"{src_name}.jsonl"
        job_dir = CACHE_JOBS_DIR / src_name
    if not job_file.exists():
        print(f"[skip] {src_name}: 文件不存在 {job_file}")
        return

    vecs_path = job_dir / "vectors.npy"
    cands_path = job_dir / "candidates.npy"

    # 读有效行
    all_texts = []
    with open(job_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            if args.limit and lineno >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not args.range and not is_zhuanke(rec):
                # 单文件已预筛，不再二次过滤；多源模式仍按学历筛
                continue
            text = build_job_text(extract_fields(rec))
            if text:
                all_texts.append(text)

    n = len(all_texts)
    if n == 0:
        print(f"[{src_name}] 无有效岗位")
        return

    # 续跑
    job_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_resume:
        existing = vecs_path.stat().st_size // (EMBED_DIM * 4) if vecs_path.exists() else 0
        if existing >= n:
            print(f"[{src_name}] 已完成 ({existing}/{n})，跳过")
            return
    else:
        for p in [vecs_path, cands_path]:
            if p.exists():
                p.unlink()
        existing = 0

    now_str = time.strftime("%H:%M:%S")
    print(f"\n{'='*60}\n[{now_str}] {src_name}  {n:,} 条 ({existing} 已完成)\n{'='*60}")

    t0 = time.time()
    with open(vecs_path, "ab") as vf, open(cands_path, "ab") as cf:
        pbar = tqdm(total=n, initial=existing, desc=f"[{src_name}]", unit="条",
                    bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for i in range(existing, n, EMBED_BATCH):
            end = min(i + EMBED_BATCH, n)
            bt = all_texts[i:end]

            vecs = embedder.encode(bt, is_query=True)
            _, cand_idx = index.search(vecs, TOP_K_STAGE1)

            for bi in range(len(bt)):
                vf.write(vecs[bi].tobytes())
                cf.write(cand_idx[bi].astype(np.int64).tobytes())

            vf.flush(); cf.flush()
            pbar.update(end - i)

    pbar.close()
    speed = n / max(time.time() - t0, 1e-6)
    mb = n * EMBED_DIM * 4 / 1048576
    print(f"[{src_name}] ✅ {n}条  vectors={mb:.1f}MB  {speed:.1f}条/秒  {time.time()-t0:.0f}s")


# ══════════════════════════════════════════════════════════════
def list_sources():
    total = 0
    for p in sorted(JOBS_DIR.glob("*.jsonl")):
        n = sum(1 for _ in open(p, "rb"))
        total += n
        print(f"  {n:>10,}  {p.stem}")
    print(f"\n  {'─'*10}\n  {total:>10,}  总计")


def main():
    p = argparse.ArgumentParser(description="Phase 1: Embedding → cache only")
    p.add_argument("--source")
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--list-sources", action="store_true")
    p.add_argument("--range",
                   help="单文件模式：读全量 data_dwd/{range}岗位.jsonl（回退 usedto_top3_data/），"
                        "缓存写 cache/jobs_{suffix}/。如 --range 2026(01-06)_专科")
    args = p.parse_args()

    if args.list_sources:
        list_sources()
        return

    majors = load_majors(MAJORS_FILE)
    print(f"[load] {len(majors)} 专业")

    CACHE_MAJORS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.range:
        CACHE_JOBS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[device] {device}  {EMBED_MODEL.split('/')[-1]}")

    embedder = Embedder(EMBED_MODEL, device)
    index = build_or_load_majors_index(embedder, majors)

    if args.range:
        suffix = args.range.removesuffix("岗位")
        print(f"[range] 单文件模式（全量）：{args.range} → cache/jobs_{suffix}/")
        sources = ["single"]
    else:
        sources = [args.source] if args.source else sorted(sp.stem for sp in JOBS_DIR.glob("*.jsonl"))
    t_all = time.time()
    for src in sources:
        process_source(src, embedder, index, majors, args)
    print(f"\n[完成] {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
