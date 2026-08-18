#!/usr/bin/env python3
"""
典型岗位归类 — Qwen3-Embedding-4B 版
=====================================
输入:
  种子   models/典型岗位名称_gt100_{范围}.jsonl   (≥100 专业；种子数随专业职业颗粒度而定，不设上限)
         默认与 --range 同名；可用 --seeds 指定别的种子文件
  数据   models/majors_data/{月份范围}/majors_data_dwd_all/{专业代码}_{专业名称}.jsonl
  模型   models/Qwen3-Embedding-4B  (MRL 缩维 1024d)

输出: models/jobsname/output/datarow_gt100/{月份范围}/{专业代码}_{专业名称}.jsonl

═══════════════════════════════════════════════════════════════
命名规范（详见 工作流方案.md）：
  · 输入 dwd：majors_data/{月份范围}/majors_data_dwd_all/
  · 输出：    jobsname/output/datarow_gt100/{月份范围}/
  · 月份范围 = Stage 3 产出的累积月份标识（如 2026(01-06)），由 --range 指定

阈值说明：
  · MIN_JOBS_PER_MAJOR = 100
    含义：只对该专业有效匹配岗位数 ≥ 100 的专业做典型岗位归类。
    理由：<100 的专业样本太少，词频归纳出的种子不可靠；种子文件
    典型岗位名称_gt100_{范围}.jsonl 覆盖所有 ≥100 的专业。
    ⚠️ 此值可能随甲方需求变化，改这里 + 同步种子文件即可。
  · THETA1 = 0.65：Qwen 余弦匹配阈值，≥ 则归入种子，< 则归"其他"。
  · 标题清洗：embedding/匹配用 clean_name()（剥"(急招)"【直聘】前缀、公司名段），
    输出"岗位名称"保留原始值（2026-08-16：eval 实测前缀噪声拖低余弦，护理 +26pt）。
  · 向量缓存：output/embed_cache_fp16.npz（key=文本，跨专业共享，fp16 落盘）。
    种子迭代/重跑时岗位名不变，命中缓存不重编码；崩溃由原子替换保证缓存不损坏。

用法:
    source ~/workspace/venv/bin/activate
    # 指定处理的月份范围（必填）
    python run_classify.py --range 2026(01-06)            # 该范围全量大专业
    python run_classify.py --range 2026(01-06) --major 530701   # 单个专业
"""

import argparse, json, time
from pathlib import Path
from collections import Counter
import numpy as np

MODELS_DIR   = Path("/home/cl-715/workspace/models")
SEEDS_ROOT   = MODELS_DIR        # 种子文件放在 models/ 根目录，命名 典型岗位名称_gt100_{范围}.jsonl
MAJORS_DATA_ROOT = MODELS_DIR / "majors_data"
EMBED_DIR    = MODELS_DIR / "Qwen3-Embedding-4B"
JOBSNAME_OUTPUT_ROOT = MODELS_DIR / "jobsname" / "output" / "datarow_gt100"

# ── 阈值（可能随甲方需求调整）──────────────────────────────
MIN_JOBS_PER_MAJOR = 100   # 只分析有效匹配岗位数 ≥100 的专业；与种子文件 ≥100 对应
THETA1       = 0.65        # Qwen 余弦匹配阈值（待校准脚本确认）
EMBED_DIM    = 1024        # Qwen MRL 缩维
EMBED_BATCH  = 512       # 2026-08-17：GPU 空闲后 128→512（显存富余时更大批次吞吐更高）
OUT_COLUMNS  = ["岗位唯一标识", "岗位名称", "典型岗位名称"]

# ── 标题清洗（embedding 前用，输出保留原始名）─────────────────
# 2026-08-16 eval：新批次大量 "(急招)口腔护士"、"【直聘】护士"、"公司名\t-\t岗位"
# 式标题，前缀噪声把余弦拖到 θ 以下（护理专业 raw→clean 覆盖率 55%→81%）。
# 清洗只用于编码与匹配；落盘的"岗位名称"始终是原始值。
import re as _re
_PFX = _re.compile(r'^[\(\[【「『\s]*[^\)\]」』]{1,8}[\)\]」』]\s*')

def clean_name(nm: str) -> str:
    """剥招聘前缀 + 公司名段：'(急聘)口腔护士'→'口腔护士'，'XX公司\t-\t组装'→'组装'。"""
    nm = nm.strip()
    parts = _re.split(r"\s*[\t\-—–]+\s*", nm)
    if len(parts) > 1:
        nm = min(parts, key=len)
    return _PFX.sub("", nm).strip()


def load_seeds(seeds_file):
    seeds = {}
    with open(seeds_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            seeds[r["专业代码"]] = {"major_name": r["专业名称"], "seeds": r["典型岗位名称"]}
    return seeds


def load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        str(EMBED_DIR),
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
    )


def encode(embedder, texts):
    """Qwen 编码 + MRL 缩维 1024d + 归一化"""
    return embedder.encode(
        texts, normalize_embeddings=True, batch_size=EMBED_BATCH,
        show_progress_bar=False, truncate_dim=EMBED_DIM,
    ).astype("float32")


# ── 向量缓存（fp16 npz，key=文本，跨专业共享）────────────────
# 同一批 dwd 的岗位名在种子迭代/重跑时完全复用，只有种子变 —— 缓存后
# 重跑只编码新名字。fp16 落盘（体积减半），使用时转回 fp32，余弦影响 <1e-3。
EMBED_CACHE_PATH   = MODELS_DIR / "jobsname" / "output" / "embed_cache_fp16.npz"
CACHE_SAVE_NAMES   = 5000   # 累计新增多少个名字落盘一次（崩溃最多损失这段）
import numpy as _np

class CachedEmbedder:
    """包一层 encode：命中缓存直接取，缺失的批量编码后入缓存。"""

    def __init__(self, embedder, path=EMBED_CACHE_PATH):
        self.embedder = embedder
        self.path = path
        self.cache = {}          # text -> fp16 向量(EMBED_DIM,)
        self._dirty = 0          # 上次落盘后新增的名字数
        if path.exists():
            data = _np.load(path, allow_pickle=False)
            for nm, v in zip(data["names"], data["vecs"]):
                self.cache[str(nm)] = v
            print(f"[cache] 载入 {len(self.cache):,} 条已有向量 ({path.name})")

    def encode_cached(self, texts):
        """等价于 encode(embedder, texts)，但命中的不重复编码。返回 fp32 (N, dim)。"""
        missing = sorted({t for t in texts if t not in self.cache})
        if missing:
            new = encode(self.embedder, missing).astype("float16")
            for t, v in zip(missing, new):
                self.cache[t] = v
            self._dirty += len(missing)
        return _np.stack([self.cache[t].astype("float32") for t in texts])

    def maybe_save(self, force=False):
        if not self._dirty:
            return
        if not force and self._dirty < CACHE_SAVE_NAMES:
            return
        names = _np.array(list(self.cache.keys()))
        vecs = _np.stack(list(self.cache.values()))
        tmp = self.path.with_suffix(".tmp.npz")
        _np.savez(tmp, names=names, vecs=vecs)
        tmp.replace(self.path)   # 原子替换，避免写一半崩溃留下坏文件
        print(f"[cache] 落盘 {len(self.cache):,} 条向量 (+{self._dirty:,})")
        self._dirty = 0


def process_major(major_code, enc, seeds_data, dwd_dir, output_dir):
    info = seeds_data[major_code]
    major_name = info["major_name"]
    seeds = info["seeds"]

    data_file = None
    p = dwd_dir / f"{major_code}_{major_name}.jsonl"
    if p.exists():
        data_file = p
    else:
        for f in dwd_dir.glob(f"{major_code}_*.jsonl"):
            data_file = f
            break
    if data_file is None:
        return None  # 该专业在本次数据里无有效匹配

    jobs = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                jid = r.get("岗位唯一标识", "")
                jname = (r.get("原始岗位名称") or "").strip()
                if jname: jobs.append((jid, jname))
            except: continue

    n = len(jobs)
    if n == 0:
        return None

    # 阈值过滤：< MIN_JOBS_PER_MAJOR 的专业不归类（样本不足，种子不可靠）
    if n < MIN_JOBS_PER_MAJOR:
        print(f"  [skip] {major_code} {major_name}: {n} 条 < {MIN_JOBS_PER_MAJOR}，不归类")
        return None

    # 去重按【清洗后】名计（"(急招)X"与"X"编码一次），输出仍写原始名
    name_freq = Counter(clean_name(j[1]) for j in jobs)
    unique_names = list(name_freq.keys())

    seed_emb = enc.encode_cached(seeds)
    name_emb = enc.encode_cached(unique_names)

    sim = name_emb @ seed_emb.T
    best_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)

    # 典型岗位归类：按 unique name 匹配种子（去重后只编码一次），命中阈值即归入对应种子，否则"其他"
    typical_map = {}
    matched_names = 0          # 命中的 unique name 数（去重口径，仅供诊断）
    for i, name in enumerate(unique_names):
        if best_sim[i] >= THETA1:
            typical_map[name] = seeds[best_idx[i]]
            matched_names += 1
        else:
            typical_map[name] = "其他"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{major_code}_{major_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for jid, jname in jobs:
            tn = typical_map.get(clean_name(jname), "其他")
            f.write(json.dumps({
                "岗位唯一标识": jid,
                "岗位名称": jname,
                "典型岗位名称": tn,
            }, ensure_ascii=False) + "\n")

    # 统计口径：matched/other 按【岗位行数】计（与落盘文件、与 n 同口径），
    # 而非去重 unique name 数 —— 否则高频重复名会严重稀释命中率百分比
    matched = sum(c for name, c in name_freq.items() if typical_map[name] != "其他")
    other_count = n - matched
    print(f"  ✅ {major_code} {major_name}: {n:,}条 → 种子桶{matched:,}({matched/n*100:.0f}%) 其他{other_count:,}({other_count/n*100:.0f}%) "
          f"[去重名{len(unique_names)},命中{matched_names}]")
    return major_code, major_name, n, matched, other_count


def main():
    ap = argparse.ArgumentParser(description="典型岗位归类（Qwen3-Embedding-4B，按月份范围）")
    ap.add_argument("--range", required=True,
                    help="处理的月份范围标识，如 2026(01-06)；读 majors_data/{range}/，写 datarow_gt100/{range}/")
    ap.add_argument("--major", help="只跑单个专业代码")
    ap.add_argument("--seeds", help="种子文件路径；默认 models/典型岗位名称_gt100_{range}.jsonl")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已输出的专业（默认跳过已完成的）")
    args = ap.parse_args()

    rng = args.range
    dwd_dir    = MAJORS_DATA_ROOT / rng / "majors_data_dwd_all"
    output_dir = JOBSNAME_OUTPUT_ROOT / rng

    if not dwd_dir.exists():
        print(f"[error] 输入 dwd 目录不存在: {dwd_dir}")
        print(f"        请先跑 Stage 3 (build_by_major.py --output-range {rng})")
        return

    # 种子文件：默认与 --range 同名（典型岗位名称_gt100_{range}.jsonl），可用 --seeds 覆盖
    seeds_file = Path(args.seeds) if args.seeds else SEEDS_ROOT / f"典型岗位名称_gt100_{rng}.jsonl"
    if not seeds_file.exists():
        print(f"[error] 种子文件不存在: {seeds_file}")
        print(f"        可用 --seeds 指定路径，或先构建种子（见 jobsname/种子构建说明.md）")
        return

    print(f"[config] 月份范围: {rng}")
    print(f"[config] 输入 dwd: {dwd_dir}")
    print(f"[config] 种子:     {seeds_file}")
    print(f"[config] 输出:     {output_dir}")
    print(f"[config] 阈值:     MIN_JOBS_PER_MAJOR={MIN_JOBS_PER_MAJOR}  THETA1={THETA1}\n")

    seeds_data = load_seeds(seeds_file)
    print(f"[load] {len(seeds_data)} 个专业种子")

    codes = [args.major] if args.major else sorted(seeds_data.keys())

    embedder = load_embedder()
    enc = CachedEmbedder(embedder)   # 向量缓存：重跑/种子迭代时命中不重编码
    print(f"[model] Qwen3-Embedding-4B  dim={EMBED_DIM}\n")

    results = []
    skipped = 0
    t_all = time.time()
    try:
        for i, code in enumerate(codes, 1):
            if code not in seeds_data:
                continue
            # 断点续跑：输出文件已存在的专业跳过（向量有缓存，中断后无损续）
            if not args.overwrite:
                done_path = output_dir / f"{code}_{seeds_data[code]['major_name']}.jsonl"
                if done_path.exists():
                    skipped += 1
                    continue
            print(f"[{i}/{len(codes)}]", end="", flush=True)
            res = process_major(code, enc, seeds_data, dwd_dir, output_dir)
            if res: results.append(res)
            enc.maybe_save()
    finally:
        enc.maybe_save(force=True)   # 收尾/异常退出前把缓存落盘

    print(f"\n{'='*60}")
    print(f"本次新跑 {len(results)} 专业  跳过已完成 {skipped}  耗时 {time.time()-t_all:.0f}s")
    if skipped:
        print(f"（跳过的 {skipped} 个沿用已有输出文件；全量统计请扫输出目录）")
    print(f"{'='*60}")
    total_jobs = sum(r[2] for r in results)
    total_matched = sum(r[3] for r in results)
    if total_jobs:
        print(f"总岗位: {total_jobs:,}  种子桶: {total_matched:,} ({total_matched/total_jobs*100:.1f}%)  其他: {total_jobs-total_matched:,}")
    print(f"\n输出: {output_dir}/")


if __name__ == "__main__":
    main()
