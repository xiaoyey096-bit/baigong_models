#!/usr/bin/env python3
"""
模块 B: 固定 TopN —— 每个岗位固定输出 top_n 个专业（1~20，运行配置决定）
======================================================================
- 基础: run_topn.py 的 {rng}_top20.jsonl（BGE top20 带分）
- 规则(可选): run_rule_match_v2.py 的规则长表 —— 精确命中(1.0)的专业
  强制入选并排前，剩余名额按 cosine 降序补满，恰好 N 个
- 置信层(margin 在此模块的语义):
    HIGH   sim ≥ top1 − margin     （真匹配，合同里可进权威报表）
    MEDIUM sim ≥ MEDIUM_FLOOR(0.5) （候选展示）
    LOW    其余                     （复核池）
- 输出长表: {job_id, title, code, name, score, rank, confidence_level, rule_hit}
  schema 与模块 A(灵活N)同构 + rank/confidence 列，平台选配。

用法:
    python run_fixed_topn.py --range "2026(01-06)_专科" --n 5
    python run_fixed_topn.py --range "..." --n 10 --with-rule --margin 0.05
输出: outputs_topn_{suffix}/{rng}_fixed{n}.jsonl
"""

import argparse
import json
from pathlib import Path

from config import *

MEDIUM_FLOOR = 0.50   # LOW/MEDIUM 分界（BGE cosine 绝对下限，eval 后可调）


def classify(score: float, top1: float, margin: float) -> str:
    if score >= top1 - margin:
        return "HIGH"
    if score >= MEDIUM_FLOOR:
        return "MEDIUM"
    return "LOW"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--range", required=True)
    p.add_argument("--n", type=int, required=True, help="固定输出专业数 1~20")
    p.add_argument("--with-rule", action="store_true", help="规则精确命中强制入选置前")
    p.add_argument("--margin", type=float, default=MARGIN, help="HIGH 分界（top1−margin）")
    args = p.parse_args()

    if not 1 <= args.n <= 20:
        raise SystemExit(f"[error] --n 需在 1~20，收到 {args.n}")

    rng = args.range
    suffix = rng.removesuffix("岗位")

    top20_path = BASE_DIR / f"outputs_topn_{suffix}" / f"{rng}_top20.jsonl"
    if not top20_path.exists():
        raise SystemExit(f"[error] {top20_path} 不存在，先跑 run_topn.py --range {rng}")
    rule_path = BASE_DIR / f"outputs_rule_{suffix}" / f"{rng}_rule_v2.jsonl"
    rule = {}
    if args.with_rule:
        if not rule_path.exists():
            print(f"[warn] {rule_path} 不存在，忽略 --with-rule")
        else:
            with open(rule_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r["score"] >= SCORE_RULE_EXACT:      # 只用精确命中置顶
                            rule.setdefault(r["job_id"], set()).add(r["code"])
            print(f"[rule] 精确命中岗位 {len(rule):,}")

    out_path = BASE_DIR / f"outputs_topn_{suffix}" / f"{rng}_fixed{args.n}.jsonl"
    n_jobs = n_high = n_rows = 0
    with open(top20_path, encoding="utf-8") as f, \
         open(out_path, "w", encoding="utf-8") as of:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            top = r["top"][:max(args.n, 20)]                    # 候选池
            top1 = top[0]["score"]
            sem_by_code = {t["code"]: t for t in top}

            # 排序: 规则精确命中置前（多按分），其余按 cosine 降序；恰好截 N
            rule_codes = rule.get(r["job_id"], set()) & set(sem_by_code)
            ordered = [t for c in rule_codes for t in [sem_by_code[c]]]
            ordered += [t for t in top if t["code"] not in rule_codes]
            ordered = sorted(ordered[:args.n], key=lambda t: t["score"], reverse=True)

            has_high = any(classify(t["score"], top1, args.margin) == "HIGH" for t in ordered)
            n_jobs += 1
            n_high += has_high
            for rank, t in enumerate(ordered, 1):
                of.write(json.dumps({
                    "job_id": r["job_id"], "title": r["title"],
                    "code": t["code"], "name": t["name"], "score": t["score"],
                    "rank": rank,
                    "confidence_level": classify(t["score"], top1, args.margin),
                    "rule_hit": t["code"] in rule_codes,
                }, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"[完成] {n_jobs:,} 岗 × 固定 {args.n} = {n_rows:,} 行 → {out_path}")
    print(f"  含 HIGH 的岗位 {n_high:,} ({n_high/max(n_jobs,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
