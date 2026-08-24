#!/usr/bin/env python3
"""
merge v2: 语义 margin 集 ∪ 规则集 → 最终 TopN 长表
======================================================================
- 语义: run_topn.py --margin 产的 {rng}_topn.jsonl（sim ≥ top1−margin, 原始分）
- 规则: run_rule_match_v2.py 产的 {rng}_rule_v2.jsonl（精确1.0/泛称0.9）
- 合并: 按专业 code 去重取并集；同 code 撞车时分数高者胜
  （1.0精确 > 0.9泛称 > 语义原始分，天然保来源语义，无独立标签列）
- 输出: outputs_topn_{suffix}/{rng}_final.jsonl —— 入 top 表的数据源

用法:
    python merge_topn.py --range "2026(01-06)_专科" --margin 0.05
"""

import argparse
import json
from pathlib import Path

from config import *


def load_long(p):
    out = {}
    if not Path(p).exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            jid = r["job_id"]
            out.setdefault(jid, {"title": r.get("title", ""), "majors": {}})
            prev = out[jid]["majors"].get(r["code"])
            if prev is None or r["score"] > prev["score"]:   # 同 code 分高者胜
                out[jid]["majors"][r["code"]] = {"name": r["name"], "score": r["score"]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--range", required=True)
    p.add_argument("--margin", type=float, default=MARGIN)
    args = p.parse_args()

    rng = args.range
    suffix = rng.removesuffix("岗位")

    sem_path = BASE_DIR / f"outputs_topn_{suffix}" / f"{rng}_topn.jsonl"
    rule_path = BASE_DIR / f"outputs_rule_{suffix}" / f"{rng}_rule_v2.jsonl"
    if not sem_path.exists():
        raise SystemExit(f"[error] 语义长表不存在: {sem_path}\n"
                         f"        先跑 run_topn.py --range {rng} --margin {args.margin}")
    if not rule_path.exists():
        print(f"[warn] 规则文件不存在（{rule_path}），纯语义模式")

    sem, rule = load_long(sem_path), load_long(rule_path)
    all_jobs = set(sem) | set(rule)
    print(f"[load] 语义 {len(sem):,} 岗 / 规则 {len(rule):,} 岗 / 并集 {len(all_jobs):,} 岗")

    out_path = BASE_DIR / f"outputs_topn_{suffix}" / f"{rng}_final.jsonl"
    n_rows = both = 0
    with open(out_path, "w", encoding="utf-8") as of:
        for jid in all_jobs:
            s = sem.get(jid, {}).get("majors", {})
            r = rule.get(jid, {}).get("majors", {})
            both += len(set(s) & set(r))
            # 并集: 规则独有 + 语义全集；撞车时取分高者（1.0精确>0.9泛称>语义原始分）
            merged = {**{c: v for c, v in r.items() if c not in s},
                      **{c: v for c, v in s.items()}}
            for c, v in r.items():
                if c in merged and v["score"] > merged[c]["score"]:
                    merged[c] = v
            title = (sem.get(jid) or rule.get(jid)).get("title", "")
            for c, v in merged.items():
                of.write(json.dumps({"job_id": jid, "title": title,
                                     "code": c, "name": v["name"], "score": v["score"]},
                                    ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"[完成] → {out_path}  {n_rows:,} 行 (均 {n_rows/max(len(all_jobs),1):.2f} 专业/岗, 规则语义双命中 {both:,})")


if __name__ == "__main__":
    main()
