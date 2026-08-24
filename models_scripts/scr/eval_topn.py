#!/usr/bin/env python3
"""
eval: 用金标横向评 TopN 方案（margin 扫描 / 规则并集 / 指标）
======================================================================
- 输入: eval/gold_{range}.jsonl（judge_eval.py 产）+ outputs_topn_{suffix}/{rng}_top20.jsonl
        + 可选 outputs_rule_{suffix}/{rng}_rule.jsonl（规则命中，评"加规则并集"用）
- 指标: 集合 Precision / Recall / F1（不排序，不做排名指标）

用法:
    python eval_topn.py --range "2026(01-06)_专科"                       # margin 网格扫描
    python eval_topn.py --range "2026(01-06)_专科" --margin 0.05         # 单值详评
    python eval_topn.py --range "2026(01-06)_专科" --with-rule           # 叠加规则并集
    python eval_topn.py --range "..." --calibrate eval/calibrate_x.jsonl # 裁判vs人工一致率
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from config import *


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return None
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def set_metrics(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(gold) if gold else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--range", required=True)
    p.add_argument("--margin", type=float, help="单值详评（不给则网格扫描）")
    p.add_argument("--with-rule", action="store_true", help="叠加规则命中并集再评")
    p.add_argument("--calibrate", help="校准模式: 传人工标注 jsonl，对 LLM 裁判一致率")
    args = p.parse_args()

    rng = args.range
    suffix = rng.removesuffix("岗位")

    # ── 校准模式 ──
    if args.calibrate:
        human = {r["job_id"]: set(r.get("人工标注") or []) for r in (load_jsonl(args.calibrate) or [])}
        gold_rows = load_jsonl(EVAL_DIR_LOCAL / f"gold_{rng}.jsonl") or []
        llm = {r["job_id"]: set(r["gold"]) for r in gold_rows if r["job_id"] in human}
        if not llm:
            print(f"[error] 金标 {EVAL_DIR_LOCAL}/gold_{rng}.jsonl 里没有校准样本的 job_id，"
                  f"先用同 --seed 重跑 judge_eval")
            return
        exact = sum(1 for j in llm if llm[j] == human[j])
        ps, rs, f1s = [], [], []
        for j in llm:
            pr, rc, f1 = set_metrics(llm[j], human[j])
            ps.append(pr); rs.append(rc); f1s.append(f1)
        n = len(llm)
        print(f"裁判 vs 人工 (n={n}): 集合完全一致 {exact}/{n} ({exact/n*100:.0f}%)")
        print(f"  平均 Precision {sum(ps)/n:.3f}  Recall {sum(rs)/n:.3f}  F1 {sum(f1s)/n:.3f}")
        print(f"  ≥85% 完全一致或 F1≥0.9 → 裁判可信")
        return

    # ── 常规 eval ──
    gold = {r["job_id"]: set(r["gold"]) for r in (load_jsonl(EVAL_DIR_LOCAL / f"gold_{rng}.jsonl") or [])}
    if not gold:
        sys_exit_no_gold(EVAL_DIR_LOCAL / f"gold_{rng}.jsonl")
        return
    top20 = {r["job_id"]: r["top"] for r in (load_jsonl(BASE_DIR / f"outputs_topn_{suffix}" / f"{rng}_top20.jsonl") or [])}

    rule = {}
    if args.with_rule:
        for r in (load_jsonl(BASE_DIR / f"outputs_rule_{suffix}" / f"{rng}_rule.jsonl") or []):
            codes = {r.get(f"专业{i}代码") for i in (1, 2, 3)} - {"", None}
            rule[r["岗位唯一标识"]] = codes

    def predict(jid, margin):
        top = top20.get(jid, [])
        if not top:
            return set()
        top1 = top[0]["score"]
        pred = {t["code"] for t in top if t["score"] >= top1 - margin}
        pred = set(sorted(pred)[:TOP_N_CAP]) if len(pred) > TOP_N_CAP else pred
        if args.with_rule:
            pred = pred | rule.get(jid, set())
        return pred

    margins = [args.margin] if args.margin else [round(0.01 * i, 2) for i in range(1, 16)]
    print(f"\n金标 {len(gold)} 岗位" + ("（+规则并集）" if args.with_rule else ""))
    print(f"{'margin':>7} {'P':>7} {'R':>7} {'F1':>7} {'均N':>6}")
    for m in margins:
        ps, rs, fs, ns = [], [], [], []
        for jid, g in gold.items():
            pred = predict(jid, m)
            pr, rc, f1 = set_metrics(pred, g)
            ps.append(pr); rs.append(rc); fs.append(f1); ns.append(len(pred))
        n = len(gold)
        print(f"{m:>7.2f} {sum(ps)/n:>7.3f} {sum(rs)/n:>7.3f} {sum(fs)/n:>7.3f} {sum(ns)/n:>6.2f}")

    if args.margin:
        # 单值详评: N 分布 + 典型错例
        n_dist = Counter()
        for jid in gold:
            n_dist[len(predict(jid, args.margin))] += 1
        print(f"\nN 分布: " + " ".join(f"{k}个→{v}" for k, v in sorted(n_dist.items())))
        print("\n错例 top5 (F1 最低):")
        scored = []
        for jid, g in gold.items():
            pred = predict(jid, args.margin)
            _, _, f1 = set_metrics(pred, g)
            scored.append((f1, jid, pred, g))
        for f1, jid, pred, g in sorted(scored)[:5]:
            print(f"  F1={f1:.2f} {top20.get(jid, [{}])[0].get('name','?')} "
                  f"| 漏={sorted(g-pred)} 多={sorted(pred-g)}")


EVAL_DIR_LOCAL = BASE_DIR / "eval"


def sys_exit_no_gold(p):
    print(f"[error] 金标不存在: {p}\n        先跑 judge_eval.py --range {p}")


if __name__ == "__main__":
    main()
