#!/usr/bin/env python3
"""
合并规则匹配 + 语义匹配 → 统一 Top3（规则置顶 + 语义补满 + 专业去重）
=====================================================================
新逻辑（2026-08-14 起）：
  ① 语义匹配对【全量岗位】跑，每个岗位都有语义 Top（带分数）
  ② 合并时规则命中的专业强制 score=1.0 并置顶，语义结果按分降序补位
  ③ 同一专业 code 在 Top3 内不重复出现
  ④ 固定截取 Top3

Top3 构成规则：
  · 规则命中 ≥3 个 → Top3 全是规则专业（1.0），语义不补
  · 规则命中 <3 个 → 规则专业(1.0)置顶 + 语义按分补齐到 3 个（去重）
  · 规则未命中     → 纯语义 Top3

用法:
    python merge_results.py \
        --rule     ../outputs_rule_2026(01-06)_专科/ \
        --semantic ../outputs_json_2026(01-06)_专科/ \
        --range    "2026(01-06)_专科"

    # 也支持直接传文件
    python merge_results.py --rule xx.jsonl --semantic yy.jsonl --range test

输出: outputs_merged_{range_suffix}/{range}_merged.jsonl（12 列，下游 Stage3 直接读）
"""

import argparse
import json
from pathlib import Path

from config import *


def collect_jsonl(src):
    """把参数解析成 jsonl 文件列表（目录 → 全部 jsonl，文件 → 单个）。"""
    p = Path(src)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if p.is_file() and p.suffix == ".jsonl":
        return [p]
    return []


def load_rows(files, key="岗位唯一标识"):
    """读所有文件 → {key: row}，后读的覆盖先读的（去重，保留最新）。"""
    rows = {}
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                k = r.get(key, "")
                if k:
                    rows[k] = r
    return rows


def parse_top3(row):
    """从 12 列行里解析出 Top3 → [(code, name, score), ...]（空位跳过）。"""
    top3 = []
    for i in range(1, 4):
        code = row.get(f"专业{i}代码", "")
        name = row.get(f"专业{i}名称", "")
        score = row.get(f"专业{i}匹配分数", "")
        if code:
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            top3.append((str(code), name, score))
    return top3


def merge_top3(rule_top3, sem_top3):
    """合并规则 + 语义 → Top3（规则置顶 1.0，语义补位，专业 code 去重）。

    Args:
        rule_top3: 规则命中的 [(code, name, score)]，score 通常已是 1.0
        sem_top3:  语义 Top [(code, name, score)]，按 score 降序

    Returns:
        list[(code, name, score)]，长度 ≤ 3
    """
    result = []
    seen = set()

    # ① 规则命中全部置顶（score 强制 1.0）
    for code, name, _ in rule_top3:
        code = str(code)
        if code and code not in seen:
            result.append((code, name, 1.0))
            seen.add(code)
        if len(result) >= TOP_K_STAGE2:
            break

    # ② 语义补位（去重，按分降序补到 Top3）
    for code, name, score in sem_top3:
        if len(result) >= TOP_K_STAGE2:
            break
        code = str(code)
        if code and code not in seen:
            result.append((code, name, score))
            seen.add(code)

    return result[:TOP_K_STAGE2]


def main():
    ap = argparse.ArgumentParser(description="合并规则 + 语义 → Top3（规则置顶补满去重）")
    ap.add_argument("--rule", required=True,
                    help="规则结果目录或文件（run_rule_match 的 _rule.jsonl）；可为空目录（无规则命中时）")
    ap.add_argument("--semantic", required=True,
                    help="语义结果目录或文件（run_rerank 全量语义输出）")
    ap.add_argument("--range", required=True, help="范围标识，决定输出路径")
    args = ap.parse_args()

    rng = args.range
    rsuffix = rng.removesuffix("岗位")
    out_dir = BASE_DIR / f"outputs_merged_{rsuffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rng}_merged.jsonl"

    # 1. 读规则结果（命中的岗位，12列；可能为空）
    rule_rows = {}
    rule_files = collect_jsonl(args.rule)
    if rule_files:
        rule_rows = load_rows(rule_files)
    print(f"[rule]    {len(rule_files)} 文件 → {len(rule_rows):,} 条规则命中")

    # 2. 读语义结果（全量岗位，12列）
    sem_files = collect_jsonl(args.semantic)
    if not sem_files:
        print(f"[error] 语义结果不存在: {args.semantic}")
        return
    sem_rows = load_rows(sem_files)
    print(f"[semantic] {len(sem_files)} 文件 → {len(sem_rows):,} 条语义结果（全量）")

    # 3. 逐岗位合并 Top3
    #    以语义全量为基底（每个岗位都有语义 Top），叠加规则置顶
    rule_only_ids = set(rule_rows.keys()) - set(sem_rows.keys())
    if rule_only_ids:
        print(f"  [warn] {len(rule_only_ids):,} 条岗位只在规则结果里、语义里没有"
              f"（正常应为 0：语义应全量覆盖）→ 这些岗位用纯规则 Top3")

    merged = 0          # 有规则命中并影响了 Top3 的岗位数
    rule_capped = 0     # 规则命中 ≥3、语义未补的岗位数
    dedup_dropped = 0   # 因去重被剔除的语义专业条次数

    with open(out_path, "w", encoding="utf-8") as f:
        for jid, sem_row in sem_rows.items():
            sem_top3 = parse_top3(sem_row)
            rule_top3 = parse_top3(rule_rows[jid]) if jid in rule_rows else []

            if rule_top3:
                merged += 1
                if len(rule_top3) >= TOP_K_STAGE2:
                    rule_capped += 1

            final_top3 = merge_top3(rule_top3, sem_top3)

            # 用语义行的原始 3 字段（岗位唯一标识/名称/全文），保证全量覆盖
            fields = {k: sem_row.get(k, "") for k in OUT_COLUMNS[:3]}
            f.write(json.dumps(build_row(fields, final_top3), ensure_ascii=False) + "\n")

        # 只在规则里、语义没有的岗位（理论上不该有，兜底）
        for jid in rule_only_ids:
            rule_top3 = parse_top3(rule_rows[jid])
            final_top3 = merge_top3(rule_top3, [])  # 无语义，纯规则
            fields = {k: rule_rows[jid].get(k, "") for k in OUT_COLUMNS[:3]}
            f.write(json.dumps(build_row(fields, final_top3), ensure_ascii=False) + "\n")

    total = len(sem_rows) + len(rule_only_ids)

    print(f"\n{'='*60}")
    print(f"  语义全量:     {len(sem_rows):>8,}")
    print(f"  其中规则命中: {merged:>8,}（Top3 含规则专业，score=1.0 置顶）")
    print(f"    └ 规则≥3纯规则: {rule_capped:>6,}（语义未补位）")
    print(f"  纯语义岗位:   {len(sem_rows)-merged:>8,}")
    print(f"  合计:         {total:>8,}")
    print(f"\n输出 → {out_path}")


if __name__ == "__main__":
    main()
