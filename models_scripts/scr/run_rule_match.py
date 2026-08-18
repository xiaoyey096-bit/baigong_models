#!/usr/bin/env python3
"""
Stage 0: 规则匹配（专业标准名子串匹配）→ 规则 Top3（供 merge 置顶）
=====================================================================
- 用 majors_optimized.jsonl 的 744 专业标准名，在岗位的「任职要求」∪「专业要求」
  字段里做子串匹配；命中的岗位产出 Top3（分=1.0）。
- 新逻辑（2026-08-14）：语义对【全量】岗位跑，规则不再「短路」语义。
  规则产出的 _rule.jsonl 在 merge_results.py 阶段做「置顶合并」——
  规则命中的专业 score=1.0 顶到 Top3 最前，语义按分补满到 3 个（专业去重）。
- _unmatched.jsonl 仍产出（向后兼容），但新流程下语义不再消费它。

设计依据（实测 30 万条采样）：
  · 任职要求：非空 71.9%，命中密度 16.5%（主力，多为「XX专业优先」标准表述，精度高）
  · 专业要求：非空 0.1%，但非空即标准专业名原文（`医生→中医学`），精度极高，零噪音
  · 岗位职责：命中是工作动作描述（皮肤护理/健康管理），误命中多，不用
  → 两字段或关系，覆盖率 ~11.9%

用法:
    # 标准用法：读 data_dwd/{range}岗位.jsonl（与 run_embed --range 一致）
    python run_rule_match.py --range "2026(01-06)_专科"

    # 测试用法：读任意目录下的 jsonl（--input，兼容开发测试）
    python run_rule_match.py --range test --input ../outputs_json_old/

    python run_rule_match.py --range "2026(01-06)_专科" --limit 10000   # 限量
    python run_rule_match.py --range "2026(01-06)_专科" --topn 3         # 每岗最多取几个专业(默认3)

输出（outputs_rule_{range_suffix}/ 下）:
    {range}_rule.jsonl       命中的岗位（12 列，专业X匹配分数=1.0）—— merge 置顶输入
    {range}_unmatched.jsonl  未命中的岗位（向后兼容保留，新流程语义不读）
"""

import argparse
import json
import time
from pathlib import Path

from config import *
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# 规则匹配用的字段（或关系：任一命中即可）
# ══════════════════════════════════════════════════════════════
RULE_FIELDS = ["任职要求", "专业要求"]


def load_major_names(path: Path):
    """读 majors_optimized.jsonl → [(code, name)]，按 name 长度降序（长名优先）。"""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            pairs.append((str(m.get("major_code", "")), m.get("major_name", "")))
    # 长名优先：避免"国际贸易"抢占"国际经济与贸易"
    pairs.sort(key=lambda x: len(x[1]), reverse=True)
    return pairs


def match_major_names(text: str, major_pairs, topn: int = 3):
    """在 text 里子串匹配专业名 → [(code, name)]，去重取前 topn。"""
    if not text:
        return []
    found = []
    seen_codes = set()
    for code, name in major_pairs:
        if not name:
            continue
        if name in text and code not in seen_codes:
            found.append((code, name))
            seen_codes.add(code)
            if len(found) >= topn:
                break
    return found


def build_rule_text(rec: dict) -> str:
    """拼接规则匹配用文本：任职要求 + 专业要求（两字段或关系 → 拼一起扫一次即可）。"""
    return " ".join((rec.get(f) or "").strip() for f in RULE_FIELDS)


# ══════════════════════════════════════════════════════════════
def resolve_input_files(args):
    """解析输入文件列表：--range 走 data_dwd/（回退 usedto_top3_data/）单文件；--input 走任意目录/文件。"""
    if args.range and not args.input:
        p, exists = resolve_range_file(args.range)
        return [p] if exists else []
    if args.input:
        p = Path(args.input)
        if p.is_dir():
            return sorted(p.glob("*.jsonl"))
        if p.is_file() and p.suffix == ".jsonl":
            return [p]
    return []


def main():
    p = argparse.ArgumentParser(description="Stage 0: 规则匹配（专业标准名子串）→ 快路径 Top3")
    p.add_argument("--range", required=True,
                   help="范围标识。默认读 usedto_top3_data/{range}岗位.jsonl，写 outputs_rule_{range}/。"
                        "如 --range 2026(01-06)_专科")
    p.add_argument("--input",
                   help="输入目录或文件（测试用，覆盖 --range 的默认输入路径）")
    p.add_argument("--limit", type=int, help="限量（每个输入文件最多读多少行）")
    p.add_argument("--topn", type=int, default=3, help="每个岗位最多取几个命中专业（默认3）")
    args = p.parse_args()

    rng = args.range
    rsuffix = rng.removesuffix("岗位")
    out_dir = BASE_DIR / f"outputs_rule_{rsuffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rule_path = out_dir / f"{rng}_rule.jsonl"
    unmatched_path = out_dir / f"{rng}_unmatched.jsonl"

    # 专业名
    major_pairs = load_major_names(MAJORS_FILE)
    print(f"[load] {len(major_pairs)} 专业标准名（长名优先）")

    # 输入
    in_files = resolve_input_files(args)
    if not in_files:
        print(f"[error] 输入不存在：--range {rng} / --input {args.input}")
        return
    print(f"[input] {len(in_files)} 个文件")

    # 统计
    total = 0
    hit = 0
    hit_counts = {1: 0, 2: 0, 3: 0}
    field_hit = {f: 0 for f in RULE_FIELDS}  # 各字段独立命中数（诊断用）

    t0 = time.time()
    with open(rule_path, "w", encoding="utf-8") as rf, \
         open(unmatched_path, "w", encoding="utf-8") as uf:
        pbar = tqdm(unit="条", bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for jp in in_files:
            pbar.set_description(f"[{jp.name[:20]}]")
            with open(jp, "r", encoding="utf-8") as f:
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
                    total += 1

                    # 诊断：各字段独立命中
                    for fld in RULE_FIELDS:
                        if match_major_names((rec.get(fld) or "").strip(), major_pairs, 1):
                            field_hit[fld] += 1

                    # 或关系：两字段拼一起扫
                    text = build_rule_text(rec)
                    matched = match_major_names(text, major_pairs, args.topn)

                    if matched:
                        hit += 1
                        hit_counts[len(matched)] = hit_counts.get(len(matched), 0) + 1
                        # 12 列输出，分数填 1.0
                        fields = extract_fields(rec)
                        top3 = [(code, name, 1.0) for code, name in matched]
                        rf.write(json.dumps(build_row(fields, top3), ensure_ascii=False) + "\n")
                    else:
                        # 未命中：原样全部字段写入，供 Stage1/2 语义路径读
                        uf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    pbar.update(1)
        pbar.close()

    rate = hit / total * 100 if total else 0
    print(f"\n{'='*60}")
    print(f"[完成] {time.time()-t0:.0f}s")
    print(f"  总计:       {total:,}")
    print(f"  规则命中:   {hit:,} ({rate:.1f}%)")
    print(f"  未命中:     {total-hit:,} ({100-rate:.1f}%)")
    print(f"  命中数分布: " + " ".join(f"{k}个→{v:,}" for k, v in sorted(hit_counts.items())))
    print(f"  字段独立命中(诊断): " + " ".join(f"{k}={v:,}" for k, v in field_hit.items()))
    print(f"\n输出:")
    print(f"  命中   → {rule_path}")
    print(f"  未命中 → {unmatched_path}")


if __name__ == "__main__":
    main()
