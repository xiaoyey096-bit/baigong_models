#!/usr/bin/env python3
"""
Stage 0 v2: 规则匹配（AC 自动机: 标准名精确 + 泛称映射表）→ 规则集（供并集）
======================================================================
- pyahocorasick 一次扫描「任职要求 ∪ 专业要求」，三类词全进自动机:
    专业标准名(1.0) + 泛称映射表别名(0.9，映射到类下全部专业)
- 分数即来源标签: 1.0=精确 / 0.9=泛称 / 语义通道是 0.5~0.85 原始分
- 输出长表（岗位×专业一行），与 run_topn.py --margin 的长表同构，
  merge_topn.py 做并集。

前置: alias_table.json 已生成并人工抽查过（build_alias_table.py）。
      没有该文件时自动退化为纯标准名匹配（等价旧版语义）。

用法:
    python run_rule_match_v2.py --range "2026(01-06)_专科"

输出: outputs_rule_{suffix}/{rng}_rule_v2.jsonl  长表
      {job_id, title, code, name, score, source_tag}
"""

import argparse
import json
import time
from pathlib import Path

from config import *

RULE_FIELDS = ["任职要求", "专业要求"]
ALIAS_PATH = BASE_DIR / "majors_usedtoembed" / "alias_table.json"


def build_automaton(major_pairs, alias_rows):
    """AC 自动机: value = (score, [codes])。标准名 value=[自身code]，泛称 value=类 codes。"""
    try:
        import ahocorasick
    except ImportError:
        sys.exit("[error] 未安装 pyahocorasick: pip install pyahocorasick")
    A = ahocorasick.Automaton()
    for code, name in major_pairs:
        if name and len(name) >= 2:
            A.add_word(name, (SCORE_RULE_EXACT, [code], name))
    n_alias = 0
    for row in alias_rows or []:
        alias, codes = row["alias"], row["codes"]
        if alias and len(alias) >= 2 and len(codes) >= 2:   # 只映射 1 个专业的别名无增量
            A.add_word(alias, (SCORE_RULE_FUZZY, codes, alias))
            n_alias += 1
    A.make_automaton()
    return A, n_alias


def scan(A, text, majors_by_code, topn_cap=TOP_N_CAP):
    """一次扫描 → [(code, name, score)]，精确优先去重，封顶 topn_cap。"""
    exact, fuzzy = set(), set()
    for _, (score, codes, _) in A.iter(text):
        if score >= SCORE_RULE_EXACT:
            exact.update(codes)
        else:
            fuzzy.update(codes)
    fuzzy -= exact
    out = [(c, majors_by_code[c]["major_name"], SCORE_RULE_EXACT) for c in exact]
    out += [(c, majors_by_code[c]["major_name"], SCORE_RULE_FUZZY) for c in fuzzy]
    return out[:topn_cap] if topn_cap else out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--range", required=True)
    p.add_argument("--input", help="输入目录/文件（测试用）")
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    import sys
    rng = args.range
    suffix = rng.removesuffix("岗位")

    majors = load_majors(MAJORS_FILE)
    majors_by_code = {m["major_code"]: m for m in majors}
    major_pairs = [(m["major_code"], m["major_name"]) for m in majors]
    major_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    alias_rows = None
    if ALIAS_PATH.exists():
        alias_rows = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
        print(f"[alias] {len(alias_rows)} 泛称词")
    else:
        print(f"[alias] 无 alias_table.json，退化为纯标准名匹配")

    A, n_alias = build_automaton(major_pairs, alias_rows)
    print(f"[automaton] {len(major_pairs)} 标准名 + {n_alias} 泛称词")

    if args.input:
        ip = Path(args.input)
        in_files = sorted(ip.glob("*.jsonl")) if ip.is_dir() else [ip]
    else:
        fp, exists = resolve_range_file(rng)
        if not exists:
            sys.exit(f"[error] 输入不存在: {fp}")
        in_files = [fp]

    out_dir = BASE_DIR / f"outputs_rule_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rng}_rule_v2.jsonl"

    from tqdm import tqdm
    t0 = time.time()
    total = hit = n_rows = 0
    with open(out_path, "w", encoding="utf-8") as of:
        pbar = tqdm(unit="条", bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for jp in in_files:
            pbar.set_description(f"[{jp.name[:20]}]")
            with open(jp, encoding="utf-8") as f:
                for line in f:
                    if args.limit and total >= args.limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    total += 1
                    text = " ".join((rec.get(fld) or "").strip() for fld in RULE_FIELDS)
                    matched = scan(A, text, majors_by_code) if text else []
                    if matched:
                        hit += 1
                        fields = extract_fields(rec)
                        for code, name, score in matched:
                            of.write(json.dumps({
                                "job_id": fields["岗位唯一标识"],
                                "title": fields["原始岗位名称"],
                                "code": code, "name": name, "score": score,
                            }, ensure_ascii=False) + "\n")
                            n_rows += 1
                    pbar.update(1)
        pbar.close()

    rate = hit / total * 100 if total else 0
    print(f"\n[完成] {time.time()-t0:.0f}s  总计 {total:,}  命中 {hit:,} ({rate:.1f}%)  "
          f"规则行 {n_rows:,} (均 {n_rows/max(hit,1):.2f} 专业/岗)")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
