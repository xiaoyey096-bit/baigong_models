#!/usr/bin/env python3
"""
统计每个专业的匹配岗位数量，输出 xlsx 汇总表。

输入: models/majors_data/{月份范围}/majors_data_dwd_all/*.jsonl
输出: models/majors_data/专业匹配岗位统计_{月份范围}.xlsx
      列: 专业代码, 专业名称, 匹配岗位数量

用法:
    python stat_by_major.py --range 2026(01-06)
"""

import argparse
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

BASE = Path("/home/cl-715/workspace/models/majors_data")


def main():
    ap = argparse.ArgumentParser(description="专业匹配岗位数统计（按月份范围）")
    ap.add_argument("--range", required=True,
                    help="月份范围标识，如 2026(01-06)；读 majors_data/{range}/majors_data_dwd_all/")
    args = ap.parse_args()

    rng = args.range
    SRC_DIR = BASE / rng / "majors_data_dwd_all"
    OUT     = BASE / f"专业匹配岗位统计_{rng}.xlsx"

    if not SRC_DIR.exists():
        print(f"[error] 输入目录不存在: {SRC_DIR}")
        print(f"        请先跑 Stage 3 (build_by_major.py --output-range {rng})")
        return

    print(f"[config] 月份范围: {rng}")
    print(f"[config] 输入: {SRC_DIR}")
    print(f"[config] 输出: {OUT}\n")

    rows = []
    for jp in sorted(SRC_DIR.glob("*.jsonl")):
        # 文件名格式: {专业代码}_{专业名称}.jsonl
        stem = jp.stem
        if "_" in stem:
            code, name = stem.split("_", 1)
        else:
            code, name = stem, ""
        n = sum(1 for _ in open(jp, "r", encoding="utf-8"))
        rows.append((code, name, n))

    # 按岗位数降序
    rows.sort(key=lambda x: -x[2])
    total = sum(r[2] for r in rows)

    # 写 xlsx
    wb = Workbook()
    ws = wb.active
    ws.title = "专业匹配岗位统计"

    # 表头
    headers = ["专业代码", "专业名称", "匹配岗位数量"]
    header_fill = PatternFill("solid", fgColor="4472C4")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    # 数据
    for r, (code, name, n) in enumerate(rows, 2):
        ws.cell(row=r, column=1, value=code)
        ws.cell(row=r, column=2, value=name)
        ws.cell(row=r, column=3, value=n)

    # 合计行
    total_row = len(rows) + 2
    ws.cell(row=total_row, column=1, value="合计").font = Font(bold=True)
    ws.cell(row=total_row, column=3, value=total).font = Font(bold=True)

    # 列宽
    ws.column_dimensions["A"].width = 14
    ws.column_dimensions["B"].width = 32
    ws.column_dimensions["C"].width = 16

    wb.save(str(OUT))

    print(f"[done] {len(rows)} 个专业, 共 {total:,} 条岗位 → {OUT}")
    print("\nTop10 岗位最多的专业:")
    for code, name, n in rows[:10]:
        print(f"  {code} {name:25s} {n:>8,}")
    print(f"\nBottom5 岗位最少的专业:")
    for code, name, n in rows[-5:]:
        print(f"  {code} {name:25s} {n:>8,}")


if __name__ == "__main__":
    main()
