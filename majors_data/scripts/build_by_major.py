#!/usr/bin/env python3
"""
按专业分组构建岗位库（单一全量库）。

═══════════════════════════════════════════════════════════════
命名规范（详见 工作流方案.md）：
  · 输入：outputs_merged_{range}/*.jsonl（合并后，可多个范围段合并）
  · 输出：majors_data/{月份范围}/majors_data_dwd_all/{专业代码}_{专业名称}.jsonl
    月份范围 = 累积汇总标识，只标月份、不带爬取日期
    例：majors_data/2026(01-06)/majors_data_dwd_all/

用法：
  python build_by_major.py \
      --input /path/outputs_merged_2026(01-08)_专科/2026(01-08)_专科_merged.jsonl \
      --output-range "2026(01-08)"

  # 合并多个范围段 → 汇总成累积口径（跨文件按 job_id 去重）
  python build_by_major.py \
      --input 范围段A.jsonl \
      --input 范围段B.jsonl \
      --output-range 2026(01-08)

输出 1 类（在 majors_data/{月份范围}/majors_data_dwd_all/ 下）：
  {专业代码}_{专业名称}.jsonl   全部岗位按匹配专业分组（不做质量分层）

字段 (6 列):
  岗位唯一标识 | 原始岗位名称 | 岗位全文 | 匹配专业代码 | 匹配专业名称 | 匹配专业分数
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

BASE      = Path("/home/cl-715/workspace/models")
MAJORS    = BASE / "models_scripts" / "majors_usedtoembed" / "majors_optimized.jsonl"
MAJORS_DATA_ROOT = BASE / "majors_data"


def resolve_input_paths(inputs):
    """把 --input 参数解析成实际要读的 jsonl 文件列表。

    每项可以是：
      · outputs_merged_{range}/ 下的文件
      · 直接的目录绝对/相对路径（收集其下所有 jsonl）
      · 单个 jsonl 文件
    """
    files = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file() and p.suffix == ".jsonl":
            files.append(p)
        else:
            print(f"[warn] 输入不存在，跳过: {inp}")
    return files


def main():
    ap = argparse.ArgumentParser(description="按专业分组构建岗位库（单一全量库，不做质量分层）")
    ap.add_argument("--input", action="append", required=True,
                    help="输入 jsonl 文件或目录（可多次指定以合并）")
    ap.add_argument("--output-range", required=True,
                    help="输出的月份范围标识，如 2026(01-08)；产物写到 majors_data/{此值}/ 下")
    args = ap.parse_args()

    out_range = args.output_range
    DWD_DIR = MAJORS_DATA_ROOT / out_range / "majors_data_dwd_all"

    print(f"[config] 输出月份范围: {out_range}")
    print(f"[config] 输出目录:     {DWD_DIR}")

    # 1. 专业代码 → 名称 映射
    code2name = {}
    with open(MAJORS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            code2name[str(m["major_code"])] = m["major_name"]
    print(f"[load] {len(code2name)} 专业")

    # 2. 解析输入
    job_files = resolve_input_paths(args.input)
    if not job_files:
        print("[error] 没有找到任何输入 jsonl，检查 --input 参数")
        return
    print(f"[input] {len(job_files)} 个 jsonl 文件")

    # 3. 全量分组收集器：code -> [row]
    dwd_by_major = defaultdict(list)
    total_in = 0
    seen_ids = set()   # 跨文件去重（合并多范围段时同一岗位可能出现多次）

    for jp in job_files:
        with open(jp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                job_id = r.get("岗位唯一标识", "")
                if job_id in seen_ids:
                    continue
                seen_ids.add(job_id)
                total_in += 1

                title = r.get("原始岗位名称", "")
                full = (r.get("岗位全文") or "").strip()

                # 反透视：Top3 每个专业各产出一条，全量归入对应专业文件（不分层）
                for i in range(1, 4):
                    code = r.get(f"专业{i}代码", "")
                    if not code:
                        continue
                    name = r.get(f"专业{i}名称", "")
                    score = r.get(f"专业{i}匹配分数", "")
                    try:
                        score = round(float(score), 4)
                    except (TypeError, ValueError):
                        score = ""
                    canonical = code2name.get(str(code), name)
                    dwd_by_major[str(code)].append({
                        "岗位唯一标识": job_id,
                        "原始岗位名称": title,
                        "岗位全文": full,
                        "匹配专业代码": str(code),
                        "匹配专业名称": canonical,
                        "匹配专业分数": score,
                    })

    # 4. 写全量库
    DWD_DIR.mkdir(parents=True, exist_ok=True)
    for old in DWD_DIR.glob("*.jsonl"):   # 清旧文件，防专业改名残留
        old.unlink()
    for code, rows in sorted(dwd_by_major.items()):
        name = code2name.get(code, "未知")
        with open(DWD_DIR / f"{code}_{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 统计
    dwd_total = sum(len(v) for v in dwd_by_major.values())
    print(f"\n[stat] 输入岗位行(去重后): {total_in:,}")
    print(f"  全量分组: {dwd_total:,} 条 ({len(dwd_by_major)} 专业文件)")
    print(f"\n[done] → {DWD_DIR}")


if __name__ == "__main__":
    main()
