#!/usr/bin/env python3
"""统计每个专业匹配的岗位数（≥0.6），输出到 models/"""
import json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *

# 收集所有 JSONL
dirs = [
    Path("/home/cl-715/workspace/models/models_scripts/outputs_json"),
    Path("/home/cl-715/workspace/models/models_scripts/outputs_json_cloud"),
]

# 按专业统计
major_counts = defaultdict(int)
total_jobs = 0
matched_jobs = 0

for d in dirs:
    for fp in sorted(d.glob("*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: r = json.loads(line)
                except: continue
                total_jobs += 1
                has_match = False
                for i in range(1, 4):
                    score = r.get(f"专业{i}匹配分数", 0)
                    code = r.get(f"专业{i}代码", "")
                    if score >= 0.6 and code:
                        major_counts[code] += 1
                        has_match = True
                if has_match:
                    matched_jobs += 1

# 加载专业名称
majors = load_majors(MAJORS_FILE)
major_name = {m["major_code"]: m["major_name"] for m in majors}

# 按岗位数降序排列
rows = [(code, major_name.get(code, "未知"), cnt) for code, cnt in major_counts.items()]
rows.sort(key=lambda x: -x[2])

# 补全 744 个专业中未出现的（计数为0）
seen = set(r[0] for r in rows)
for m in majors:
    if m["major_code"] not in seen:
        rows.append((m["major_code"], m["major_name"], 0))

# 输出
out_dir = Path("/home/cl-715/workspace/models")
out_dir.mkdir(parents=True, exist_ok=True)

# XLSX
from openpyxl import Workbook
wb = Workbook(write_only=True)
ws = wb.create_sheet("统计")
ws.append(["专业代码", "专业名称", "匹配岗位数"])
for code, name, cnt in rows:
    ws.append([code, name, cnt])
xl = out_dir / "统计_专业匹配岗位数.xlsx"
wb.save(str(xl))

# 汇总
has_jobs = sum(1 for _, _, c in rows if c > 0)
zero = sum(1 for _, _, c in rows if c == 0)
grand = sum(c for _, _, c in rows)
print(f"总岗位: {total_jobs:,}")
print(f"至少1个匹配(≥0.6)的岗位: {matched_jobs:,} ({matched_jobs/max(total_jobs,1)*100:.1f}%)")
print(f"匹配 pair 总数: {grand:,}")
print(f"有岗位的专业: {has_jobs}/744")
print(f"无岗位的专业: {zero}/744")
print(f"\nTop20 专业:")
for code, name, cnt in rows[:20]:
    print(f"  {code} {name}: {cnt:,}")
print(f"\n输出: {xl}")
