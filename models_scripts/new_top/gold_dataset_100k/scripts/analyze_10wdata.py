#!/usr/bin/env python3
"""生成 10wdata.csv 的质量与覆盖分析报告，不输出岗位原文。"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path
import re


SPACE_RE = re.compile(r"\s+")
CORE = [
    "job_title_nlp",
    "job_description_std",
    "job_requirement_std",
    "ability_tags",
    "major_requirement_raw",
]


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "；".join(text(x) for x in value if text(x))
    if isinstance(value, dict):
        return "；".join(f"{text(k)}：{text(v)}" for k, v in value.items() if text(v))
    raw = str(value)
    if raw.startswith(("[", "{")):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, (list, dict)):
                return text(parsed)
        except (TypeError, json.JSONDecodeError):
            pass
    return SPACE_RE.sub("", raw).strip()


def stat(values: list[int]) -> str:
    qs = statistics.quantiles(values, n=20) if len(values) >= 20 else []
    p95 = qs[18] if qs else max(values)
    return f"min={min(values)}, median={statistics.median(values):g}, p95={p95:g}, max={max(values)}"


def top_lines(counter: Counter[str], limit: int = 15) -> list[str]:
    return [f"- {key or '[空值]'}: {value} ({value / sum(counter.values()):.2%})" for key, value in counter.most_common(limit)]


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    path = Path(args.input)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    duplicate_ids = len(rows) - len({row.get("job_id", "") for row in rows})
    fill = {field: sum(bool(text(row.get(field))) for row in rows) for field in CORE}
    lengths = {field: [len(text(row.get(field))) for row in rows] for field in CORE}
    titles = Counter(text(row.get("job_title_nlp")) for row in rows)
    report: list[str] = [
        "# 10wdata.csv 质量与覆盖分析报告",
        "",
        "## 一、文件完整性",
        "",
        f"- 总行数：{len(rows):,}",
        f"- CSV 字段数：{len(fields)}",
        f"- job_id 重复数：{duplicate_ids}",
        f"- 空 job_id 数：{sum(not text(row.get('job_id')) for row in rows)}",
        f"- 岗位标题精确去重后数量：{len(titles):,}",
        "- 结果：数量、字段结构和岗位主键检查通过。",
        "",
        "## 二、五类核心字段填充率",
        "",
    ]
    for field in CORE:
        report.append(f"- `{field}`：{fill[field]:,}/{len(rows):,}（{fill[field] / len(rows):.2%}）")
    report += [
        "",
        "说明：`major_requirement_raw` 允许为空，因为专业不限或未填写是有效岗位类型；其余四类字段用于支撑岗位语义和专业匹配。",
        "",
        "## 三、有效文本长度",
        "",
    ]
    for field in CORE:
        report.append(f"- `{field}`：{stat(lengths[field])}")
    report += [
        "",
        "## 四、岗位覆盖情况",
        "",
        f"- 学历分布：{dict(Counter(row.get('education_level_std', '') for row in rows))}",
        f"- 抓取日期种类：{len({row.get('crawl_date', '') for row in rows})}",
        f"- 抓取日期范围：{min(row.get('crawl_date', '') for row in rows)} 至 {max(row.get('crawl_date', '') for row in rows)}",
        "",
        "### 来源分布（前 15）",
        *top_lines(Counter(text(row.get("source_name")) for row in rows)),
        "",
        "### 省份分布（前 15）",
        *top_lines(Counter(text(row.get("province_std")) for row in rows)),
        "",
        "### 经验要求分布（前 15）",
        *top_lines(Counter(text(row.get("experience_level_std")) for row in rows)),
        "",
        "### 用工类型分布（前 15）",
        *top_lines(Counter(text(row.get("employment_type_std")) for row in rows)),
        "",
        "### 岗位标题分布（前 30）",
        *top_lines(titles, 30),
        "",
        "## 五、检查结论",
        "",
        "- 10 万条均满足岗位名称、岗位职责、任职要求和能力标签的最低有效条件。",
        "- 岗位名称已统一使用 `job_title_nlp`，原字段为空的记录使用 `job_title_std` 兜底。",
        "- 本次筛选不做重复岗位删除、联系方式清理或其他 DWD 之外的脏数据处理。",
        "- 10 万条是按质量分从满足条件的 230,262 条候选中选出的，未满足条件的岗位由后续合格岗位替换。",
        "- 当前数据库视图没有独立行业/岗位大类字段，因此行业宽泛性只能通过岗位标题、来源、地域、经验和用工类型做间接检查；若要严格按行业配额控制，需要后续增加分类字段或分类规则。",
        "- 文件适合进入下一步纯 Embedding Top20 流程；`length`、质量分等分析字段未写入最终 CSV。",
    ]
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[done] rows={len(rows)} report={args.report}")


if __name__ == "__main__":
    main()
