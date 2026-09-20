#!/usr/bin/env python3
"""第二步：只对岗位原始专业要求做标准专业名称精确匹配。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_DIR = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR.parents[2]
DEFAULT_EMBEDDING_TOP10 = NEW_TOP_DIR / "output" / "embedding_top10.csv"
DEFAULT_MAJORS = MODELS_DIR / "majors_intro.jsonl"
DEFAULT_OUTPUT = NEW_TOP_DIR / "output" / "rule_matches.csv"
OUTPUT_FIELDS = ("job_id", "major_requirement_raw", "rule_match_type", "rule_evidence", "major_code", "major_name")
GENERIC_REQUIREMENTS = {"", "不限", "不限专业", "专业不限", "无要求", "无专业要求", "不要求专业", "不作要求", "不限制", "无限制", "相关专业", "专业优先", "相关专业优先"}
SPLIT_RE = re.compile(r"[，,、；;|/\\\n\r]+|(?:以及|或者|或|和)")
DEGREE_PREFIX_RE = re.compile(r"^(?:(?:【|\[)?(?:博士|硕士|本科|大专|专科|高职)(?:】|\])?\s*[:：]?)")
TRAILING_RE = re.compile(r"(?:等)?(?:相关|相近|类似|对口)?专业(?:均可|皆可|即可|优先)?$|(?:均可|皆可|优先)$")

def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).strip(".。:：,，;；、")

def parse_requirement_tokens(value: object) -> list[str]:
    """只做确定性的清洗、拆分和完整标准专业名称匹配，不做小类/别名映射。"""
    text = str(value or "").strip()
    if not text:
        return []
    values: list[object] = [text]
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                values = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    tokens, seen = [], set()
    for value_item in values:
        normalized = unicodedata.normalize("NFKC", str(value_item or ""))
        normalized = re.sub(r"【(?:博士|硕士|本科|大专|专科|高职)】", "", normalized)
        for raw_part in SPLIT_RE.split(normalized):
            part = clean_text(raw_part)
            part = DEGREE_PREFIX_RE.sub("", part)
            part = TRAILING_RE.sub("", part)
            part = re.sub(r"[（(].*$", "", part)
            part = clean_text(part)
            if part and part not in GENERIC_REQUIREMENTS and part not in seen:
                seen.add(part)
                tokens.append(part)
    return tokens

def load_majors(path: Path) -> dict[str, dict[str, str]]:
    required = {"专业代码", "专业名称"}
    by_name, codes = {}, set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"专业源第{line_number}行缺少字段：{sorted(missing)}")
            major = {"major_code": clean_text(row["专业代码"]), "major_name": re.sub(r"\(略\)$", "", clean_text(row["专业名称"]))}
            if not all(major.values()):
                raise ValueError(f"专业源第{line_number}行存在空字段")
            if major["major_code"] in codes:
                raise ValueError(f"专业代码重复：{major['major_code']}")
            if major["major_name"] in by_name:
                raise ValueError(f"专业名称重复：{major['major_name']}")
            codes.add(major["major_code"])
            by_name[major["major_name"]] = major
    if len(by_name) != 744:
        raise ValueError(f"专业源应为744个专业，实际为{len(by_name)}")
    return by_name

def read_jobs(path: Path) -> list[tuple[str, str]]:
    jobs, seen = [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"job_id", "major_requirement_raw"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Embedding Top10缺少字段：{sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            job_id = clean_text(row.get("job_id"))
            if not job_id:
                raise ValueError(f"Embedding Top10第{row_number}行job_id为空")
            if job_id in seen:
                continue
            seen.add(job_id)
            jobs.append((job_id, str(row.get("major_requirement_raw") or "").strip()))
    return jobs

def job_sort_key(job_id: str) -> tuple[int, object]:
    return (0, int(job_id)) if job_id.isdigit() else (1, job_id)

def main() -> None:
    parser = argparse.ArgumentParser(description="只应用标准专业名称精确规则")
    parser.add_argument("--embedding-top10", default=str(DEFAULT_EMBEDDING_TOP10))
    parser.add_argument("--majors", default=str(DEFAULT_MAJORS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path, majors_path, output_path = map(lambda x: Path(x).resolve(), (args.embedding_top10, args.majors, args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"结果已存在：{output_path}；要重新生成请使用--overwrite")
    majors_by_name = load_majors(majors_path)
    majors_by_code = {m["major_code"]: m for m in majors_by_name.values()}
    jobs = read_jobs(input_path)
    rows, jobs_with_requirement, exact_jobs, matches_per_job = [], 0, set(), []
    for job_id, raw_requirement in jobs:
        tokens = parse_requirement_tokens(raw_requirement)
        if tokens:
            jobs_with_requirement += 1
        job_matches = defaultdict(set)
        for token in tokens:
            major = majors_by_name.get(token)
            if major:
                job_matches[major["major_code"]].add(token)
                exact_jobs.add(job_id)
        if job_matches:
            matches_per_job.append(len(job_matches))
        for code, evidences in job_matches.items():
            major = majors_by_code[code]
            rows.append({"job_id": job_id, "major_requirement_raw": raw_requirement, "rule_match_type": 1, "rule_evidence": "；".join(sorted(evidences)), "major_code": code, "major_name": major["major_name"]})
    rows.sort(key=lambda row: (job_sort_key(str(row["job_id"])), str(row["major_code"])))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader(); writer.writerows(rows)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists(): temporary.unlink()
    unmatched = jobs_with_requirement - len(exact_jobs)
    average = sum(matches_per_job) / len(matches_per_job) if matches_per_job else 0.0
    print(f"[jobs] total={len(jobs)} with_requirement={jobs_with_requirement}")
    print(f"[hit_jobs] exact={len(exact_jobs)} unmatched_with_requirement={unmatched}")
    print(f"[matches] rows={len(rows)} average_per_hit_job={average:.2f} max_per_job={max(matches_per_job, default=0)}")
    print(f"[done] {output_path}")

if __name__ == "__main__":
    main()
