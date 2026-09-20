#!/usr/bin/env python3
"""第三步：固定使用Embedding Top10，仅对其中的精确规则命中候选重排。"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_DIR = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR.parents[2]
DEFAULT_EMBEDDING_TOP10 = NEW_TOP_DIR / "output" / "embedding_top10.csv"
DEFAULT_RULE_MATCHES = NEW_TOP_DIR / "output" / "rule_matches.csv"
DEFAULT_MAJORS = MODELS_DIR / "models_scripts" / "majors_usedtoembed" / "new_majors.jsonl"
DEFAULT_OUTPUT = NEW_TOP_DIR / "output" / "final_top10.csv"
TOP_K = 10
EMBEDDING_FIELDS = {"job_id", "job_title_nlp", "major_requirement_raw", "semantic_rank", "major_code", "major_name", "embedding_score"}
RULE_FIELDS = {"job_id", "major_requirement_raw", "rule_match_type", "rule_evidence", "major_code", "major_name"}
OUTPUT_FIELDS = (
    "job_id", "job_title_nlp", "major_requirement_raw", "final_rank", "major_code", "major_name",
    "fusion_priority", "fusion_source", "semantic_rank", "embedding_score", "rule_match_type",
    "rule_evidence", "embedding_text",
)
SOURCE_LABELS = {1: "embedding_and_rule", 2: "embedding_only"}

def normalized(value: object) -> str:
    return str(value or "").strip()

def load_majors(path: Path) -> dict[str, dict[str, str]]:
    required = {"major_code", "major_name", "embedding_text"}
    majors, names = {}, set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"专业表第{line_number}行缺少字段：{sorted(missing)}")
            major = {field: normalized(row[field]) for field in required}
            if not all(major.values()):
                raise ValueError(f"专业表第{line_number}行存在空值")
            code, name = major["major_code"], major["major_name"]
            if code in majors:
                raise ValueError(f"专业代码重复：{code}")
            if name in names:
                raise ValueError(f"专业名称重复：{name}")
            majors[code] = major; names.add(name)
    if len(majors) != 744:
        raise ValueError(f"专业表应为744个专业，实际为{len(majors)}")
    return majors

def validate_major(row: dict[str, str], majors: dict[str, dict[str, str]], source: str, row_number: int) -> None:
    code, name = normalized(row.get("major_code")), normalized(row.get("major_name"))
    major = majors.get(code)
    if major is None:
        raise ValueError(f"{source}第{row_number}行专业代码无效：{code}")
    if major["major_name"] != name:
        raise ValueError(f"{source}第{row_number}行专业代码与名称不一致：{code}/{name}，标准名={major['major_name']}")

def load_rules(path: Path, majors: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, dict[str, str]]], int]:
    rules: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    raw_by_job, row_count = {}, 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = RULE_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"规则结果缺少字段：{sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            row_count += 1
            job_id, code = normalized(row.get("job_id")), normalized(row.get("major_code"))
            if not job_id:
                raise ValueError(f"规则结果第{row_number}行job_id为空")
            validate_major(row, majors, "规则结果", row_number)
            if normalized(row.get("rule_match_type")) != "1":
                raise ValueError(f"规则结果第{row_number}行只允许rule_match_type=1")
            if code in rules[job_id]:
                raise ValueError(f"规则结果存在重复岗位—专业：{job_id}/{code}")
            raw_requirement = normalized(row.get("major_requirement_raw"))
            previous_raw = raw_by_job.setdefault(job_id, raw_requirement)
            if previous_raw != raw_requirement:
                raise ValueError(f"规则结果同一job_id的major_requirement_raw不一致：{job_id}")
            rules[job_id][code] = {
                "rule_match_type": "1", "rule_evidence": normalized(row.get("rule_evidence")),
                "major_name": normalized(row.get("major_name")), "major_requirement_raw": raw_requirement,
            }
    return dict(rules), row_count

def candidate_sort_key(candidate: dict[str, object]) -> tuple:
    priority = int(candidate["fusion_priority"])
    return (
        priority,
        -float(candidate["embedding_score"]),
        int(candidate["semantic_rank"]),
        str(candidate["major_code"]),
    )

def merge_job(embedding_rows: list[dict[str, str]], rule_rows: dict[str, dict[str, str]], majors: dict[str, dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    job_id = normalized(embedding_rows[0]["job_id"])
    job_title = normalized(embedding_rows[0]["job_title_nlp"])
    raw_requirement = normalized(embedding_rows[0]["major_requirement_raw"])
    embedding_by_code: dict[str, dict[str, object]] = {}
    ranks: set[int] = set()
    for row in embedding_rows:
        if normalized(row["job_id"]) != job_id:
            raise ValueError("Embedding分组内部job_id不一致")
        if normalized(row["job_title_nlp"]) != job_title:
            raise ValueError(f"同一job_id的job_title_nlp不一致：{job_id}")
        if normalized(row["major_requirement_raw"]) != raw_requirement:
            raise ValueError(f"同一job_id的major_requirement_raw不一致：{job_id}")
        code = normalized(row["major_code"])
        if code in embedding_by_code:
            raise ValueError(f"Embedding Top10存在重复岗位—专业：{job_id}/{code}")
        try:
            semantic_rank, embedding_score = int(normalized(row["semantic_rank"])), float(normalized(row["embedding_score"]))
        except ValueError as error:
            raise ValueError(f"Embedding Top10排名或分数无效：{job_id}/{code}") from error
        ranks.add(semantic_rank)
        embedding_by_code[code] = {"semantic_rank": semantic_rank, "embedding_score": embedding_score}
    if ranks != set(range(1, TOP_K + 1)):
        raise ValueError(f"岗位{job_id}的Embedding排名不是完整1—{TOP_K}：{sorted(ranks)}")
    if rule_rows:
        rule_raw = next(iter(rule_rows.values()))["major_requirement_raw"]
        if rule_raw != raw_requirement:
            raise ValueError(f"Embedding与规则的major_requirement_raw不一致：{job_id}")

    candidates = []
    embedding_codes = set(embedding_by_code)
    inside_rule_codes = embedding_codes & set(rule_rows)
    outside_rule_codes = set(rule_rows) - embedding_codes
    for code, embedding in embedding_by_code.items():
        rule = rule_rows.get(code)
        priority = 1 if rule is not None else 2
        candidates.append({
            "job_id": job_id, "job_title_nlp": job_title, "major_requirement_raw": raw_requirement,
            "major_code": code, "major_name": majors[code]["major_name"],
            "fusion_priority": priority, "fusion_source": SOURCE_LABELS[priority],
            "semantic_rank": embedding["semantic_rank"], "embedding_score": embedding["embedding_score"],
            "rule_match_type": rule["rule_match_type"] if rule else "0",
            "rule_evidence": rule["rule_evidence"] if rule else "",
            "embedding_text": majors[code]["embedding_text"],
        })
    candidates.sort(key=candidate_sort_key)
    for final_rank, candidate in enumerate(candidates, 1):
        candidate["final_rank"] = final_rank
    stats = {
        "same_as_embedding_top10": int({str(row["major_code"]) for row in candidates} == embedding_codes),
        "reordered_jobs": int(bool(inside_rule_codes) and [str(row["major_code"]) for row in candidates] != [str(row["major_code"]) for row in embedding_rows]),
        "rule_hits_inside": len(inside_rule_codes),
        "exact_rules_outside_dropped": len(outside_rule_codes),
    }
    return candidates, stats

def main() -> None:
    parser = argparse.ArgumentParser(description="固定Embedding Top10，仅对其中的精确规则命中候选重排")
    parser.add_argument("--embedding-top10", default=str(DEFAULT_EMBEDDING_TOP10))
    parser.add_argument("--rule-matches", default=str(DEFAULT_RULE_MATCHES))
    parser.add_argument("--majors", default=str(DEFAULT_MAJORS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    embedding_path, rules_path, majors_path, output_path = map(lambda x: Path(x).resolve(), (args.embedding_top10, args.rule_matches, args.majors, args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"结果已存在：{output_path}；要重新生成请使用--overwrite")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists(): temporary.unlink()
    majors = load_majors(majors_path)
    rules_by_job, rule_input_rows = load_rules(rules_path, majors)
    print(f"[start] majors={len(majors)} rule_jobs={len(rules_by_job)} rule_rows={rule_input_rows}")
    jobs = output_rows = 0
    seen_jobs: set[str] = set()
    aggregate, source_rows, rule_type_rows, top5_source_rows = Counter(), Counter(), Counter(), Counter()
    try:
        with embedding_path.open("r", encoding="utf-8-sig", newline="") as source, temporary.open("w", encoding="utf-8-sig", newline="") as target:
            reader = csv.DictReader(source)
            missing = EMBEDDING_FIELDS - set(reader.fieldnames or [])
            if missing: raise ValueError(f"Embedding Top10缺少字段：{sorted(missing)}")
            writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS); writer.writeheader()
            enumerated = enumerate(reader, 2)
            for job_id, grouped in itertools.groupby(enumerated, key=lambda item: normalized(item[1].get("job_id"))):
                if not job_id: raise ValueError("Embedding Top10存在空job_id")
                if job_id in seen_jobs: raise ValueError(f"Embedding Top10同一job_id不连续或重复：{job_id}")
                seen_jobs.add(job_id)
                numbered_rows = list(grouped)
                if len(numbered_rows) != TOP_K: raise ValueError(f"岗位{job_id}的Embedding候选应为{TOP_K}行，实际为{len(numbered_rows)}")
                for row_number, row in numbered_rows: validate_major(row, majors, "Embedding Top10", row_number)
                selected, job_stats = merge_job([row for _, row in numbered_rows], rules_by_job.pop(job_id, {}), majors)
                writer.writerows(selected)
                jobs += 1; output_rows += len(selected); aggregate.update(job_stats)
                for row in selected:
                    source_rows[str(row["fusion_source"])] += 1; rule_type_rows[str(row["rule_match_type"])] += 1
                    if int(row["final_rank"]) <= 5: top5_source_rows[str(row["fusion_source"])] += 1
                if jobs % 10000 == 0: print(f"[progress] jobs={jobs} rows={output_rows}")
        if rules_by_job:
            raise ValueError(f"规则结果包含Embedding中不存在的job_id，例如：{sorted(rules_by_job)[:5]}")
        if output_rows != jobs * TOP_K: raise RuntimeError("最终结果行数不等于岗位数乘Top10")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists(): temporary.unlink()
    print(f"[jobs] total={jobs} output_rows={output_rows} top_k={TOP_K}")
    print(f"[checks] same_as_embedding_top10_jobs={aggregate['same_as_embedding_top10']} reordered_jobs={aggregate['reordered_jobs']} rule_hits_inside={aggregate['rule_hits_inside']} exact_rules_outside_dropped={aggregate['exact_rules_outside_dropped']}")
    print(f"[final_sources] {dict(sorted(source_rows.items()))}")
    print(f"[final_rule_types] {dict(sorted(rule_type_rows.items()))}")
    print(f"[top5_sources] {dict(sorted(top5_source_rows.items()))}")
    print(f"[done] {output_path}")

if __name__ == "__main__":
    main()
