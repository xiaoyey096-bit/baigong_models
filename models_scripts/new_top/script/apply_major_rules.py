#!/usr/bin/env python3
"""第二步：对原始专业要求应用精确规则与高置信模糊映射。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_DIR = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR.parents[2]
DEFAULT_EMBEDDING_TOP10 = NEW_TOP_DIR / "output" / "embedding_top10.csv"
DEFAULT_MAJORS = MODELS_DIR / "majors_intro.jsonl"
DEFAULT_MAPPING = NEW_TOP_DIR / "mapping" / "fuzzy_major_mapping.json"
DEFAULT_OUTPUT = NEW_TOP_DIR / "output" / "rule_matches.csv"

OUTPUT_FIELDS = (
    "job_id",
    "major_requirement_raw",
    "rule_match_type",
    "rule_evidence",
    "mapping_source",
    "mapping_confidence",
    "major_code",
    "major_name",
)
GENERIC_REQUIREMENTS = {
    "",
    "不限",
    "不限专业",
    "专业不限",
    "无要求",
    "无专业要求",
    "不要求专业",
    "不作要求",
    "不限制",
    "无限制",
    "相关专业",
    "专业优先",
    "相关专业优先",
}
SPLIT_RE = re.compile(r"[，,、；;|/\\\n\r]+|(?:以及|或者|或|和)")
DEGREE_PREFIX_RE = re.compile(
    r"^(?:(?:【|\[)?(?:博士|硕士|本科|大专|专科|高职)(?:】|\])?\s*[:：]?)"
)
TRAILING_RE = re.compile(
    r"(?:等)?(?:相关|相近|类似|对口)?专业(?:均可|皆可|即可|优先)?$|(?:均可|皆可|优先)$"
)


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", text).strip(".。:：,，;；、")


def parse_requirement_tokens(value: object) -> list[str]:
    """把专业要求拆成可做完整相等匹配的词，不做危险的子串命中。"""
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

    tokens: list[str] = []
    seen: set[str] = set()
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


def load_majors(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    required = {"专业代码", "专业名称", "所属小类名称"}
    by_name: dict[str, dict[str, str]] = {}
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    codes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"专业源第{line_number}行缺少字段：{sorted(missing)}")
            major = {
                "major_code": clean_text(row["专业代码"]),
                "major_name": re.sub(r"\(略\)$", "", clean_text(row["专业名称"])),
                "category_name": clean_text(row["所属小类名称"]),
            }
            if not all(major.values()):
                raise ValueError(f"专业源第{line_number}行存在空字段")
            if major["major_code"] in codes:
                raise ValueError(f"专业代码重复：{major['major_code']}")
            if major["major_name"] in by_name:
                raise ValueError(f"专业名称重复：{major['major_name']}")
            codes.add(major["major_code"])
            by_name[major["major_name"]] = major
            by_category[major["category_name"]].append(major)

    for majors in by_category.values():
        majors.sort(key=lambda item: item["major_code"])
    if len(by_name) != 744:
        raise ValueError(f"专业源应为744个专业，实际为{len(by_name)}")
    return by_name, dict(by_category)


def load_mapping(
    path: Path,
    majors_by_name: dict[str, dict[str, str]],
    majors_by_category: dict[str, list[dict[str, str]]],
) -> tuple[dict[str, dict], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("rules"), list):
        raise ValueError("模糊映射表schema_version或rules不合法")

    aliases: dict[str, dict] = {}
    for rule_number, rule in enumerate(data["rules"], 1):
        rule_aliases = [clean_text(item) for item in rule.get("aliases", [])]
        categories = [clean_text(item) for item in rule.get("target_categories", [])]
        major_names = [clean_text(item) for item in rule.get("target_major_names", [])]
        if not rule_aliases or not categories and not major_names:
            raise ValueError(f"模糊映射第{rule_number}条缺少别名或目标")
        if len(rule_aliases) != len(set(rule_aliases)):
            raise ValueError(f"模糊映射第{rule_number}条存在重复别名")
        if len(categories) != len(set(categories)) or len(major_names) != len(set(major_names)):
            raise ValueError(f"模糊映射第{rule_number}条存在重复目标")

        invalid_categories = sorted(set(categories) - set(majors_by_category))
        invalid_majors = sorted(set(major_names) - set(majors_by_name))
        if invalid_categories or invalid_majors:
            raise ValueError(
                f"模糊映射第{rule_number}条目标无效："
                f"小类={invalid_categories}，专业={invalid_majors}"
            )

        targets: dict[str, dict[str, str]] = {}
        for category in categories:
            for major in majors_by_category[category]:
                targets[major["major_code"]] = major
        for major_name in major_names:
            major = majors_by_name[major_name]
            targets[major["major_code"]] = major

        rule_data = {
            "targets": [targets[code] for code in sorted(targets)],
            "confidence": clean_text(rule.get("confidence", "high")) or "high",
        }
        for alias in rule_aliases:
            if alias in aliases:
                raise ValueError(f"别名重复映射到多条规则：{alias}")
            aliases[alias] = rule_data
    return aliases, str(data.get("mapping_version", ""))


def read_jobs(path: Path) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"job_id", "major_requirement_raw"}
        missing = required - set(reader.fieldnames or [])
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


def add_match(
    matches: dict[str, dict],
    major: dict[str, str],
    match_type: int,
    evidence: str,
    source: str,
    confidence: str,
) -> None:
    code = major["major_code"]
    current = matches.get(code)
    if current is None or match_type < current["rule_match_type"]:
        matches[code] = {
            "major_code": code,
            "major_name": major["major_name"],
            "rule_match_type": match_type,
            "evidence": {evidence},
            "sources": {source},
            "confidences": {confidence},
        }
    elif match_type == current["rule_match_type"]:
        current["evidence"].add(evidence)
        current["sources"].add(source)
        current["confidences"].add(confidence)


def job_sort_key(job_id: str) -> tuple[int, object]:
    return (0, int(job_id)) if job_id.isdigit() else (1, job_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="应用专业精确规则和高置信模糊映射")
    parser.add_argument("--embedding-top10", default=str(DEFAULT_EMBEDDING_TOP10))
    parser.add_argument("--majors", default=str(DEFAULT_MAJORS))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.embedding_top10).resolve()
    majors_path = Path(args.majors).resolve()
    mapping_path = Path(args.mapping).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"结果已存在：{output_path}；要重新生成请使用--overwrite")

    majors_by_name, majors_by_category = load_majors(majors_path)
    aliases, mapping_version = load_mapping(
        mapping_path, majors_by_name, majors_by_category
    )
    jobs = read_jobs(input_path)

    rows: list[dict[str, object]] = []
    jobs_with_requirement = 0
    exact_jobs: set[str] = set()
    category_jobs: set[str] = set()
    alias_jobs: set[str] = set()
    matched_jobs: set[str] = set()
    matches_per_job: list[int] = []
    alias_hit_counter: Counter[str] = Counter()

    for job_id, raw_requirement in jobs:
        tokens = parse_requirement_tokens(raw_requirement)
        if tokens:
            jobs_with_requirement += 1
        job_matches: dict[str, dict] = {}
        for token in tokens:
            exact_major = majors_by_name.get(token)
            if exact_major:
                add_match(
                    job_matches, exact_major, 1, token, "standard_major_name", "exact"
                )
                exact_jobs.add(job_id)

            category_majors = majors_by_category.get(token)
            if category_majors:
                for major in category_majors:
                    add_match(
                        job_matches, major, 2, token, "standard_category", "high"
                    )
                category_jobs.add(job_id)

            alias_rule = aliases.get(token)
            if alias_rule:
                for major in alias_rule["targets"]:
                    add_match(
                        job_matches,
                        major,
                        2,
                        token,
                        "curated_alias",
                        alias_rule["confidence"],
                    )
                alias_jobs.add(job_id)
                alias_hit_counter[token] += 1

        if job_matches:
            matched_jobs.add(job_id)
            matches_per_job.append(len(job_matches))
        for match in job_matches.values():
            rows.append(
                {
                    "job_id": job_id,
                    "major_requirement_raw": raw_requirement,
                    "rule_match_type": match["rule_match_type"],
                    "rule_evidence": "；".join(sorted(match["evidence"])),
                    "mapping_source": ";".join(sorted(match["sources"])),
                    "mapping_confidence": ";".join(sorted(match["confidences"])),
                    "major_code": match["major_code"],
                    "major_name": match["major_name"],
                }
            )

    rows.sort(
        key=lambda row: (
            job_sort_key(str(row["job_id"])),
            int(row["rule_match_type"]),
            str(row["major_code"]),
        )
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    unmatched = jobs_with_requirement - len(matched_jobs)
    average = sum(matches_per_job) / len(matches_per_job) if matches_per_job else 0.0
    maximum = max(matches_per_job, default=0)
    print(f"[mapping] version={mapping_version} aliases={len(aliases)}")
    print(f"[jobs] total={len(jobs)} with_requirement={jobs_with_requirement}")
    print(
        f"[hit_jobs] exact={len(exact_jobs)} standard_category={len(category_jobs)} "
        f"curated_alias={len(alias_jobs)} any={len(matched_jobs)} unmatched={unmatched}"
    )
    print(f"[matches] rows={len(rows)} average_per_hit_job={average:.2f} max_per_job={maximum}")
    print(f"[top_aliases] {alias_hit_counter.most_common(20)}")
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
