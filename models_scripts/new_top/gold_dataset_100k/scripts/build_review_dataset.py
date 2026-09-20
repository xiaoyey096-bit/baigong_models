#!/usr/bin/env python3
"""Build human-review CSV and JSONL views from the gold dataset outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


JOB_FIELDS = [
    "job_id", "education_level_std", "crawl_date", "publish_time", "source_name",
    "province_std", "city_std", "experience_level_std", "employment_type_std",
    "job_title_std", "job_title_nlp", "job_description_std", "job_requirement_std",
    "ability_tags", "major_requirement_raw",
]


def load_jobs(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["job_id"]: {field: row.get(field, "") for field in JOB_FIELDS} for row in csv.DictReader(handle)}


def load_top20(path: Path) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            job_id = row["job_id"]
            result.setdefault(job_id, []).append({
                "embedding_rank": int(row["embedding_rank"]),
                "major_code": row["major_code"],
                "major_name": row["major_name"],
                "embedding_score": float(row["embedding_score"]),
            })
    return result


def load_labels(path: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                result[item["job_id"]] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--top20", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    args = parser.parse_args()

    jobs = load_jobs(args.jobs)
    top20 = load_top20(args.top20)
    labels = load_labels(args.labels)
    job_ids = list(labels)
    missing = [job_id for job_id in job_ids if job_id not in jobs or job_id not in top20]
    if missing:
        raise RuntimeError(f"missing source data for {len(missing)} jobs: {missing[:3]}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = JOB_FIELDS + [
        "embedding_rank", "embedding_score", "llm_rank", "major_code", "major_name",
        "parse_status", "model", "human_review_status", "human_correct", "human_note",
    ]
    with args.csv.open("w", encoding="utf-8-sig", newline="") as csv_handle, args.jsonl.open("w", encoding="utf-8") as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for job_id in job_ids:
            job = jobs[job_id]
            label = labels[job_id]
            top20_items = top20[job_id]
            enriched = {
                **job,
                "embedding_top20": top20_items,
                "llm_top10": label.get("llm_top10", []),
                "parse_status": label.get("parse_status", ""),
                "model": label.get("model", ""),
                "raw_output": label.get("raw_output", ""),
                "human_review_status": "待审核",
                "human_correct_top10": "",
                "human_note": "",
            }
            jsonl_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            labels_by_code = {item["major_code"]: item for item in label.get("llm_top10", [])}
            for candidate in label.get("llm_top10", []):
                candidate_code = candidate["major_code"]
                writer.writerow({
                    **job,
                    "embedding_rank": candidate.get("embedding_rank", ""),
                    "embedding_score": candidate.get("embedding_score", ""),
                    "llm_rank": candidate.get("llm_rank", ""),
                    "major_code": candidate_code,
                    "major_name": candidate.get("major_name", ""),
                    "parse_status": label.get("parse_status", ""),
                    "model": label.get("model", ""),
                    "human_review_status": "待审核",
                    "human_correct": "",
                    "human_note": "",
                })
    print(f"jobs={len(job_ids)}")
    print(f"csv_rows={len(job_ids) * 10}")
    print(f"csv={args.csv}")
    print(f"jsonl={args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
