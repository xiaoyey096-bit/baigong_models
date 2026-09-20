#!/usr/bin/env python3
"""Validate local-LLM Top10 labels against the corresponding Embedding Top20."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    args = parser.parse_args()

    records = []
    targets = set()
    with args.labels.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            item = json.loads(line)
            records.append(item)
            targets.add(str(item["job_id"]))

    candidates: dict[str, set[str]] = {job_id: set() for job_id in targets}
    with args.embedding.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            job_id = str(row.get("job_id") or "")
            if job_id in candidates:
                candidates[job_id].add(str(row.get("major_code") or "").strip())

    errors = []
    status = Counter()
    for item in records:
        job_id = str(item.get("job_id"))
        labels = item.get("llm_top10") or []
        codes = [str(label.get("major_code") or "").strip() for label in labels]
        status[str(item.get("parse_status"))] += 1
        if len(labels) != 10:
            errors.append((job_id, "label_count", len(labels)))
        if len(set(codes)) != len(codes):
            errors.append((job_id, "duplicate_code", codes))
        if not set(codes).issubset(candidates.get(job_id, set())):
            errors.append((job_id, "outside_top20", sorted(set(codes) - candidates.get(job_id, set()))))
        if len(candidates.get(job_id, set())) != 20:
            errors.append((job_id, "candidate_count", len(candidates.get(job_id, set()))))

    print(f"records={len(records)}")
    print(f"candidate_jobs={len(candidates)}")
    print(f"status={dict(status)}")
    print(f"errors={len(errors)}")
    for error in errors[:20]:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
