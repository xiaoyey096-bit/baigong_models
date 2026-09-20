#!/usr/bin/env python3
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--majors", required=True)
    args = parser.parse_args()
    major_codes = {
        json.loads(line)["major_code"]
        for line in Path(args.majors).open(encoding="utf-8")
        if line.strip()
    }
    counts = Counter()
    ranks = defaultdict(set)
    pairs = set()
    jobs = set()
    bad_rows = 0
    rows = 0
    with Path(args.input).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        for row in reader:
            rows += 1
            job_id = row["job_id"]
            code = row["major_code"]
            rank = int(row["embedding_rank"])
            jobs.add(job_id)
            counts[job_id] += 1
            ranks[job_id].add(rank)
            pair = (job_id, code)
            bad_rows += int(
                code not in major_codes
                or pair in pairs
                or not 0 <= float(row["embedding_score"]) <= 1
            )
            pairs.add(pair)
    bad_jobs = sum(
        counts[job_id] != 20 or ranks[job_id] != set(range(1, 21))
        for job_id in jobs
    )
    result = {
        "jobs": len(jobs),
        "rows": rows,
        "bad_jobs": bad_jobs,
        "bad_rows": bad_rows,
        "file_size_bytes": os.path.getsize(args.input),
        "fields": fields,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["jobs"] != 100_000 or result["rows"] != 2_000_000:
        raise SystemExit("岗位数或结果行数不符合预期")
    if bad_jobs or bad_rows:
        raise SystemExit("Top20 完整性校验失败")


if __name__ == "__main__":
    main()
