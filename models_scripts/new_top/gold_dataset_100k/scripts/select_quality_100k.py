#!/usr/bin/env python3
"""从 dwd.v_job_core 筛选适合岗位—专业 Embedding 的高质量岗位。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator


CORE_FIELDS = (
    "job_title_nlp",
    "job_description_std",
    "job_requirement_std",
    "ability_tags",
    "major_requirement_raw",
)
OUTPUT_FIELDS = (
    "job_id",
    "education_level_std",
    "crawl_date",
    "publish_time",
    "source_name",
    "province_std",
    "city_std",
    "experience_level_std",
    "employment_type_std",
    "job_title_std",
    *CORE_FIELDS,
)
ALLOWED_EDUCATION = ("大专", "不限")
SPACE_RE = re.compile(r"\s+")


def compact(value: Any) -> str:
    """把 JSONB/list/dict/普通文本统一为可计数的紧凑文本。"""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "；".join(compact(item) for item in value if compact(item))
    elif isinstance(value, dict):
        text = "；".join(
            f"{compact(key)}：{compact(item)}" for key, item in value.items() if compact(item)
        )
    else:
        text = str(value)
    return SPACE_RE.sub("", text).strip()


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def band_score(length: int, minimum: int, ideal_low: int, ideal_high: int, maximum: int) -> float:
    """文本在理想长度区间得满分，过短或冗长时平滑降分。"""
    if length < minimum or length > maximum:
        return 0.0
    if ideal_low <= length <= ideal_high:
        return 1.0
    if length < ideal_low:
        return (length - minimum) / max(1, ideal_low - minimum)
    return (maximum - length) / max(1, maximum - ideal_high)


def score_row(row: dict[str, Any]) -> tuple[float, dict[str, int]] | None:
    title = compact(row.get("job_title_nlp") or row.get("job_title_std"))
    description = compact(row.get("job_description_std"))
    requirement = compact(row.get("job_requirement_std"))
    ability = compact(row.get("ability_tags"))
    major_requirement = compact(row.get("major_requirement_raw"))
    lengths = {
        "title_length": len(title),
        "description_length": len(description),
        "requirement_length": len(requirement),
        "ability_length": len(ability),
        "major_requirement_length": len(major_requirement),
    }

    # 岗位名称、岗位职责、任职要求是支撑专业判断的最低信息条件。
    if not (2 <= lengths["title_length"] <= 50):
        return None
    if not (80 <= lengths["description_length"] <= 3000):
        return None
    if not (50 <= lengths["requirement_length"] <= 2500):
        return None
    if lengths["ability_length"] < 1:
        return None

    nonempty = sum(
        bool(text) for text in (title, description, requirement, ability, major_requirement)
    )
    combined = sum(lengths.values())
    if combined < 100 or combined > 5000:
        return None

    score = 0.0
    score += 15.0 * band_score(lengths["title_length"], 2, 3, 24, 50)
    score += 30.0 * band_score(lengths["description_length"], 40, 100, 900, 3000)
    score += 30.0 * band_score(lengths["requirement_length"], 20, 60, 700, 2500)
    score += 8.0 * band_score(lengths["ability_length"], 0, 5, 240, 1000)
    score += 7.0 * band_score(lengths["major_requirement_length"], 0, 2, 160, 800)
    score += 5.0 * (nonempty / 5.0)
    score += 5.0 * band_score(combined, 100, 180, 1800, 5000)
    return round(score, 4), {**lengths, "nonempty_core_fields": nonempty}


def stable_tiebreak(job_id: str) -> int:
    return int.from_bytes(hashlib.sha256(job_id.encode("utf-8")).digest()[:8], "big")


def connect(args: argparse.Namespace):
    import psycopg2

    password = os.environ.get("TOPN_DB_PASSWORD")
    if not password:
        raise RuntimeError("缺少 TOPN_DB_PASSWORD")
    connection = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=password,
        connect_timeout=10,
        application_name="new_top_quality_selector",
    )
    connection.set_session(readonly=True, autocommit=False, isolation_level="READ COMMITTED")
    return connection


def iter_rows(args: argparse.Namespace) -> Iterator[dict[str, Any]]:
    query = """
        SELECT
            job_id,
            education_level_std,
            crawl_date,
            publish_time,
            source_name,
            province_std,
            city_std,
            experience_level_std,
            employment_type_std,
            job_title_std,
            job_title_nlp,
            job_description_std,
            job_requirement_std,
            ability_tags,
            major_requirement_raw
        FROM dwd.v_job_core
        WHERE education_level_std = ANY(%s)
          AND COALESCE(NULLIF(BTRIM(job_title_nlp), ''), NULLIF(BTRIM(job_title_std), '')) IS NOT NULL
        ORDER BY job_id
    """
    connection = connect(args)
    cursor = connection.cursor(name="quality_job_reader")
    cursor.itersize = args.batch_size
    cursor.execute(query, (list(ALLOWED_EDUCATION),))
    # Named cursor 在部分 psycopg2/代理组合下不会立即暴露 description；
    # SELECT 字段固定，显式维护列顺序也更稳妥。
    columns = [
        "job_id",
        "education_level_std",
        "crawl_date",
        "publish_time",
        "source_name",
        "province_std",
        "city_std",
        "experience_level_std",
        "employment_type_std",
        "job_title_std",
        "job_title_nlp",
        "job_description_std",
        "job_requirement_std",
        "ability_tags",
        "major_requirement_raw",
    ]
    try:
        while True:
            batch = cursor.fetchmany(args.batch_size)
            if not batch:
                break
            for values in batch:
                yield dict(zip(columns, values))
    finally:
        cursor.close()
        connection.rollback()
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", default=os.getenv("TOPN_DB_HOST", "192.168.111.40"))
    parser.add_argument("--db-port", default=os.getenv("TOPN_DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.getenv("TOPN_DB_NAME", "baigongdb"))
    parser.add_argument("--db-user", default=os.getenv("TOPN_DB_USER", "baigong"))
    parser.add_argument("--target-count", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"结果已存在：{output}；如需覆盖请添加 --overwrite")

    heap: list[tuple[float, int, int, dict[str, Any], dict[str, int]]] = []
    scanned = eligible = 0
    sequence = 0
    for row in iter_rows(args):
        scanned += 1
        scored = score_row(row)
        if scored is None:
            continue
        eligible += 1
        score, features = scored
        job_id = compact(row.get("job_id"))
        if not job_id:
            continue
        item = (score, stable_tiebreak(job_id), sequence, row, features)
        sequence += 1
        if len(heap) < args.target_count:
            heapq.heappush(heap, item)
        elif item[:2] > heap[0][:2]:
            heapq.heapreplace(heap, item)
        if scanned % 100_000 == 0:
            cutoff = heap[0][0] if heap else 0.0
            print(
                f"[progress] scanned={scanned} eligible={eligible} "
                f"selected={len(heap)} cutoff={cutoff:.4f}",
                flush=True,
            )

    if len(heap) < args.target_count:
        raise RuntimeError(
            f"满足最低质量条件的岗位只有 {len(heap)} 条，少于目标 {args.target_count} 条"
        )

    selected = sorted(heap, key=lambda item: (-item[0], -item[1]))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for score, _, _, row, features in selected:
            effective_title = row.get("job_title_nlp") or row.get("job_title_std")
            record = {field: csv_value(row.get(field)) for field in OUTPUT_FIELDS}
            record["job_title_nlp"] = csv_value(effective_title)
            writer.writerow(record)
    os.replace(temporary, output)

    scores = [item[0] for item in selected]
    print(f"[done] scanned={scanned} eligible={eligible} selected={len(selected)}")
    print(
        f"[scores] min={min(scores):.4f} max={max(scores):.4f} "
        f"mean={sum(scores) / len(scores):.4f}"
    )
    print(f"[output] {output}")


if __name__ == "__main__":
    main()
