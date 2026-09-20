#!/usr/bin/env python3
"""从 PostgreSQL dwd.v_job_core 直接分批读取岗位并生成固定Top10。

当前脚本是数据库输入端到Top10计算端的正式入口；结果暂时只做终端抽样输出。
目标结果表结构确认后，再增加数据库结果sink，不把岗位源数据落成中间JSON/CSV。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterator

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_DIR = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_major_rules import clean_text, load_majors as load_rule_majors, parse_requirement_tokens  # noqa: E402
from merge_final_top10 import merge_job  # noqa: E402
from run_embedding_top10 import (  # noqa: E402
    ALLOWED_EDUCATION,
    TOP_K,
    Embedder,
    build_job_embedding_text,
    choose_device,
    load_majors as load_embedding_majors,
)

DEFAULT_MODEL = MODELS_DIR / "bge-large-zh-v1.5"
DEFAULT_EMBEDDING_MAJORS = MODELS_DIR / "models_scripts" / "majors_usedtoembed" / "new_majors.jsonl"
DEFAULT_RULE_MAJORS = MODELS_DIR / "majors_intro.jsonl"
DEFAULT_VIEW = "dwd.v_job_core"
JSONB_FIELDS = {"job_description_std", "job_requirement_std", "ability_tags"}


def database_connection(args):
    try:
        import psycopg2
    except ImportError as error:
        raise RuntimeError("缺少psycopg2，请安装数据库驱动") from error

    password = os.getenv("TOPN_DB_PASSWORD")
    if password is None:
        raise RuntimeError("未设置TOPN_DB_PASSWORD，密码只能通过环境变量或平台密钥注入")
    connection = psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        dbname=args.db_name,
        user=args.db_user,
        password=password,
        sslmode=args.db_sslmode,
        connect_timeout=args.db_connect_timeout,
    )
    connection.set_session(
        readonly=True,
        autocommit=False,
        isolation_level="READ COMMITTED",
    )
    return connection


def build_query(args) -> tuple[str, list[object]]:
    clauses = ["education_level_std = ANY(%s)"]
    params: list[object] = [args.education_level]
    if args.crawl_date_from:
        clauses.append("crawl_date >= %s")
        params.append(args.crawl_date_from)
    if args.crawl_date_to:
        clauses.append("crawl_date < %s")
        params.append(args.crawl_date_to)
    clauses.append("COALESCE(job_title_nlp, job_title_std, '') <> ''")
    limit_sql = ""
    if args.max_jobs is not None:
        limit_sql = "LIMIT %s"
        params.append(args.max_jobs)
    query = f"""
        SELECT
            job_id,
            education_level_std,
            job_title_std,
            major_requirement_raw,
            job_description_std,
            job_requirement_std,
            ability_tags,
            job_title_nlp
        FROM {DEFAULT_VIEW}
        WHERE {' AND '.join(clauses)}
        ORDER BY job_id
        {limit_sql}
    """
    return query, params


def iter_job_batches(args) -> Iterator[list[dict]]:
    """使用服务端游标分批读取，不把全量岗位加载到内存。"""
    connection = database_connection(args)
    cursor = connection.cursor(name="topn_job_reader")
    cursor.itersize = args.batch_size
    query, params = build_query(args)
    cursor.execute(query, params)
    previous_job_id = None
    try:
        while True:
            fetched = cursor.fetchmany(args.batch_size)
            if not fetched:
                break
            batch: list[dict] = []
            for values in fetched:
                row = dict(zip((
                    "job_id", "education_level_std", "job_title_std",
                    "major_requirement_raw", "job_description_std",
                    "job_requirement_std", "ability_tags", "job_title_nlp",
                ), values))
                job_id = clean_text(row.get("job_id"))
                if not job_id:
                    raise ValueError("dwd.v_job_core返回空job_id")
                if previous_job_id is not None and job_id <= previous_job_id:
                    raise ValueError(f"job_id未保持唯一递增顺序，发现重复或乱序：{job_id}")
                previous_job_id = job_id
                # 数据库第二层保险：程序再次严格筛选大专/不限。
                if clean_text(row.get("education_level_std")) not in ALLOWED_EDUCATION:
                    continue
                batch.append(row)
            if batch:
                yield batch
    finally:
        cursor.close()
        connection.rollback()
        connection.close()


def jsonb_as_text(value: object) -> object:
    """把psycopg2解码后的JSONB数组转成现有文本构造器可解析的JSON字符串。"""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def normalize_job_row(row: dict) -> dict:
    normalized = dict(row)
    for field in JSONB_FIELDS:
        normalized[field] = jsonb_as_text(normalized.get(field))
    normalized["job_title_nlp"] = clean_text(
        normalized.get("job_title_nlp") or normalized.get("job_title_std")
    )
    normalized["major_requirement_raw"] = str(
        normalized.get("major_requirement_raw") or ""
    ).strip()
    return normalized


def build_exact_rules(raw_requirement: object, rule_majors: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    rules: dict[str, dict[str, str]] = {}
    for token in parse_requirement_tokens(raw_requirement):
        major = rule_majors.get(token)
        if major is None:
            continue
        code = major["major_code"]
        if code not in rules:
            rules[code] = {
                "rule_match_type": "1",
                "rule_evidence": token,
                "major_name": major["major_name"],
                "major_requirement_raw": str(raw_requirement or "").strip(),
            }
        else:
            evidence = {item for item in rules[code]["rule_evidence"].split("；") if item}
            evidence.add(token)
            rules[code]["rule_evidence"] = "；".join(sorted(evidence))
    return rules


def process_batch(
    rows: list[dict],
    embedder: Embedder,
    major_vectors: np.ndarray,
    embedding_majors: list[dict],
    rule_majors: dict[str, dict[str, str]],
) -> list[list[dict[str, object]]]:
    normalized_rows = [normalize_job_row(row) for row in rows]
    texts = [build_job_embedding_text(row) for row in normalized_rows]
    valid = [(row, text) for row, text in zip(normalized_rows, texts) if text]
    if not valid:
        return []
    job_vectors = embedder.encode([text for _, text in valid], is_query=True)
    scores = job_vectors @ major_vectors.T
    results: list[list[dict[str, object]]] = []
    for index, (job, _) in enumerate(valid):
        # 先用argpartition取候选前10，再只对10条做稳定排序，避免全量岗位对744个专业完整排序。
        partition_indices = np.argpartition(-scores[index], TOP_K - 1)[:TOP_K]
        ranked_indices = sorted(
            partition_indices.tolist(),
            key=lambda major_index: (-float(scores[index, major_index]), embedding_majors[major_index]["major_code"]),
        )
        embedding_rows = [
            {
                "job_id": job["job_id"],
                "job_title_nlp": job["job_title_nlp"],
                "major_requirement_raw": job["major_requirement_raw"],
                "semantic_rank": str(rank),
                "major_code": embedding_majors[major_index]["major_code"],
                "major_name": embedding_majors[major_index]["major_name"],
                "embedding_score": f"{float(scores[index, major_index]):.6f}",
            }
            for rank, major_index in enumerate(ranked_indices, 1)
        ]
        rule_rows = build_exact_rules(job["major_requirement_raw"], rule_majors)
        merged, _ = merge_job(embedding_rows, rule_rows, {
            major["major_code"]: major for major in embedding_majors
        })
        results.append(merged)
    return results


def print_result_sample(results: list[list[dict[str, object]]], max_jobs: int) -> None:
    for rows in results[:max_jobs]:
        if not rows:
            continue
        first = rows[0]
        print(f"[job] {first['job_id']} | {first['job_title_nlp']} | raw_major={first['major_requirement_raw']}")
        for row in rows:
            print(
                f"  {row['final_rank']:>2}. {row['major_name']} "
                f"score={row['embedding_score']} source={row['fusion_source']} "
                f"rule={row['rule_match_type']} evidence={row['rule_evidence']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="直接从PostgreSQL View读取岗位并计算固定Top10")
    parser.add_argument("--db-host", default=os.getenv("TOPN_DB_HOST"))
    parser.add_argument("--db-port", default=os.getenv("TOPN_DB_PORT", "5432"))
    parser.add_argument("--db-name", default=os.getenv("TOPN_DB_NAME", "baigongdb"))
    parser.add_argument("--db-user", default=os.getenv("TOPN_DB_USER"))
    parser.add_argument("--db-sslmode", default=os.getenv("TOPN_DB_SSLMODE", "prefer"))
    parser.add_argument("--db-connect-timeout", type=int, default=10)
    parser.add_argument("--education-level", action="append", default=None, choices=sorted(ALLOWED_EDUCATION))
    parser.add_argument("--crawl-date-from", type=date.fromisoformat)
    parser.add_argument("--crawl-date-to", type=date.fromisoformat)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-jobs", type=int, help="测试时限制岗位数；正式全量运行必须显式指定--confirm-full-run")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--print-jobs", type=int, default=5)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--embedding-majors", default=str(DEFAULT_EMBEDDING_MAJORS))
    parser.add_argument("--rule-majors", default=str(DEFAULT_RULE_MAJORS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    args = parser.parse_args()

    if not args.db_host or not args.db_user:
        parser.error("请通过--db-host/--db-user或TOPN_DB_HOST/TOPN_DB_USER提供数据库连接信息")
    if args.max_jobs is None and not args.confirm_full_run:
        parser.error("为避免误触发600万级全量任务，测试请指定--max-jobs；全量运行请显式加--confirm-full-run")
    if args.batch_size <= 0 or args.embedding_batch_size <= 0:
        parser.error("batch-size必须为正数")
    if args.crawl_date_from and args.crawl_date_to and args.crawl_date_from >= args.crawl_date_to:
        parser.error("crawl-date-from必须早于crawl-date-to")
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("max-jobs必须为正数")

    if args.education_level is None:
        args.education_level = sorted(ALLOWED_EDUCATION)
    device = choose_device(args.device)
    embedding_majors = load_embedding_majors(Path(args.embedding_majors).resolve())
    rule_majors = load_rule_majors(Path(args.rule_majors).resolve())
    embedder = Embedder(Path(args.model).resolve(), device, args.embedding_batch_size)
    major_vectors = embedder.encode([major["embedding_text"] for major in embedding_majors], is_query=False)

    stats = Counter()
    printed = 0
    for batch in iter_job_batches(args):
        stats["db_rows_after_python_filter"] += len(batch)
        results = process_batch(batch, embedder, major_vectors, embedding_majors, rule_majors)
        stats["processed_jobs"] += len(results)
        stats["output_rows"] += sum(len(rows) for rows in results)
        stats["rule_hit_pairs_inside_top10"] += sum(
            1 for rows in results for row in rows if row["rule_match_type"] == "1"
        )
        if printed < args.print_jobs:
            print_result_sample(results, args.print_jobs - printed)
            printed += min(len(results), args.print_jobs - printed)
        if stats["processed_jobs"] % 1000 == 0:
            print(f"[progress] processed_jobs={stats['processed_jobs']} output_rows={stats['output_rows']}")
    print(f"[done] {dict(stats)} device={device} top_k={TOP_K}")


if __name__ == "__main__":
    main()
