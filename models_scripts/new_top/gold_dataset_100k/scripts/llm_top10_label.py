#!/usr/bin/env python3
"""Use a local Qwen3/vLLM endpoint to label Top10 majors from Embedding Top20."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPT = """你是高职教育专业匹配专家。请根据岗位信息，在给定的20个候选专业中，选出与岗位最相关的10个专业，并按相关性从高到低排序。
判断依据：岗位名称、工作内容、任职要求、能力标签和专业要求；优先考虑能直接支撑岗位核心工作内容的专业，同时考虑岗位明确要求的专业。
硬性要求：只能选择候选列表中的专业代码；必须返回恰好10个不重复的专业代码；第1个最相关；只输出一个JSON对象，不要输出解释、Markdown或其它文字。
JSON格式：{{"ranked_major_codes":["专业代码1","专业代码2","专业代码3","专业代码4","专业代码5","专业代码6","专业代码7","专业代码8","专业代码9","专业代码10"]}}

岗位信息：
{job_text}

Embedding Top20候选专业：
{candidate_text}
"""


def clip(value: str, limit: int) -> str:
    value = (value or "").strip()
    return value if len(value) <= limit else value[:limit] + "……"


def field_text(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return "；".join(str(x).strip() for x in parsed if str(x).strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return str(value).strip()


def load_jobs(path: Path, max_jobs: int | None) -> tuple[dict[str, dict[str, str]], list[str]]:
    jobs: dict[str, dict[str, str]] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            job_id = (row.get("job_id") or "").strip()
            if not job_id or job_id in jobs:
                continue
            if max_jobs is not None and len(order) >= max_jobs:
                break
            jobs[job_id] = row
            order.append(job_id)
    return jobs, order


def iter_embedding_groups(path: Path, selected: set[str]):
    current_id: str | None = None
    current: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            job_id = (row.get("job_id") or "").strip()
            if not job_id:
                continue
            if current_id is None:
                current_id = job_id
            if job_id != current_id:
                if current_id in selected:
                    yield current_id, current
                current_id = job_id
                current = []
            current.append(row)
        if current_id in selected:
            yield current_id, current


def make_job_text(row: dict[str, str]) -> str:
    return "\n".join(
        [
            f"岗位名称：{clip(row.get('job_title_nlp') or row.get('job_title_std') or '', 120)}",
            f"岗位描述：{clip(field_text(row.get('job_description_std', '')), 1400)}",
            f"任职要求：{clip(field_text(row.get('job_requirement_std', '')), 1400)}",
            f"能力标签：{clip(field_text(row.get('ability_tags', '')), 600)}",
            f"专业要求：{clip(field_text(row.get('major_requirement_raw', '')), 600)}",
        ]
    )


def make_prompt(job: dict[str, str], candidates: list[dict[str, str]]) -> str:
    candidate_text = "\n".join(
        f"{i}. code={c.get('major_code','')} name={c.get('major_name','')}"
        for i, c in enumerate(candidates, 1)
    )
    return DEFAULT_PROMPT.format(job_text=make_job_text(job), candidate_text=candidate_text)


def request_model(api_url: str, model: str, prompt: str, timeout: int, retries: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 192,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        api_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return str(result["choices"][0]["message"].get("content") or "")
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"model request failed: {last_error}")


def extract_values(raw: str) -> list[Any]:
    objects: list[dict[str, Any]] = []
    for match in re.findall(r"\{.*?\}", raw or "", flags=re.DOTALL):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                objects.append(parsed)
        except json.JSONDecodeError:
            continue
    for obj in objects:
        for key in ("ranked_major_codes", "major_codes", "top10", "top_10", "majors"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
    return re.findall(r"(?<!\d)\d{4,8}(?!\d)", raw or "")


def normalize_codes(raw: str, candidates: list[dict[str, str]]) -> tuple[list[str], str]:
    by_code = {str(c.get("major_code", "")).strip(): str(c.get("major_code", "")).strip() for c in candidates}
    by_name = {str(c.get("major_name", "")).strip(): str(c.get("major_code", "")).strip() for c in candidates}
    by_position = {
        str(c.get("embedding_rank", "")).strip(): str(c.get("major_code", "")).strip()
        for c in candidates
    }
    selected: list[str] = []
    normalized_index = False
    for value in extract_values(raw):
        text = str(value).strip()
        code = by_code.get(text) or by_name.get(text)
        if not code and text in by_position:
            code = by_position[text]
            normalized_index = True
        if code and code not in selected:
            selected.append(code)
    if len(selected) == 10:
        status = "model_index_normalized" if normalized_index else "model"
    else:
        status = "model_repaired"
    for candidate in candidates:
        code = str(candidate.get("major_code", "")).strip()
        if code and code not in selected:
            selected.append(code)
        if len(selected) == 10:
            break
    return selected[:10], status


def build_record(job_id: str, candidates: list[dict[str, str]], raw: str, model: str) -> dict[str, Any]:
    codes, status = normalize_codes(raw, candidates)
    by_code = {str(c.get("major_code", "")).strip(): c for c in candidates}
    top10 = []
    for rank, code in enumerate(codes, 1):
        candidate = by_code[code]
        top10.append(
            {
                "llm_rank": rank,
                "major_code": code,
                "major_name": candidate.get("major_name", ""),
                "embedding_rank": int(candidate.get("embedding_rank", rank)),
                "embedding_score": float(candidate.get("embedding_score", 0.0)),
            }
        )
    return {
        "job_id": job_id,
        "model": model,
        "candidate_count": len(candidates),
        "parse_status": status,
        "llm_top10": top10,
        "raw_output": clip(raw, 2000),
    }


def chunks(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flat-output", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="/home/cl-715/workspace/dwd/NLP/models/Qwen3-4B-AWQ")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        for output in (args.output, args.flat_output):
            if output.exists():
                output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.flat_output.parent.mkdir(parents=True, exist_ok=True)

    jobs, order = load_jobs(args.jobs, args.max_jobs)
    selected = set(order)
    groups = {job_id: candidates for job_id, candidates in iter_embedding_groups(args.embedding, selected)}
    missing = [job_id for job_id in order if job_id not in groups]
    if missing:
        raise RuntimeError(f"missing embedding Top20 for {len(missing)} jobs; first={missing[:3]}")

    done: set[str] = set()
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if item.get("job_id"):
                        done.add(str(item["job_id"]))
                except json.JSONDecodeError:
                    continue
    pending = [job_id for job_id in order if job_id not in done]
    print(f"jobs={len(order)} pending={len(pending)} candidates={len(groups[order[0]])}", flush=True)

    jsonl_needs_header = not args.output.exists() or args.output.stat().st_size == 0
    csv_needs_header = not args.flat_output.exists() or args.flat_output.stat().st_size == 0
    with args.output.open("a", encoding="utf-8") as jsonl, args.flat_output.open("a", encoding="utf-8", newline="") as flat_handle:
        flat = csv.writer(flat_handle)
        if csv_needs_header:
            flat.writerow(["job_id", "llm_rank", "major_code", "major_name", "embedding_rank", "embedding_score", "parse_status", "model"])
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            completed = 0
            for batch in chunks(pending, args.concurrency):
                futures = {
                    job_id: pool.submit(
                        request_model,
                        args.api_url,
                        args.model,
                        make_prompt(jobs[job_id], groups[job_id]),
                        args.timeout,
                        args.retries,
                    )
                    for job_id in batch
                }
                raw_outputs = {job_id: futures[job_id].result() for job_id in batch}
                retry_jobs = [
                    job_id
                    for job_id in batch
                    if normalize_codes(raw_outputs[job_id], groups[job_id])[1] == "model_repaired"
                ]
                if retry_jobs:
                    strict_suffix = (
                        "\n\n严格复核：你刚才的输出包含了不在候选列表中的值。请重新检查并只复制候选列表中"
                        " code= 后面的完整专业代码，绝对不要输出候选序号、专业类别代码或自造代码。"
                        "仍然只返回恰好10个不重复代码的JSON对象。"
                    )
                    retry_futures = {
                        job_id: pool.submit(
                            request_model,
                            args.api_url,
                            args.model,
                            make_prompt(jobs[job_id], groups[job_id]) + strict_suffix,
                            args.timeout,
                            args.retries,
                        )
                        for job_id in retry_jobs
                    }
                    for job_id in retry_jobs:
                        retry_raw = retry_futures[job_id].result()
                        if normalize_codes(retry_raw, groups[job_id])[1] != "model_repaired":
                            raw_outputs[job_id] = retry_raw
                for job_id in batch:
                    raw = raw_outputs[job_id]
                    record = build_record(job_id, groups[job_id], raw, args.model)
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    for label in record["llm_top10"]:
                        flat.writerow([
                            job_id,
                            label["llm_rank"],
                            label["major_code"],
                            label["major_name"],
                            label["embedding_rank"],
                            label["embedding_score"],
                            record["parse_status"],
                            args.model,
                        ])
                    completed += 1
                jsonl.flush()
                flat_handle.flush()
                if completed % max(1, args.concurrency * 4) == 0 or completed == len(pending):
                    print(f"completed={completed}/{len(pending)}", flush=True)
    print(f"wrote={args.output}", flush=True)
    print(f"wrote={args.flat_output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
