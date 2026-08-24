#!/usr/bin/env python3
"""第一步：全量岗位与专业Embedding，输出每岗固定语义Top10。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

import numpy as np
from openpyxl import load_workbook


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_DIR = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR.parents[2]
DEFAULT_JOBS = MODELS_DIR / "hbb" / "高质量10w样本_专科清洗.csv"
DEFAULT_MAJORS = MODELS_DIR / "models_scripts" / "majors_usedtoembed" / "new_majors.xlsx"
DEFAULT_MODEL = MODELS_DIR / "bge-large-zh-v1.5"
DEFAULT_OUTPUT = NEW_TOP_DIR / "output" / "embedding_top10.csv"

TOP_K = 10
MAX_LENGTH = 512
EMBEDDING_DIMENSION = 1024
ALLOWED_EDUCATION = {"大专", "不限"}
QUERY_INSTRUCTION = (
    "Given a job posting, retrieve the most relevant college major "
    "(higher vocational education specialty)"
)
JOB_EMBEDDING_FIELDS = (
    "major_requirement_raw",
    "job_title_nlp",
    "job_description_std",
    "job_requirement_std",
    "ability_tags",
)
OUTPUT_FIELDS = (
    "job_id",
    "job_title_nlp",
    "major_requirement_raw",
    "semantic_rank",
    "major_code",
    "major_name",
    "embedding_score",
)

GENERIC_MAJOR_REQUIREMENT_RE = re.compile(
    r"^(?:专业)?(?:不限|无要求|不作要求|不限制|无限制|无专业要求|不要求专业)(?:专业)?$"
)
GENERIC_ABILITY_RE = re.compile(
    r"^(?:良好的?|较强的?|强烈的?)?"
    r"(?:团队协作|团队合作|团精神|沟通|沟通协调|协调|表达|"
    r"执行|抗压|责任心|学习|逻辑思维|吃苦耐劳|主动性|细心|耐心|"
    r"服务意识|问题处理)(?:能力|精神|意识|强|力)?$"
)


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_listish(value: object) -> list[str]:
    text = normalize(value)
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [normalize(item) for item in parsed if normalize(item)]
        except (json.JSONDecodeError, TypeError):
            pass
    return [text]


def clean_major_requirement(value: object) -> list[str]:
    cleaned: list[str] = []
    for raw in parse_listish(value):
        for part in re.split(r"[，,、；;/|]+", raw):
            item = normalize(part).strip(".。：:")
            compact = re.sub(r"\s+", "", item)
            if not compact or GENERIC_MAJOR_REQUIREMENT_RE.fullmatch(compact):
                continue
            if compact in {"专业", "相关专业", "专业优先", "相关专业优先"}:
                continue
            if item not in cleaned:
                cleaned.append(item)
    return cleaned


def clean_ability_tags(value: object) -> list[str]:
    cleaned: list[str] = []
    for item in parse_listish(value):
        compact = re.sub(r"[\s，,、；;]+", "", item)
        if compact and not GENERIC_ABILITY_RE.fullmatch(compact) and item not in cleaned:
            cleaned.append(item)
    return cleaned


def build_job_embedding_text(row: dict) -> str:
    title = normalize(row.get("job_title_nlp") or row.get("job_title_std"))
    field_values = (
        ("专业要求", clean_major_requirement(row.get("major_requirement_raw"))),
        ("岗位名称", [title] if title else []),
        ("岗位职责", parse_listish(row.get("job_description_std"))),
        ("任职要求", parse_listish(row.get("job_requirement_std"))),
        ("专业能力", clean_ability_tags(row.get("ability_tags"))),
    )
    return "\n".join(f"{label}：{'；'.join(values)}" for label, values in field_values if values)


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_majors(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        values = sheet.iter_rows(values_only=True)
        headers = [normalize(value) for value in next(values)]
        raw_rows = [dict(zip(headers, row)) for row in values]
        workbook.close()
    elif path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            raw_rows = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("专业文件只支持.xlsx或.jsonl")

    required = {"major_code", "major_name", "embedding_text"}
    majors: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows, 2):
        missing = required - set(row)
        if missing:
            raise ValueError(f"专业表第{index}行缺少字段：{sorted(missing)}")
        major = {field: normalize(row.get(field)) for field in required}
        if not all(major.values()):
            raise ValueError(f"专业表第{index}行存在空值")
        if major["major_code"] in seen:
            raise ValueError(f"专业代码重复：{major['major_code']}")
        seen.add(major["major_code"])
        majors.append(major)
    if len(majors) < TOP_K:
        raise ValueError(f"专业数量小于Top{TOP_K}")
    return majors


def eligible_jobs(path: Path) -> Iterator[dict]:
    required = {
        "job_id",
        "education_level_std",
        "job_title_std",
        *JOB_EMBEDDING_FIELDS,
    }
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"岗位CSV缺少字段：{sorted(missing)}")
        for input_row, row in enumerate(reader, 2):
            if normalize(row.get("education_level_std")) not in ALLOWED_EDUCATION:
                continue
            job_id = normalize(row.get("job_id"))
            if not job_id:
                raise ValueError(f"岗位CSV第{input_row}行job_id为空")
            if job_id in seen:
                raise ValueError(f"job_id重复：{job_id}")
            seen.add(job_id)
            text = build_job_embedding_text(row)
            if not text:
                continue
            yield {
                "job_id": job_id,
                "job_title_nlp": normalize(row.get("job_title_nlp") or row.get("job_title_std")),
                "major_requirement_raw": normalize(row.get("major_requirement_raw")),
                "job_embedding_text": text,
            }


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self, model_path: Path, device: str, batch_size: int):
        import torch
        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
        self.model = SentenceTransformer(str(model_path), device=device, model_kwargs=model_kwargs)
        self.model.max_seq_length = MAX_LENGTH
        try:
            self.model.prompts = {"query": QUERY_INSTRUCTION, "passage": ""}
        except Exception:
            pass
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        kwargs = {
            "normalize_embeddings": True,
            "batch_size": self.batch_size,
            "show_progress_bar": False,
        }
        try:
            vectors = self.model.encode(
                texts,
                prompt_name="query" if is_query else "passage",
                **kwargs,
            )
        except (TypeError, ValueError, KeyError):
            inputs = [f"{QUERY_INSTRUCTION}: {text}" for text in texts] if is_query else texts
            vectors = self.model.encode(inputs, **kwargs)
        array = np.asarray(vectors, dtype="float32")
        if array.ndim != 2 or array.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"向量维度异常：{array.shape}")
        return array


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="全量岗位Embedding Top10")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--majors", default=str(DEFAULT_MAJORS))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto", help="auto/cuda/cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs_path = Path(args.jobs).resolve()
    majors_path = Path(args.majors).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    major_vectors_path = output_path.with_suffix(output_path.suffix + ".major_vectors.npy")
    major_cache_path = output_path.with_suffix(output_path.suffix + ".major_vectors.json")
    temporary_paths = (part_path, progress_path, major_vectors_path, major_cache_path)

    if args.overwrite:
        for path in (output_path, *temporary_paths):
            if path.exists():
                path.unlink()
    elif output_path.exists():
        raise SystemExit(f"结果已存在：{output_path}；要重新计算请使用--overwrite")

    majors = load_majors(majors_path)
    input_sha = sha256(jobs_path)
    majors_sha = sha256(majors_path)
    fingerprint_data = {
        "input_sha256": input_sha,
        "majors_sha256": majors_sha,
        "model": str(model_path),
        "top_k": TOP_K,
        "job_embedding_fields": JOB_EMBEDDING_FIELDS,
        "major_embedding_field": "embedding_text",
        "query_instruction": QUERY_INSTRUCTION,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    completed = 0
    if progress_path.exists() or part_path.exists():
        if not (progress_path.exists() and part_path.exists()):
            raise SystemExit("断点文件不完整，请使用--overwrite重新开始")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise SystemExit("断点口径与当前不一致，请使用--overwrite")
        completed = int(progress["completed_jobs"])
        with part_path.open("r", encoding="utf-8-sig", newline="") as handle:
            result_rows = max(sum(1 for _ in handle) - 1, 0)
        if result_rows != completed * TOP_K:
            raise SystemExit("断点CSV行数与进度不一致，请使用--overwrite")

    device = choose_device(args.device)
    print(f"[start] device={device} majors={len(majors)} completed_jobs={completed}")
    embedder = Embedder(model_path, device, args.batch_size)

    cache_valid = False
    if major_vectors_path.exists() and major_cache_path.exists():
        cache = json.loads(major_cache_path.read_text(encoding="utf-8"))
        cache_valid = cache.get("majors_sha256") == majors_sha and cache.get("model") == str(model_path)
    if cache_valid:
        major_vectors = np.load(major_vectors_path)
    else:
        print("[majors] 编码embedding_text")
        major_vectors = embedder.encode([row["embedding_text"] for row in majors], is_query=False)
        np.save(major_vectors_path, major_vectors)
        atomic_json(major_cache_path, {"majors_sha256": majors_sha, "model": str(model_path)})

    import faiss

    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(np.asarray(major_vectors, dtype="float32"))
    jobs = itertools.islice(eligible_jobs(jobs_path), completed, None)
    csv_mode = "a" if completed else "w"
    start_time = time.time()

    with part_path.open(csv_mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if not completed:
            writer.writeheader()
        while True:
            batch = list(itertools.islice(jobs, args.batch_size))
            if not batch:
                break
            vectors = embedder.encode([row["job_embedding_text"] for row in batch], is_query=True)
            scores, indices = index.search(vectors, TOP_K)
            for offset, job in enumerate(batch):
                for rank in range(TOP_K):
                    major = majors[int(indices[offset, rank])]
                    writer.writerow({
                        "job_id": job["job_id"],
                        "job_title_nlp": job["job_title_nlp"],
                        "major_requirement_raw": job["major_requirement_raw"],
                        "semantic_rank": rank + 1,
                        "major_code": major["major_code"],
                        "major_name": major["major_name"],
                        "embedding_score": round(float(scores[offset, rank]), 6),
                    })
            handle.flush()
            os.fsync(handle.fileno())
            completed += len(batch)
            atomic_json(progress_path, {"fingerprint": fingerprint, "completed_jobs": completed})
            elapsed = max(time.time() - start_time, 1e-6)
            print(f"[jobs] {completed:,}  batch_speed={len(batch) / elapsed:.1f}岗/秒")
            start_time = time.time()

    if completed == 0:
        raise RuntimeError("没有可用岗位，未生成Top10")
    os.replace(part_path, output_path)
    for path in (progress_path, major_vectors_path, major_cache_path):
        if path.exists():
            path.unlink()
    print(json.dumps({
        "output": str(output_path),
        "jobs": completed,
        "rows": completed * TOP_K,
        "top_k": TOP_K,
        "job_embedding_fields": JOB_EMBEDDING_FIELDS,
        "major_embedding_field": "embedding_text",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
