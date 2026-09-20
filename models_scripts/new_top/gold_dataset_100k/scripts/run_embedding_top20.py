#!/usr/bin/env python3
"""为 10 万岗位生成纯 BGE Embedding Top20；不使用规则、Reranker 或 LLM。"""

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


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent
NEW_TOP_DIR = WORK_DIR.parent
MODELS_DIR = NEW_TOP_DIR.parents[1]

DEFAULT_JOBS = WORK_DIR / "data" / "10wdata.csv"
DEFAULT_MAJORS = MODELS_DIR / "models_scripts" / "majors_usedtoembed" / "new_majors.jsonl"
DEFAULT_MODEL = MODELS_DIR / "bge-large-zh-v1.5"
DEFAULT_OUTPUT = WORK_DIR / "output" / "embedding_top20.csv"

TOP_K = 20
EXPECTED_MAJORS = 744
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
    "embedding_rank",
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
    if text.startswith(("[", "{")):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [normalize(item) for item in parsed if normalize(item)]
            if isinstance(parsed, dict):
                return [
                    f"{normalize(key)}：{normalize(item)}"
                    for key, item in parsed.items()
                    if normalize(item)
                ]
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


def build_job_embedding_text(row: dict[str, str]) -> str:
    title = normalize(row.get("job_title_nlp") or row.get("job_title_std"))
    field_values = (
        ("专业要求", clean_major_requirement(row.get("major_requirement_raw"))),
        ("岗位名称", [title] if title else []),
        ("岗位职责", parse_listish(row.get("job_description_std"))),
        ("任职要求", parse_listish(row.get("job_requirement_std"))),
        ("专业能力", clean_ability_tags(row.get("ability_tags"))),
    )
    return "\n".join(
        f"{label}：{'；'.join(values)}" for label, values in field_values if values
    )


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_majors(path: Path) -> list[dict[str, str]]:
    required = {"major_code", "major_name", "embedding_text"}
    majors: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(f"专业目录第 {line_number} 行缺少字段：{sorted(missing)}")
            major = {field: normalize(row.get(field)) for field in required}
            if not all(major.values()):
                raise ValueError(f"专业目录第 {line_number} 行存在空字段")
            if major["major_code"] in seen_codes:
                raise ValueError(f"专业代码重复：{major['major_code']}")
            seen_codes.add(major["major_code"])
            majors.append(major)
    if len(majors) != EXPECTED_MAJORS:
        raise ValueError(f"专业目录应有 {EXPECTED_MAJORS} 个专业，实际为 {len(majors)}")
    return majors


def iter_jobs(path: Path) -> Iterator[dict[str, str]]:
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
            raise ValueError(f"岗位文件缺少字段：{sorted(missing)}")
        for row_number, row in enumerate(reader, 2):
            education = normalize(row.get("education_level_std"))
            if education not in ALLOWED_EDUCATION:
                raise ValueError(f"第 {row_number} 行学历口径无效：{education}")
            job_id = normalize(row.get("job_id"))
            if not job_id:
                raise ValueError(f"第 {row_number} 行 job_id 为空")
            if job_id in seen:
                raise ValueError(f"job_id 重复：{job_id}")
            seen.add(job_id)
            embedding_text = build_job_embedding_text(row)
            if not embedding_text:
                raise ValueError(f"岗位 {job_id} 无可用 Embedding 文本")
            yield {"job_id": job_id, "job_embedding_text": embedding_text}


def inspect_jobs(path: Path, expected_jobs: int) -> int:
    count = sum(1 for _ in iter_jobs(path))
    if expected_jobs and count != expected_jobs:
        raise ValueError(f"岗位数应为 {expected_jobs}，实际为 {count}")
    return count


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
        self.model = SentenceTransformer(
            str(model_path), device=device, model_kwargs=model_kwargs
        )
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
                texts, prompt_name="query" if is_query else "passage", **kwargs
            )
        except (TypeError, ValueError, KeyError):
            inputs = (
                [f"{QUERY_INSTRUCTION}: {item}" for item in texts] if is_query else texts
            )
            vectors = self.model.encode(inputs, **kwargs)
        result = np.asarray(vectors, dtype="float32")
        if result.ndim != 2 or result.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(f"向量维度异常：{result.shape}")
        return result


def rank_top20(scores: np.ndarray, majors: list[dict[str, str]]) -> list[int]:
    candidates = np.argpartition(-scores, TOP_K - 1)[:TOP_K]
    return sorted(
        candidates.tolist(),
        key=lambda index: (-float(scores[index]), majors[index]["major_code"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="10 万岗位纯 BGE Embedding Top20")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS))
    parser.add_argument("--majors", default=str(DEFAULT_MAJORS))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expected-jobs", type=int, default=100_000)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs_path = Path(args.jobs).resolve()
    majors_path = Path(args.majors).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    for path, label in ((jobs_path, "岗位文件"), (majors_path, "专业目录"), (model_path, "模型目录")):
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在：{path}")
    if args.batch_size <= 0 or (args.max_jobs is not None and args.max_jobs <= 0):
        raise ValueError("batch-size 和 max-jobs 必须为正数")

    majors = load_majors(majors_path)
    total_jobs = inspect_jobs(jobs_path, args.expected_jobs)
    target_jobs = min(total_jobs, args.max_jobs) if args.max_jobs else total_jobs
    validation = {
        "jobs": total_jobs,
        "target_jobs": target_jobs,
        "majors": len(majors),
        "top_k": TOP_K,
        "job_embedding_fields": JOB_EMBEDDING_FIELDS,
        "major_embedding_field": "embedding_text",
        "reranker": False,
        "rules": False,
    }
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress.json")
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    fingerprint_data = {
        **validation,
        "jobs_sha256": sha256(jobs_path),
        "majors_sha256": sha256(majors_path),
        "model": str(model_path),
        "query_instruction": QUERY_INSTRUCTION,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if args.overwrite:
        for path in (output_path, part_path, progress_path, metadata_path):
            if path.exists():
                path.unlink()
    elif output_path.exists():
        raise SystemExit(f"结果已存在：{output_path}；如需覆盖请添加 --overwrite")

    completed = 0
    if part_path.exists() or progress_path.exists():
        if not (part_path.exists() and progress_path.exists()):
            raise SystemExit("断点文件不完整，请使用 --overwrite 重新运行")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("fingerprint") != fingerprint:
            raise SystemExit("断点口径与当前任务不一致，请使用 --overwrite")
        completed = int(progress["completed_jobs"])

    device = choose_device(args.device)
    print(f"[start] device={device} jobs={target_jobs} majors={len(majors)} top_k={TOP_K}")
    embedder = Embedder(model_path, device, args.batch_size)
    major_vectors = embedder.encode(
        [major["embedding_text"] for major in majors], is_query=False
    )

    jobs = itertools.islice(iter_jobs(jobs_path), completed, target_jobs)
    mode = "a" if completed else "w"
    with part_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        if not completed:
            writer.writeheader()
        while completed < target_jobs:
            batch = list(itertools.islice(jobs, args.batch_size))
            if not batch:
                break
            started = time.time()
            job_vectors = embedder.encode(
                [job["job_embedding_text"] for job in batch], is_query=True
            )
            score_matrix = job_vectors @ major_vectors.T
            for row_index, job in enumerate(batch):
                ranked_indices = rank_top20(score_matrix[row_index], majors)
                for rank, major_index in enumerate(ranked_indices, 1):
                    major = majors[major_index]
                    writer.writerow(
                        {
                            "job_id": job["job_id"],
                            "embedding_rank": rank,
                            "major_code": major["major_code"],
                            "major_name": major["major_name"],
                            "embedding_score": f"{float(score_matrix[row_index, major_index]):.6f}",
                        }
                    )
            handle.flush()
            os.fsync(handle.fileno())
            completed += len(batch)
            atomic_json(
                progress_path,
                {"fingerprint": fingerprint, "completed_jobs": completed},
            )
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[progress] jobs={completed}/{target_jobs} "
                f"speed={len(batch) / elapsed:.2f} jobs/s",
                flush=True,
            )

    if completed != target_jobs:
        raise RuntimeError(f"实际完成岗位数 {completed}，目标岗位数 {target_jobs}")
    os.replace(part_path, output_path)
    if progress_path.exists():
        progress_path.unlink()
    atomic_json(
        metadata_path,
        {
            **fingerprint_data,
            "output": str(output_path),
            "output_rows": target_jobs * TOP_K,
            "device": device,
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "jobs": target_jobs,
                "rows": target_jobs * TOP_K,
                "top_k": TOP_K,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
