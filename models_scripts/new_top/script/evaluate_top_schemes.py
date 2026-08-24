#!/usr/bin/env python3
"""第四步：复算纯Embedding、当前融合和推荐融合的评测指标。"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NEW_TOP_DIR = SCRIPT_DIR.parent
DEFAULT_EMBEDDING = NEW_TOP_DIR / "output" / "embedding_top10.csv"
DEFAULT_RULES = NEW_TOP_DIR / "output" / "rule_matches.csv"
DEFAULT_FINAL = NEW_TOP_DIR / "output" / "final_top10.csv"
DEFAULT_LABELS = NEW_TOP_DIR / "evaluation" / "llm_blind_labels.csv"

STRATA = (
    "A_no_injection",
    "B_low_injection_rank1_kept",
    "C_low_injection_rank1_dropped",
    "D_heavy_injection_rank1_dropped",
    "E_heavy_injection_rank1_kept",
)


def classify_final_group(rows: list[dict[str, str]]) -> str:
    injected = sum(row["fusion_source"] == "rule_only" for row in rows)
    rank1_dropped = not any(row["semantic_rank"] == "1" for row in rows)
    if injected == 0:
        return "A_no_injection"
    if injected <= 4 and not rank1_dropped:
        return "B_low_injection_rank1_kept"
    if injected <= 4 and rank1_dropped:
        return "C_low_injection_rank1_dropped"
    if rank1_dropped:
        return "D_heavy_injection_rank1_dropped"
    return "E_heavy_injection_rank1_kept"


def population_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for _, group in itertools.groupby(reader, key=lambda row: row["job_id"]):
            counts[classify_final_group(list(group))] += 1
    if set(counts) != set(STRATA):
        raise ValueError(f"融合分层不完整：{dict(counts)}")
    return counts


def load_labels(path: Path) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    required = {
        "sample_id", "stratum", "job_id", "major_code", "llm_relevance_label",
        "semantic_rank", "final_rank",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"盲审标签缺少字段：{sorted(missing)}")
        rows = list(reader)
    jobs: dict[str, list[dict[str, str]]] = defaultdict(list)
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        pair = (row["job_id"], row["major_code"])
        if pair in pairs:
            raise ValueError(f"盲审标签存在重复岗位—专业：{pair}")
        pairs.add(pair)
        if row["stratum"] not in STRATA:
            raise ValueError(f"未知盲审分层：{row['stratum']}")
        if row["llm_relevance_label"] not in {"0", "1", "2"}:
            raise ValueError(f"盲审标签只能为0、1、2：{pair}")
        jobs[row["job_id"]].append(row)
    if len(jobs) != 50 or len(rows) != 704:
        raise ValueError(f"盲审规模异常：jobs={len(jobs)} rows={len(rows)}")
    stratum_jobs = Counter(values[0]["stratum"] for values in jobs.values())
    if any(stratum_jobs[name] != 10 for name in STRATA):
        raise ValueError(f"盲审分层应各10个岗位：{dict(stratum_jobs)}")
    return rows, dict(jobs)


def dcg(labels: list[int]) -> float:
    return sum((2**label - 1) / math.log2(index + 2) for index, label in enumerate(labels))


def per_job_metric(
    rows: list[dict[str, str]], ranking: list[str], k: int, threshold: int
) -> dict[str, float]:
    labels = {row["major_code"]: int(row["llm_relevance_label"]) for row in rows}
    selected_labels = [labels.get(code, 0) for code in ranking[:k]]
    relevant = [label for label in labels.values() if label >= threshold]
    hits = sum(label >= threshold for label in selected_labels)
    ideal = sorted(labels.values(), reverse=True)[:k]
    return {
        "precision": hits / k,
        "recall": hits / len(relevant) if relevant else 1.0,
        "ndcg": dcg(selected_labels) / dcg(ideal) if dcg(ideal) else 1.0,
    }


def weighted_metrics(
    jobs: dict[str, list[dict[str, str]]],
    rankings: dict[str, list[str]],
    population: Counter[str],
    k: int,
    threshold: int,
) -> dict[str, float]:
    by_stratum: dict[str, list[dict[str, float]]] = defaultdict(list)
    for job_id, rows in jobs.items():
        by_stratum[rows[0]["stratum"]].append(
            per_job_metric(rows, rankings[job_id], k, threshold)
        )
    total = sum(population.values())
    result: dict[str, float] = {}
    for metric in ("precision", "recall", "ndcg"):
        result[metric] = sum(
            population[stratum] / total
            * sum(row[metric] for row in values) / len(values)
            for stratum, values in by_stratum.items()
        )
    return result


def label_rankings(
    jobs: dict[str, list[dict[str, str]]], rank_field: str
) -> dict[str, list[str]]:
    result = {}
    for job_id, rows in jobs.items():
        result[job_id] = [
            row["major_code"]
            for row in sorted(
                (row for row in rows if row[rank_field]),
                key=lambda row: int(row[rank_field]),
            )
        ]
    return result


def sample_rules(path: Path, job_ids: set[str]) -> dict[str, dict[str, dict[str, str]]]:
    rules: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["job_id"] in job_ids:
                rules[row["job_id"]][row["major_code"]] = row
    return dict(rules)


def recommended_rankings(
    jobs: dict[str, list[dict[str, str]]],
    rules: dict[str, dict[str, dict[str, str]]],
) -> dict[str, list[str]]:
    """精确规则可插入；模糊规则只提升原Embedding Top10内候选。"""
    result = {}
    for job_id, rows in jobs.items():
        embedding = sorted(
            (row for row in rows if row["semantic_rank"]),
            key=lambda row: int(row["semantic_rank"]),
        )
        embedding_codes = {row["major_code"] for row in embedding}
        job_rules = rules.get(job_id, {})
        both = [row["major_code"] for row in embedding if row["major_code"] in job_rules]
        exact_outside = sorted(
            code
            for code, rule in job_rules.items()
            if code not in embedding_codes and rule["rule_match_type"] == "1"
        )
        embedding_only = [
            row["major_code"] for row in embedding if row["major_code"] not in job_rules
        ]
        result[job_id] = (both + exact_outside + embedding_only)[:10]
    return result


def exact_targets(path: Path) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["rule_match_type"] == "1":
                targets[row["job_id"]].add(row["major_code"])
    return dict(targets)


def exact_recall(
    path: Path, targets: dict[str, set[str]], rank_field: str
) -> dict[int, dict[str, float]]:
    hits = Counter()
    full_jobs = Counter()
    total_targets = sum(map(len, targets.values()))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for job_id, group in itertools.groupby(reader, key=lambda row: row["job_id"]):
            if job_id not in targets:
                continue
            rows = list(group)
            for k in (5, 10):
                found = {
                    row["major_code"] for row in rows if int(row[rank_field]) <= k
                }
                count = len(found & targets[job_id])
                hits[k] += count
                full_jobs[k] += int(count == len(targets[job_id]))
    return {
        k: {
            "micro_recall": hits[k] / total_targets,
            "full_job_recall": full_jobs[k] / len(targets),
        }
        for k in (5, 10)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="复算Top5/Top10评测指标")
    parser.add_argument("--embedding", default=str(DEFAULT_EMBEDDING))
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--final", default=str(DEFAULT_FINAL))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    args = parser.parse_args()

    embedding_path = Path(args.embedding).resolve()
    rules_path = Path(args.rules).resolve()
    final_path = Path(args.final).resolve()
    label_path = Path(args.labels).resolve()

    _, jobs = load_labels(label_path)
    population = population_counts(final_path)
    rule_subset = sample_rules(rules_path, set(jobs))
    rankings = {
        "pure_embedding": label_rankings(jobs, "semantic_rank"),
        "current_fusion": label_rankings(jobs, "final_rank"),
        "recommended_exact_only_injection": recommended_rankings(jobs, rule_subset),
    }

    print(f"[population] {dict(population)}")
    print(f"[blind_eval] jobs={len(jobs)} candidate_pairs={sum(map(len, jobs.values()))}")
    for label_name, threshold in (("broad_relevance", 1), ("core_relevance", 2)):
        for scheme, scheme_rankings in rankings.items():
            for k in (5, 10):
                values = weighted_metrics(jobs, scheme_rankings, population, k, threshold)
                print(
                    f"[{label_name}] scheme={scheme} k={k} "
                    f"precision={values['precision']:.6f} "
                    f"recall={values['recall']:.6f} ndcg={values['ndcg']:.6f}"
                )

    targets = exact_targets(rules_path)
    print(
        f"[exact_benchmark] jobs={len(targets)} "
        f"target_pairs={sum(map(len, targets.values()))}"
    )
    for scheme, path, rank in (
        ("pure_embedding", embedding_path, "semantic_rank"),
        ("current_fusion", final_path, "final_rank"),
    ):
        values = exact_recall(path, targets, rank)
        for k in (5, 10):
            print(
                f"[exact_recall] scheme={scheme} k={k} "
                f"micro={values[k]['micro_recall']:.6f} "
                f"full_job={values[k]['full_job_recall']:.6f}"
            )


if __name__ == "__main__":
    main()
