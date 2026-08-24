#!/usr/bin/env python3
"""
LLM 裁判: top20 候选 → 金标（哪些专业真匹配，灵活 N，允许零匹配）
======================================================================
- 从 run_topn.py 的 {range}_top20.jsonl 抽样岗位，把 20 个候选专业
  （名称 + embedding_text 摘要）连同岗位原文发 GLM，让它选出真匹配的。
- 输出金标 eval/gold_{range}.jsonl: {job_id, gold: [codes...]}
- 岗位原文从 dwd 原文件按 job_id 回查（top20 文件只存了 id/title）。

裁判校准（前置必做）:
    python judge_eval.py --range "2026(01-06)_专科" --calibrate 50
    → eval/calibrate_{range}_50.jsonl 人工填"人工标注"列，
      与 LLM 判的结果对一致率 ≥85% 才可信。

用法:
    python judge_eval.py --range "2026(01-06)_专科" --sample 300        # 产金标
    python judge_eval.py --range "2026(01-06)_专科" --calibrate 50      # 人工校准样本
    python judge_eval.py --range "..." --sample 300 --judged-file ...    # 续跑

环境: export ZHIPUAI_API_KEY="..."（走智谱 Coding Plan 端点，openai SDK）
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import *

# ── LLM 后端（与 jobsname/scripts/name_clusters.py 同款接入）──────────
DEFAULT_MODEL = "glm-5.2"
LLM_MAX_TOKENS = 16384        # GLM 思考 ~8K 计入 max_tokens，必须给足
LLM_TIMEOUT = 360
MAX_RETRY = 3
WORKERS = 6

EVAL_DIR = BASE_DIR / "eval"

SYSTEM_PROMPT = """你是「职业教育岗位-专业匹配评审专家」。你会收到一个招聘岗位的原文，\
以及若干个专科专业候选（含专业简介）。你的任务：判断哪些专业与该岗位真实匹配。

【判断标准】
  · 该专业毕业生能胜任此岗位的核心职责，或岗位任职要求指向该专业的培养方向。
  · 只要"方向对得上"就算匹配，不要求字面出现专业名。
  · 宁缺毋滥：方向不符的坚决不选；确实一个都不符就返回空列表。

【边界示例】
  · 岗位是"药店营业员" → 药学、药品经营与管理 匹配；临床医学 不匹配。
  · 岗位是"药学相关专业优先" → 药学、中药学、药物制剂等药学类专业都算匹配。
  · 岗位是"行政文员"（无专业指向） → 若无任何候选方向对得上，返回空。

【输出格式】只输出一个 JSON 数组，元素为匹配的专业代码字符串，不要任何其他文字:
  ["520301", "590203"]
或空: []"""


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[error] 未安装 openai，请: pip install openai")
    base_url = os.environ.get("ZHIPUAI_BASE_URL") or "https://open.bigmodel.cn/api/coding/paas/v4"
    key = os.environ.get("ZHIPUAI_API_KEY")
    if not key:
        sys.exit("[error] 未设置 ZHIPUAI_API_KEY")
    return OpenAI(api_key=key, base_url=base_url)


def call_llm(client, model, user):
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                max_tokens=LLM_MAX_TOKENS, temperature=0.2, timeout=LLM_TIMEOUT,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                raise ValueError(f"无可解析 JSON: {text[:100]}")
            codes = json.loads(m.group(0))
            assert isinstance(codes, list) and all(isinstance(c, str) for c in codes)
            return codes
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                print(f"  [fail] {e}")
                return None
            time.sleep(3 * (attempt + 1))


# ── 岗位原文回查 ─────────────────────────────────────────────
def load_job_texts(rng, job_ids):
    """从 dwd 原文件按 job_id 取 (title, 岗位全文, 任职要求, 专业要求)。"""
    fp, exists = resolve_range_file(rng)
    if not exists:
        sys.exit(f"[error] 岗位原文件不存在: {fp}")
    want = set(job_ids)
    out = {}
    with open(fp, encoding="utf-8") as f:
        for line in f:
            if len(out) >= len(want):
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            jid = (rec.get(SRC["job_id"]) or "").strip()
            if jid in want:
                out[jid] = {
                    "title": (rec.get(SRC["title"]) or "").strip(),
                    "fulltext": (rec.get(SRC["fulltext"]) or "").strip()[:800],
                    "req": (rec.get("任职要求") or "").strip()[:300],
                }
    return out


def build_user_prompt(job, cands, majors_by_code):
    lines = [f"【岗位名称】{job['title']}",
             f"【岗位全文】{job['fulltext']}"]
    if job["req"]:
        lines.append(f"【任职要求】{job['req']}")
    lines.append("\n【候选专业】")
    for c in cands:
        m = majors_by_code[c["code"]]
        desc = (m.get("embedding_text") or "")[:200]
        lines.append(f"- {c['code']} {m['major_name']}: {desc}")
    lines.append("\n请判断哪些专业与该岗位真实匹配，只输出 JSON 代码数组。")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="LLM 裁判: top20 候选 → 金标")
    p.add_argument("--range", required=True)
    p.add_argument("--sample", type=int, default=300, help="抽样岗位数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--calibrate", type=int, help="只导出 N 条人工校准样本（不调 LLM）")
    p.add_argument("--gold-file", help="金标输出路径（默认 eval/gold_{range}.jsonl）")
    args = p.parse_args()

    rng = args.range
    suffix = rng.removesuffix("岗位")
    top20_path = BASE_DIR / f"outputs_topn_{suffix}" / f"{rng}_top20.jsonl"
    if not top20_path.exists():
        sys.exit(f"[error] top20 文件不存在: {top20_path}\n        先跑 run_topn.py --range {rng}")

    majors_by_code = {m["major_code"]: m for m in load_majors(MAJORS_FILE)}

    rows = []
    with open(top20_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"[load] top20 岗位 {len(rows):,}")

    random.seed(args.seed)
    sample = random.sample(rows, min(args.sample, len(rows)))
    job_texts = load_job_texts(rng, [r["job_id"] for r in sample])
    print(f"[join] 岗位原文回查到 {len(job_texts)}/{len(sample)}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 校准模式: 导出人工标注样本 ──
    if args.calibrate:
        out = EVAL_DIR / f"calibrate_{rng}_{args.calibrate}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in sample[:args.calibrate]:
                jt = job_texts.get(r["job_id"], {})
                f.write(json.dumps({
                    "job_id": r["job_id"],
                    "title": jt.get("title", r["title"]),
                    "fulltext": jt.get("fulltext", ""),
                    "candidates": [{"code": c["code"], "name": c["name"]} for c in r["top"]],
                    "人工标注": [],   # ← 填这里（专业代码数组），校准用
                }, ensure_ascii=False) + "\n")
        print(f"[校准样本] → {out}\n填完'人工标注'列后重跑 --sample 同 seed，对比 LLM 结果算一致率")
        return

    # ── 正常模式: LLM 判金标 ──
    gold_path = Path(args.gold_file) if args.gold_file else EVAL_DIR / f"gold_{rng}.jsonl"
    done = set()
    if gold_path.exists():
        with open(gold_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["job_id"])
        print(f"[续跑] 已有 {len(done)} 条，跳过")

    client = make_client()
    lock_write = open(gold_path, "a", encoding="utf-8")
    from threading import Lock
    wlock = Lock()

    def judge_one(r):
        jt = job_texts.get(r["job_id"])
        if not jt:
            return r["job_id"], None
        cands = [c for c in r["top"] if c["code"] in majors_by_code]
        codes = call_llm(client, args.model, build_user_prompt(jt, cands, majors_by_code))
        valid = set(c["code"] for c in cands)
        if codes is None:
            return r["job_id"], None
        return r["job_id"], [c for c in codes if c in valid]   # 只留候选内的

    todo = [r for r in sample if r["job_id"] not in done]
    print(f"[judge] {len(todo)} 岗位 × {args.model}，{WORKERS} 并发")
    t0, n_ok, n_empty = time.time(), 0, 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(judge_one, r): r["job_id"] for r in todo}
        for fut in as_completed(futs):
            jid, gold = fut.result()
            if gold is None:
                continue
            with wlock:
                lock_write.write(json.dumps({"job_id": jid, "gold": gold}, ensure_ascii=False) + "\n")
                lock_write.flush()
            n_ok += 1
            if not gold:
                n_empty += 1
            if n_ok % 20 == 0:
                print(f"  {n_ok}/{len(todo)}  空 {n_empty}  {time.time()-t0:.0f}s")
    lock_write.close()
    print(f"\n[完成] {n_ok}/{len(todo)} → {gold_path}  (零匹配岗位 {n_empty}, {n_empty/max(n_ok,1)*100:.0f}%)")


if __name__ == "__main__":
    main()
