#!/usr/bin/env python3
"""
泛称映射表生成: 97 专业类骨架 + LLM 别名 → alias_table.json
======================================================================
- 骨架: code[:4] 分 97 个专业类，每类天然映射到类下全部专业（"XX类/XX相关"命中 → 类全集）
- LLM 增强: 逐类发类下专业名清单，让 GLM 产出该域的「泛称/俗称/上位词」
  （如 药学类 → 医药、药学相关、药剂；畜牧业类 → 兽医、养殖）
- 人工抽查后定稿。匹配时泛称命中 → 类下全部专业, score=0.9 (SCORE_RULE_FUZZY)

产出: models_scripts/majors_usedtoembed/alias_table.json
    [ {"alias": "兽医", "codes": ["410301","410302",...], "source": "llm"},
      {"alias": "药学类", "codes": [...], "source": "class"} ]

用法:
    export ZHIPUAI_API_KEY="..."
    python build_alias_table.py --major 4902          # 单类试跑
    python build_alias_table.py                       # 全量 97 类（断点续跑）
    python build_alias_table.py --review              # 抽查模式: 列出全部别名供人工过目
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import *

ALIAS_PATH = BASE_DIR / "majors_usedtoembed" / "alias_table.json"

DEFAULT_MODEL = "glm-5.2"
LLM_MAX_TOKENS = 16384
LLM_TIMEOUT = 360
MAX_RETRY = 3
WORKERS = 6

SYSTEM_PROMPT = """你是「职业教育术语专家」。你会收到一个专科【专业类】下属的专业名单。\
你的任务：列举招聘企业在任职要求/专业要求里常用的、指向这个专业类的【泛称/俗称/上位词】。

【什么算合格的泛称】
  · 企业真实会写的词: "药学相关专业"、"医药类背景"、"兽医专业"、"幼师专业"
  · 2~6 个字为宜，可以是: 域名(医药)、类名变体(药学类)、俗称(幼师→学前教育类)
  · 该词指向的岗位应适合类下【多数】专业毕业生，不是只对某一个专业

【不合格】
  · 与专业标准名完全相同的词（规则已有精确匹配，不用重复）
  · 过宽的词: "相关专业"、"专业不限"、"理工科"、"文科"（无法定位到类）
  · 与其他类混淆的词: "医生"同时涉及临床/中医/口腔多个类，不要给

只输出 JSON 数组（字符串列表），不要其他文字:
["医药", "药学相关", "药剂"]"""


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[error] 未安装 openai: pip install openai")
    base_url = os.environ.get("ZHIPUAI_BASE_URL") or "https://open.bigmodel.cn/api/coding/paas/v4"
    key = os.environ.get("ZHIPUAI_API_KEY")
    if not key:
        sys.exit("[error] 未设置 ZHIPUAI_API_KEY")
    return OpenAI(api_key=key, base_url=base_url)


def call_llm(client, user):
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                max_tokens=LLM_MAX_TOKENS, temperature=0.3, timeout=LLM_TIMEOUT,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\[.*\]", text, re.S)
            if not m:
                raise ValueError(text[:100])
            arr = json.loads(m.group(0))
            assert isinstance(arr, list) and all(isinstance(s, str) for s in arr)
            return [s.strip() for s in arr if 1 <= len(s.strip()) <= 10]
        except Exception as e:
            if attempt == MAX_RETRY - 1:
                print(f"  [fail] {e}")
                return None
            time.sleep(3 * (attempt + 1))


def load_classes():
    """744 专业 → {类code: [(code, name)...]}（按 code[:4]）。"""
    classes = defaultdict(list)
    for m in load_majors(MAJORS_FILE):
        classes[m["major_code"][:4]].append((m["major_code"], m["major_name"]))
    return dict(classes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--major", help="单类试跑（类 code 前 4 位，如 4902）")
    p.add_argument("--review", action="store_true", help="抽查模式: 打印现表")
    args = p.parse_args()

    classes = load_classes()
    print(f"[load] 744 专业 → {len(classes)} 类")

    # 类名骨架条目（source=class）: 类的代表词 = 公共前缀 + "类"
    skeleton = []
    existing = []
    if ALIAS_PATH.exists():
        existing = json.loads(ALIAS_PATH.read_text(encoding="utf-8"))
        done = {e["alias"] for e in existing}
        print(f"[续跑] 已有 {len(existing)} 条")
    else:
        done = set()

    if args.review:
        for e in sorted(existing, key=lambda x: x["alias"]):
            print(f"  [{e['source']:>5}] {e['alias']:　<6} → {len(e['codes'])} 专业")
        print(f"共 {len(existing)} 条; llm 来源 {sum(1 for e in existing if e['source']=='llm')}")
        return

    targets = {args.major: classes[args.major]} if args.major and args.major in classes else None
    todo = targets or classes

    client = make_client()
    lock = {"rows": list(existing)}
    from threading import Lock
    wlock = Lock()

    def gen_class(cls_code, members):
        names = "、".join(n for _, n in members)
        codes = [c for c, _ in members]
        user = f"专业类 {cls_code} 下属专业:\n{names}\n\n请给出该类的泛称/俗称/上位词。"
        aliases = call_llm(client, user)
        if aliases is None:
            return cls_code, 0
        rows = []
        for a in aliases:
            if a in done:
                continue
            rows.append({"alias": a, "codes": codes, "source": "llm"})
        with wlock:
            lock["rows"].extend(rows)
            ALIAS_PATH.write_text(json.dumps(lock["rows"], ensure_ascii=False, indent=1), encoding="utf-8")
        return cls_code, len(rows)

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(gen_class, c, m): c for c, m in todo.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            cls_code, n = fut.result()
            print(f"  [{i}/{len(todo)}] {cls_code} +{n} 别名  ({time.time()-t0:.0f}s)")

    final = lock["rows"]
    ALIAS_PATH.write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    n_llm = sum(1 for e in final if e["source"] == "llm")
    print(f"\n[完成] → {ALIAS_PATH}  (llm 别名 {n_llm} 条)")
    print("⚠️ 下一步: python build_alias_table.py --review 人工抽查，剔除跨类混淆词")


if __name__ == "__main__":
    main()
