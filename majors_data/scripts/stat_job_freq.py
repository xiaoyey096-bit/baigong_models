#!/usr/bin/env python3
"""
统计每个专业的完整岗位名频次（清洗后），输出全量榜单供后续 LLM 归纳典型岗位名。

输入: majors_data/{月份范围}/majors_data_dwd_all/{code}_{name}.jsonl
输出: majors_data/岗位词频统计_{月份范围}.jsonl  （每专业一条，全量岗位名+频次，不截断）

⚠️ 后续 LLM 归纳典型岗位名时，请自行做抽样或截断：
   - 按"频次降序 + 累计覆盖 90%" 截取高频代表，或
   - 按 top-N (如 top100) 抽样，避免长尾噪声干扰命名
   - 不要把全量岗位名直接喂给 LLM（量大、含长尾噪声，命名易偏）

用法:
    python stat_job_freq.py --range 2026(01-06)                  # 全量统计（指定月份范围）
    python stat_job_freq.py --range 2026(01-06) --major 530701   # 单个专业
"""

import argparse, json, re
from collections import Counter
from pathlib import Path

BASE     = Path("/home/cl-715/workspace/models/majors_data")


def clean_name(name: str) -> str:
    """清洗岗位名: 去括号内容、薪资、促销词。"""
    name = re.sub(r"[（(【\[《<][^）)】\]》>]*[）)】\]》>]", "", name)  # 括号及内容
    name = re.sub(r"\d+[-~至]\d+[kKk]?\s*(元|月薪|薪|薪资|/月)?", "", name)  # 薪资
    name = re.sub(r"\d+[kKk]", "", name)
    for w in ["急聘","高薪","包吃住","包食宿","双休","五险一金","五险","六险一金","直招","内推","免费","当天入职","就近分配","提供住宿","无责底薪","五险+双休","双休+五险","正式工","长期工","短期工","小时工","日结","周结","月结","长白班","两班倒","坐岗","站岗","轻松","自由","弹性","欢迎应届生","接受小白","小白可做","无经验","经验不限","学历不限","不限经验","不限学历","岗前培训","免费培训","带薪培训","免费学","0基础","零基础","押运","派遣","外包","中介勿扰","非中介","直营","连锁","加盟","名额有限","先到先得","高提成","高回报","高收入","月入过万","月薪过万","上万"]:
        name = name.replace(w, "")
    name = re.sub(r"[-_/|·]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" -_/|·，,.")
    return name


def find_files(src_dir, filter_code: str = None):
    """扫描 dwd_all 目录，返回 [(code, name, filepath), ...]"""
    result = []
    for jp in sorted(src_dir.glob("*.jsonl")):
        stem = jp.stem
        # 文件名格式 {code}_{name}，按第一个 _ 切
        if "_" not in stem:
            continue
        code, name = stem.split("_", 1)
        if filter_code and code != filter_code:
            continue
        result.append((code, name, jp))
    return result


def stat_one(code, name, filepath) -> dict:
    """统计单个专业：清洗后岗位名频次（全量）。"""
    counter = Counter()
    total = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            t = (rec.get("原始岗位名称") or "").strip()
            if not t:
                continue
            c = clean_name(t)
            if c:
                counter[c] += 1

    # 全量按频次降序
    all_names = [{"name": n, "count": cnt} for n, cnt in counter.most_common()]
    return {
        "专业代码": code,
        "专业名称": name,
        "岗位总数": total,
        "去重岗位名数": len(counter),
        "岗位名频次": all_names,   # 全量，不截断
    }


def main():
    p = argparse.ArgumentParser(description="岗位名频次统计（按月份范围）")
    p.add_argument("--range", required=True,
                   help="月份范围标识，如 2026(01-06)；读 majors_data/{range}/majors_data_dwd_all/")
    p.add_argument("--major", help="只统计单个专业代码")
    args = p.parse_args()

    rng = args.range
    SRC_DIR = BASE / rng / "majors_data_dwd_all"
    OUT     = BASE / f"岗位词频统计_{rng}.jsonl"

    if not SRC_DIR.exists():
        print(f"[error] 输入目录不存在: {SRC_DIR}")
        print(f"        请先跑 Stage 3 (build_by_major.py --output-range {rng})")
        return

    print(f"[config] 月份范围: {rng}")
    print(f"[config] 输入: {SRC_DIR}")
    print(f"[config] 输出: {OUT}\n")

    files = find_files(SRC_DIR, args.major)
    print(f"[scan] 共 {len(files)} 个专业待统计")

    results = []
    for i, (code, name, jp) in enumerate(files, 1):
        r = stat_one(code, name, jp)
        results.append(r)
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {code} {name}: {r['岗位总数']}条, 去重{r['去重岗位名数']}个")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] {len(results)} 专业 → {OUT}")
    # 抽样
    if results:
        r = results[0]
        print(f"\n抽样 ({r['专业代码']} {r['专业名称']}, {r['岗位总数']}条岗位, {r['去重岗位名数']}个去重名):")
        for item in r["岗位名频次"][:10]:
            print(f"  {item['count']:>6}  {item['name']}")


if __name__ == "__main__":
    main()
