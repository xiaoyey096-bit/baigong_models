#!/usr/bin/env python3
"""共用配置：路径、模型参数、字段映射、工具函数。"""

import json
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 路径
# ══════════════════════════════════════════════════════════════
SCR_DIR    = Path(__file__).resolve().parent                     # scr/
BASE_DIR   = SCR_DIR.parent                                     # models_scripts/
ROOT_DIR   = BASE_DIR.parent.parent                             # workspace/
MODELS_DIR = BASE_DIR.parent                                    # models/

MAJORS_FILE      = BASE_DIR / "majors_usedtoembed" / "majors_optimized.jsonl"
JOBS_DIR         = ROOT_DIR / "data" / "new"

# 预筛单文件输入目录（--range 模式读这里）。两个位置都支持：
#   models/data_dwd/         —— 新批次专科岗位数据（当前主用，在 models/ 根下）
#   models_scripts/usedto_top3_data/ —— 旧批次（向后兼容，在 models_scripts/ 下）
DATA_DWD_DIR       = MODELS_DIR / "data_dwd"
USEDTO_TOP3_DIR    = BASE_DIR / "usedto_top3_data"

CACHE_MAJORS_DIR = BASE_DIR / "cache" / "majors"
CACHE_JOBS_DIR   = BASE_DIR / "cache" / "jobs"

OUTPUT_JSON_DIR  = BASE_DIR / "outputs_json"
OUTPUT_XLSX_DIR  = BASE_DIR / "outputs_xlsx"

PROGRESS_FILE    = BASE_DIR / "progress.json"

# ══════════════════════════════════════════════════════════════
# Embedding 模型（Phase 1 共用）
# ══════════════════════════════════════════════════════════════
EMBED_MODEL   = str(MODELS_DIR / "bge-large-zh-v1.5")
EMBED_DIM     = 1024
EMBED_BATCH   = 512
EMBED_MAX_LEN = 512

TOP_K_STAGE1  = 10                              # candidates.npy 存 Top10（Phase 1）
FLUSH_EVERY   = 500

# ══════════════════════════════════════════════════════════════
# TopN 方案（2026-08-18 起，Reranker 退役）
#   语义: BGE cosine top20 落盘，生产用 margin 截断（sim ≥ top1−margin）
#   规则: 精确命中 1.0 / 泛称命中 0.9（分数即来源标签）
# ══════════════════════════════════════════════════════════════
TOP_N_CAND    = 20                              # 语义候选落盘数（eval 候选池同源）
MARGIN        = 0.05                            # sim ≥ top1−margin 入选；eval 扫描后定稿
TOP_N_CAP     = 10                              # 最终每岗专业数上限
SCORE_RULE_EXACT  = 1.0                         # 规则: 专业标准名精确命中
SCORE_RULE_FUZZY  = 0.9                         # 规则: 泛称/别名命中（映射表）

# ══════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════
EMBED_QUERY_INST = "Given a job posting, retrieve the most relevant college major (higher vocational education specialty)"
RERANK_INST = "Given a job posting, determine whether the college major matches the job requirements"

# ══════════════════════════════════════════════════════════════
# 输出列：3 个原始字段 + 9 个专业匹配字段 = 12 列
# ══════════════════════════════════════════════════════════════
OUT_COLUMNS = [
    "岗位唯一标识", "原始岗位名称", "岗位全文",
    "专业1代码", "专业1名称", "专业1匹配分数",
    "专业2代码", "专业2名称", "专业2匹配分数",
    "专业3代码", "专业3名称", "专业3匹配分数",
]

# 模型输入字段
JOB_TEXT_FIELDS = ["原始岗位名称", "岗位全文"]

# 模型输入截断长度（仅喂模型时截断，输出文件保留原始完整值）
TITLE_MAX_LEN     = 20    # 原始岗位名称喂模型时取前20字
FULLTEXT_MAX_LEN  = 400   # 岗位全文喂模型时取前400字

# 原始 JSONL 字段名
SRC = {
    "job_id":   "岗位唯一标识",
    "title":    "原始岗位名称",
    "fulltext": "岗位全文",
}

# ══════════════════════════════════════════════════════════════
# extract_fields + build_job_text
# ══════════════════════════════════════════════════════════════
def extract_fields(rec: dict) -> dict:
    """原始岗位 JSON → 输出用字段（保留完整值，不截断）。

    ⚠️ 输出文件（merged.jsonl 等）用这里返回的值，必须保持原始完整，
       截断只在 build_job_text 喂模型时做。历史上这里曾直接截断，导致
       输出文件里「原始岗位名称」「岗位全文」被截断，已修正。
    """
    return {
        "岗位唯一标识": (rec.get(SRC["job_id"]) or "").strip(),
        "原始岗位名称": (rec.get(SRC["title"]) or "").strip(),
        "岗位全文":     (rec.get(SRC["fulltext"]) or "").strip(),
    }


def build_job_text(fields: dict) -> str:
    """拼接模型输入文本（喂模型时截断，不影响输出）。

    原始岗位名称取前 TITLE_MAX_LEN 字 | 岗位全文取前 FULLTEXT_MAX_LEN 字。
    """
    parts = [
        fields["原始岗位名称"][:TITLE_MAX_LEN],
        fields["岗位全文"][:FULLTEXT_MAX_LEN],
    ]
    return " | ".join(p for p in parts if p)


def build_row(fields: dict, top3: list) -> dict:
    """组装一行输出（12 列）。"""
    row = {k: fields.get(k, "") for k in OUT_COLUMNS[:3]}
    for i in range(1, 4):
        if i <= len(top3):
            code, name, score = top3[i - 1]
            row[f"专业{i}代码"] = code
            row[f"专业{i}名称"] = name
            row[f"专业{i}匹配分数"] = round(float(score), 4)
        else:
            row[f"专业{i}代码"] = ""
            row[f"专业{i}名称"] = ""
            row[f"专业{i}匹配分数"] = ""
    return row


def load_majors(path: Path) -> list:
    """读取 744 专业 jsonl → list[dict]。"""
    majors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            majors.append({
                "major_code": str(m.get("major_code", "")),
                "major_name": m.get("major_name", ""),
                "embedding_text": (m.get("embedding_text") or "").strip(),
            })
    return majors


# ══════════════════════════════════════════════════════════════
# 学历筛选
# ══════════════════════════════════════════════════════════════
def is_zhuanke(rec: dict) -> bool:
    """筛选专科相关岗位。

    判定口径（满足任一即算专科）：
      · 空值/缺失          → 视为不限，算专科
      · 大专/专科系列      → 含"大专"或"专科"（大专/大学专科/专科及以上/高职(专科)…）
      · 中职系列           → 含"中技"/"中专"（含"高中/中专/中技及以下"等复合值）
      · 不限系列           → v=="不限" 或 v.startswith("不限，")（不限/不限，工作/不限，月薪…）
                            或 含"学历不限"/"不限学历"/"学历无要求"/"无学历要求"

    ⚠️ 换数据后必查：招聘数据的"学历要求"标注会变，新变体参考
       models/工作流方案.md「专科筛选逻辑」章节的核查脚本，补进本函数。
    """
    v = (rec.get("学历要求") or "").strip()
    if not v:                                    # 空值 → 算专科
        return True
    if "大专" in v or "专科" in v:                # 大专/大学专科/专科及以上/高职(专科)
        return True
    if "中技" in v or "中专" in v:               # 中技/中专/高中/中专/中技及以下
        return True
    if v == "不限" or v.startswith("不限，"):     # 不限 / 不限，工作 / 不限，月薪
        return True
    if "学历不限" in v or "不限学历" in v:        # 学历不限 / 不限学历（语序颠倒）
        return True
    if "学历无要求" in v or "无学历要求" in v:    # 学历无要求 / 无学历要求
        return True
    return False


# ══════════════════════════════════════════════════════════════
# --range 单文件路径解析（run_rule_match / run_embed / run_rerank 共用）
# ══════════════════════════════════════════════════════════════
def resolve_range_file(range_id: Path, unmatched: bool = False):
    """根据 --range 标识解析输入文件路径。

    默认（unmatched=False）：优先在 data_dwd/ 找全量岗位文件，回退 usedto_top3_data/。
    unmatched=True：找 outputs_rule_{suffix}/{range}_unmatched.jsonl（规则未命中文件）。
      ⚠️ 新逻辑（2026-08-14）下语义对全量岗位跑，unmatched 参数已无人调用，
         保留仅为向后兼容（run_rule_match 仍产 _unmatched.jsonl）。

    Args:
        range_id: 范围标识，如 "2026(01-08)_专科"
        unmatched: 是否找规则未命中文件（默认 False，新流程不再用）

    Returns:
        (Path, exists_bool)
    """
    if unmatched:
        suffix = str(range_id).removesuffix("岗位")
        p = BASE_DIR / f"outputs_rule_{suffix}" / f"{range_id}_unmatched.jsonl"
        return p, p.exists()

    fname = f"{range_id}岗位.jsonl"
    for d in (DATA_DWD_DIR, USEDTO_TOP3_DIR):
        p = d / fname
        if p.exists():
            return p, True
    # 都不存在，返回 data_dwd 下的预期路径（用于报错信息）
    return DATA_DWD_DIR / fname, False
