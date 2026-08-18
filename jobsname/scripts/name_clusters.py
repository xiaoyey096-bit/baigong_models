#!/usr/bin/env python3
"""
种子构建：纯 LLM 看词频榜归纳（GLM / 兼容 OpenAI 格式的任意 LLM）
================================================================
读岗位词频统计（majors_data/岗位词频统计_{range}.jsonl），筛 ≥100 门槛专业，
逐专业把 top-N 词频榜发给 LLM 直接归纳「典型岗位名称」，输出种子文件
models/典型岗位名称_gt100_{range}.jsonl。

方法说明（2026-08-16 定稿）:
  早期纯 LLM 手工对话归纳 → 不可复现，弃用；
  中期 聚类(BGE+HDBSCAN)+LLM命名 → 2026-08-16 对照 eval 证明「巨簇吞方向」
    （同类岗位家族在 embedding 空间余弦>0.9，HDBSCAN 聚成一坨 → 只给 1 个种子，
    护理专业 3 种子覆盖率 84% vs 纯LLM 13 种子 96%），退役；
  现在 纯 LLM 看词频榜，但保留脚本化架构（固定 prompt + temperature 0.3 +
    逐专业 API 调用 + 断点续跑）—— 兼得 LLM 的辨向能力与可复现性。

输入:
  词频榜  majors_data/岗位词频统计_{range}.jsonl          （stat_job_freq.py 产出）
  门槛名  majors_data/专业匹配岗位统计_{range}.xlsx        （stat_by_major.py 产出，≥100 专业）

输出:
  models/典型岗位名称_gt100_{range}.jsonl                  （供 run_classify.py 用）

LLM 接入:
  · 默认 coding 后端（智谱 GLM Coding Plan 的 OpenAI 协议端点），model 默认 glm-5.2
  · API key 从环境变量 ZHIPUAI_API_KEY 读（或 --api-key 传）
  · 兼容 anthropic / zhipu / openai 后端（详见 --backend）

用法:
    source ~/workspace/venv/bin/activate
    export ZHIPUAI_API_KEY="你的key"
    cd models/jobsname/scripts

    python name_clusters.py --range 2026(01-08)                    # 全量（默认 6 路并发）
    python name_clusters.py --range 2026(01-08) --major 530701     # 单个专业（试跑）
    python name_clusters.py --range 2026(01-08) --dry-run          # 不调 LLM，只打印 prompt
    python name_clusters.py --range 2026(01-08) --model glm-4.6    # 指定模型
    python name_clusters.py --range 2026(01-08) --workers 8        # 调并发数

注：GLM-5.2/5.3 官方已不支持关闭思考（disabled/reasoning_effort=none 均被端点
忽略），思考 token 计入 max_tokens —— max_tokens 必须 ≥16384，否则思考挤空正文。

断点续跑:
  · 已输出的专业会跳过（除非 --overwrite）
  · 单个专业失败不会中断整体，失败专业记到 .failed 列表，最后汇总

═════════════════════════════════════════════════════════════
System Prompt 固化在下方 SYSTEM_PROMPT（与 种子构建说明.md 一致，勿随意改）。
详见 种子构建说明.md / README.md。
"""

import argparse, json, os, re, sys, time
from pathlib import Path

MODELS_DIR = Path("/home/cl-715/workspace/models")
SCRIPTS_DIR = Path(__file__).resolve().parent
MAJORS_DATA_DIR = MODELS_DIR / "majors_data"
SEEDS_ROOT = MODELS_DIR   # 输出 models/典型岗位名称_gt100_{range}.jsonl

DEFAULT_BACKEND = "coding"         # coding | anthropic | zhipu | openai
# 注：GLM-5.2/5.3 官方已不支持关闭思考（thinking.type=disabled 与 reasoning_effort=none
#     实测均被端点忽略，coding/anthropic 两端点都强制思考，单次 100~300s）。
#     应对：max_tokens 给足 16384（思考 ~8K 也计入，正文不会被挤空），速度靠并发补。
DEFAULT_MODEL   = "glm-5.2"        # 智谱默认；可改 glm-4.5 / glm-4.6 等
LLM_TEMPERATURE = 0.3              # 低温度保证可复现
LLM_MAX_TOKENS  = 16384            # 思考 ~8K token 计在 max_tokens 里，必须给足（8192 会挤空正文）
LLM_TIMEOUT     = 360              # 强制思考下单次 100~300s
MAX_RETRY       = 3                # 单专业解析失败重试次数
LLM_WORKERS     = 6                # 并发专业数（单次慢是思考固有耗时，靠并发压总时长）
TOP_FREQ_FOR_LLM = 150             # 每专业发给 LLM 的词频榜长度（eval 实测 150 足够辨向）
MIN_JOBS_PER_MAJOR = 100           # 门槛：与 run_classify.py 同口径（懒同步，见 load_inputs）

# Anthropic 协议端点（智谱官方，Claude Code 同款接入方式）
# SDK 会自动拼 /v1/messages；环境变量 ZHIPUAI_BASE_URL / ANTHROPIC_BASE_URL 可覆盖
def _pick_base_url(*candidates) -> str:
    """取第一个合法 http(s) 端点；防止误把 key 塞进 base_url 变量当域名连。"""
    for v in candidates:
        if v and str(v).startswith(("http://", "https://")):
            return str(v)
    return "https://open.bigmodel.cn/api/anthropic"

ANTHROPIC_BASE_URL = _pick_base_url(
    os.environ.get("ZHIPUAI_BASE_URL"), os.environ.get("ANTHROPIC_BASE_URL"))

# OpenAI 协议中转端点（backend=coding 用）；环境变量 ZHIPUAI_BASE_URL 可覆盖
CODING_BASE_URL = _pick_base_url(os.environ.get("ZHIPUAI_BASE_URL"),
                                 "https://open.bigmodel.cn/api/coding/paas/v4")


# ══════════════════════════════════════════════════════════════
def load_inputs(rng: str) -> tuple[set, dict, list]:
    """读门槛名单 + 词频榜 → (≥100 专业代码集, {代码:频次行}, 有序代码表)。"""
    global MIN_JOBS_PER_MAJOR
    # MIN_JOBS_PER_MAJOR 与 run_classify.py 保持单一来源
    try:
        sys.path.insert(0, str(SCRIPTS_DIR / "qwen"))
        from run_classify import MIN_JOBS_PER_MAJOR as _m
        MIN_JOBS_PER_MAJOR = _m
    except Exception:
        pass

    xlsx = MAJORS_DATA_DIR / f"专业匹配岗位统计_{rng}.xlsx"
    if not xlsx.exists():
        sys.exit(f"[error] 门槛名单不存在: {xlsx}\n"
                 f"        先跑 stat_by_major.py --range {rng}")

    from openpyxl import load_workbook
    wb = load_workbook(xlsx, read_only=True)
    ws = wb.active
    threshold = set()
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        code = str(row[0]).strip()
        cnt = row[2] or 0
        if code.isdigit() and len(code) == 6 and int(cnt) >= MIN_JOBS_PER_MAJOR:
            threshold.add(code)
    wb.close()

    freq_file = MAJORS_DATA_DIR / f"岗位词频统计_{rng}.jsonl"
    if not freq_file.exists():
        sys.exit(f"[error] 词频榜不存在: {freq_file}\n"
                 f"        先跑 stat_job_freq.py --range {rng}")
    freq_all, order = {}, []
    with open(freq_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            freq_all[r["专业代码"]] = r
            order.append(r["专业代码"])

    targets = sorted(c for c in threshold if c in freq_all)
    missing = threshold - set(freq_all.keys())
    if missing:
        print(f"[warn] {len(missing)} 个门槛专业不在词频榜里（跳过）: {sorted(missing)[:5]}...")
    return threshold, freq_all, targets


# ══════════════════════════════════════════════════════════════
# System Prompt —— 与 种子构建说明.md 完全一致，改这里须同步文档
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """你是「职业教育岗位分析专家」。你会收到一个专科专业的\
【岗位名词频榜】（岗位名×频次，降序，已截取高频段）。你的任务：直接从词频榜\
归纳出与该专业真实职业颗粒度数量一致的若干个「典型岗位名称」（数量不设上下限）。

【什么叫典型岗位名称】
该专业毕业生最可能从事的、具有代表性的【通用岗位称谓】。
  1. 是通用名，不是带公司/平台/级别的长尾变体：
     ✓ 电商运营        ✗ 天猫运营专员、亚马逊运营助理、电商运营经理
     ✓ 护士            ✗ ICU护士、手术室护士长、三甲医院护士
     ✓ 软件测试工程师  ✗ 自动化测试工程师(高级)、测试开发
  2. 若高频名本身就是通用名，直接用；
     若高频名都是带修饰的变体，归并/还原为上位通用名：
     电商运营专员 + 电商运营助理 + 电商运营岗 → 电商运营。
  3. 一个典型名要能覆盖一批原始长尾变体，起到归类锚点作用。

【怎么读词频榜】
  · 频次是代表性信号：高频名是该专业的主干职业方向；低频名里可能藏着
    真实但小众的方向，值得辨认，不为频次牺牲真实方向。
  · 同一岗位的多种叫法（护士/口腔护士长/临床护士混排）只算一个方向；
    但【职责确有不同的家族成员要分开】：护士 ≠ 护工（持证上岗 vs 生活照护）、
    护士 ≠ 护士长（临床执行 vs 病区管理）—— 这类颗粒度差异正是要辨的。
  · 明显脏标题不要据其立种子：含公司名、薪资诱饵（"300一天包吃住"、
    "保底8000"）、招聘前缀（"(急招)""【直聘】"）的条目，剥掉杂质后若指向
    已有方向就归入，指向不明就忽略。
  · 宽专业自然多方向、窄专业自然少，不人为配额。

【归纳规则】
  · 数量（不设上下限，唯一标准是合理性）：该专业有几个【边界清晰】的
    职业颗粒度就给几个种子。实质相似的必须合并、真实不同的必须保留：
    既不为凑数增补，也不为精简而牺牲真实方向。
  · 覆盖度：不能全是同一职能，要兼顾该专业的主要职业方向
    （电商 = 运营 + 客服 + 设计/美工 + 销售 + 直播/新媒体）。
  · 频次优先但不唯频次：高频名为主干，补入代表重要方向的低频名。
  · 相近专业趋同（重要）：含义相近的专业，典型岗位名应大致一致；
    但若两者确有可解释的职责侧重差异，保留差异，不为整齐而牺牲准确。

【合并去重规则（重要 —— 每个专业输出前必做）】

种子列表里若两个名存在「包含关系」（一个是另一个的字面子串，或仅差
职业等级/资质/复数后缀），必须逐对判断该合并还是该保留：

  ▶ 必须合并 —— 二者指同一个岗位，只是叫法/修饰不同：
    判定信号（满足其一即合并）：
      · 仅差"员/人员/工/师"等职业通名后缀：保安员=保安人员=安保员 → 保安
      · 仅差"注册/高级/中级/助理/专员"等资质级别前缀，且在本专业语境下
        不构成独立岗位：安全工程师 = 注册安全工程师（注册只是证书）→ 安全工程师
      · 加的行业/领域定语就是本专业默认语境，去掉不丢信息：
        在【化工装备技术】里：设备工程师 = 化工设备工程师 → 设备工程师
        在【建筑工程技术】里：施工员 = 土建施工员 → 施工员
    合并时取【字数最少、且语义完整】的那个（通常是去掉定语的上位通用名），
    频次按两者相加计。

  ▶ 必须保留 —— 字面虽包含，但有独立的职责/技术领域/工种身份：
    判定信号（满足其一即保留）：
      · 是独立持证工种，非"通用名+修饰"：维修电工 ≠ 电工（维修电工专注维修、
        单独持证）；仪表工 ≠ 仪表工程师（操作工 vs 工程师，级别职责不同）
      · 限定了不同的处理对象/技术分支：污水处理工程师 vs 水处理工程师
        （污水特指，水处理是含净水/纯水的上位）—— 在水处理/环境专业里两者并存合理
      · 限定了不同的检验领域：微生物检验员 ≠ 检验员（需无菌/微生物专业技术）
    这种情况两个都保留，它们是不同的归类锚点。

  一句话原则：字面包含只触发审查，不自动合并；
              合并的依据是"同一岗位的不同叫法"，保留的依据是"不同岗位"。

【边界自检（重要 —— 每个专业输出前必做）】
对最终种子列表两两名逐对检查一次边界：
  · 两个名的【主要职责差异】要能用一句话说清（服务对象/核心技能/工作
    产出不同）。说不清差异、或实质只是同一职责的两种侧重 → 合并。
  · 两个名的职责差异一句话能说清、招聘市场上也确实是两类岗 → 保留。
最终效果：种子之间互不重叠、边界清晰；每个原始岗位名归入哪个种子
都应有唯一答案，不存在"两个都像"的模糊地带。

【输出格式（严格）】
逐专业输出，每个专业一行 JSON，顺序与输入一致。不要 markdown、不要解释、\
不要代码块包裹，整体即一个合法的 jsonl 文件：
{"专业代码":"530701","专业名称":"电子商务","典型岗位名称":["电商运营","跨境电商运营","电商客服","电商美工","网络销售","直播运营"]}
{"专业代码":"410102","专业名称":"作物生产与经营管理","典型岗位名称":["农业技术员","农艺师","种植技术员","植保技术员","农资销售员","农业技术推广员"]}
"""


# ══════════════════════════════════════════════════════════════
def build_user_prompt(freq_row: dict, top_n: int = TOP_FREQ_FOR_LLM) -> str:
    """把单个专业的词频榜 → User Prompt 文本（取 top_n 高频名，带频次）。"""
    code = freq_row["专业代码"]
    name = freq_row["专业名称"]
    pairs = [(it["name"], it.get("count", 0)) for it in freq_row["岗位名频次"]
             if it.get("name") and it.get("count", 0) > 0][:top_n]

    lines = [f"专业代码：{code}", f"专业名称：{name}",
             f"词频榜（前 {len(pairs)} 名，按频次降序）："]
    for nm, c in pairs:
        lines.append(f"  {nm} ×{c}")
    lines.append("")
    lines.append("请按系统提示词规则，为本专业输出【一行】 JSON。")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
def make_client(backend: str, api_key: str):
    """按 backend 创建 LLM client。"""
    if backend == "anthropic":
        try:
            import anthropic
        except ImportError:
            sys.exit("[error] 未安装 anthropic，请: pip install anthropic")
        return anthropic.Anthropic(api_key=api_key, base_url=ANTHROPIC_BASE_URL)
    elif backend == "zhipu":
        try:
            from zhipuai import ZhipuAI
        except ImportError:
            sys.exit("[error] 未安装 zhipuai，请: pip install zhipuai")
        return ZhipuAI(api_key=api_key)
    elif backend in ("openai", "coding"):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("[error] 未安装 openai，请: pip install openai")
        # coding: 走 Coding Plan 专属端点；openai: 读环境变量 base_url
        base_url = CODING_BASE_URL if backend == "coding" else os.environ.get("OPENAI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        sys.exit(f"[error] 未知 backend: {backend}（支持 anthropic / coding / zhipu / openai）")


def call_llm(client, backend: str, model: str, system: str, user: str) -> str:
    """调一次 LLM，返回纯文本响应。统一两种 backend 的调用差异。

    GLM-5.2 官方已不支持关闭思考（disabled / reasoning_effort=none 均被忽略），
    思考 token 计入 max_tokens —— 因此 max_tokens 必须 ≥16384，否则思考 ~8K
    挤空正文（返回为空的主因）。下方仍传 thinking=disabled：智谱端点不报错，
    若换到支持关闭的模型/端点即可直接生效提速。
    """
    if backend == "anthropic":
        # Anthropic 协议：system 独立参数。智谱的 GLM-5.2 在此端点默认开思考，
        # 思考会吃掉 max_tokens 导致正文为空 → 显式传 thinking=disabled（实测端点接受）
        resp = client.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT,
            thinking={"type": "disabled"},
        )
        # content 是 block 列表（ThinkingBlock/TextBlock），优先取正文
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if type(b).__name__ == "TextBlock").strip()
        if not text:
            # 兜底：思考未被端点生效时，从 ThinkingBlock 尾部找（JSON 常在思考末尾）
            thinking = "".join(getattr(b, "thinking", "") for b in resp.content
                               if type(b).__name__ == "ThinkingBlock").strip()
            return thinking
        return text

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(model=model, messages=msgs,
                  temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS)
    if backend == "zhipu":
        resp = client.chat.completions.create(**kwargs, thinking={"type": "disabled"})
    else:  # openai 兼容（含 coding）
        resp = client.chat.completions.create(
            **kwargs, timeout=LLM_TIMEOUT,
            extra_body={"thinking": {"type": "disabled"}},
        )
    # 部分中转网关返回非标准结构（SDK 直接解析成了 str），做防御
    if isinstance(resp, str):
        return resp.strip()
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        # 思考未被端点支持时兜底：从 reasoning_content 里找 JSON
        rc = getattr(msg, "reasoning_content", None) or ""
        return rc.strip()
    return content


# ══════════════════════════════════════════════════════════════
JSON_OBJ_RE = re.compile(r'\{[^{}]*"典型岗位名称"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)


def parse_one_major(raw_text: str, expected_code: str) -> dict | None:
    """从 LLM 响应里解析出一行专业 JSON。容错：去 markdown 包裹、正则提取。"""
    text = raw_text.strip()
    # 去 ```json ... ``` 包裹
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 尝试整段解析（理想情况：LLM 只输出一行 JSON）
    try:
        obj = json.loads(text)
        if _valid_seed(obj, expected_code):
            return obj
    except json.JSONDecodeError:
        pass

    # 容错：从文本里正则提取含"典型岗位名称"的 JSON 对象
    for m in JSON_OBJ_RE.finditer(text):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if _valid_seed(obj, expected_code):
                return obj
        except json.JSONDecodeError:
            continue

    # 最后兜底：逐行尝试
    for line in text.splitlines():
        line = line.strip().strip("`,")
        if '"典型岗位名称"' not in line:
            continue
        try:
            obj = json.loads(line)
            if _valid_seed(obj, expected_code):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _valid_seed(obj: dict, expected_code: str) -> bool:
    """校验解析出的对象格式。"""
    if not isinstance(obj, dict):
        return False
    if str(obj.get("专业代码", "")).strip() != str(expected_code).strip():
        return False
    seeds = obj.get("典型岗位名称")
    if not isinstance(seeds, list) or not seeds:
        return False
    if not all(isinstance(s, str) and s.strip() for s in seeds):
        return False
    return True


# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="种子构建 Stage B：LLM 给簇草稿起名")
    ap.add_argument("--range", required=True, help="月份范围，如 2026(01-08)")
    ap.add_argument("--major", help="只跑单个专业代码（试跑用）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM 模型名（默认 {DEFAULT_MODEL}）")
    ap.add_argument("--backend", default=DEFAULT_BACKEND,
                    choices=["anthropic", "coding", "zhipu", "openai"],
                    help=f"LLM 后端（默认 {DEFAULT_BACKEND}；anthropic=智谱Anthropic协议端点/Coding Plan；coding=Coding Plan的OpenAI协议端点）")
    ap.add_argument("--api-key", help="API key；默认读 ZHIPUAI_API_KEY / OPENAI_API_KEY 环境变量")
    ap.add_argument("--workers", type=int, default=LLM_WORKERS,
                    help=f"并发专业数（默认 {LLM_WORKERS}；GLM-5.2 强制思考单次 100~300s，靠并发压总时长）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不调 LLM，只打印第一个专业的 prompt（调试用）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已输出的专业（默认跳过已完成的）")
    args = ap.parse_args()

    rng = args.range
    out_file = SEEDS_ROOT / f"典型岗位名称_gt100_{rng}.jsonl"

    # 读门槛名单 + 词频榜
    threshold, freq_all, targets = load_inputs(rng)

    if args.major:
        if args.major not in freq_all:
            sys.exit(f"[error] 专业 {args.major} 不在词频榜里")
        if args.major not in threshold:
            print(f"[warn] 专业 {args.major} 岗位数 < {MIN_JOBS_PER_MAJOR} 门槛，仍按指定试跑")
        targets = [args.major]

    print(f"[config] 月份范围: {rng}")
    print(f"[config] 输入: 岗位词频统计_{rng}.jsonl  top-{TOP_FREQ_FOR_LLM}  (≥{MIN_JOBS_PER_MAJOR} 门槛共 {len(targets)} 专业)")
    print(f"[config] 输出: {out_file}")
    print(f"[config] LLM: backend={args.backend}  model={args.model}\n")

    # ── dry-run：只打印 prompt ──
    if args.dry_run:
        code = targets[0]
        row = freq_all[code]
        print(f"===== DRY-RUN: {code} {row['专业名称']} =====\n")
        print("--- SYSTEM PROMPT ---")
        print(SYSTEM_PROMPT)
        print("\n--- USER PROMPT ---")
        print(build_user_prompt(row))
        return

    # ── 读已有输出（断点续跑）──
    done = {}
    if out_file.exists() and not args.overwrite:
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done[r["专业代码"]] = r
                except json.JSONDecodeError:
                    continue
        print(f"[resume] 输出文件已有 {len(done)} 个专业完成（这些将跳过，其余照常跑）\n")

    # ── API key ──
    # 智谱系后端（anthropic/coding/zhipu）读 ZHIPUAI_API_KEY；anthropic 也认 ANTHROPIC_API_KEY；
    # openai 后端额外读 OPENAI_API_KEY
    if args.backend == "zhipu":
        api_key = args.api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not api_key:
            sys.exit("[error] 缺 API key。设环境变量 ZHIPUAI_API_KEY 或用 --api-key 传入")
    else:
        api_key = (args.api_key or os.environ.get("ZHIPUAI_API_KEY")
                   or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            sys.exit("[error] 缺 API key。设环境变量 ZHIPUAI_API_KEY（或 ANTHROPIC_API_KEY / OPENAI_API_KEY），或用 --api-key 传入")

    client = make_client(args.backend, api_key)

    # ── 逐专业起名（并发；GLM-5.2 强制思考单次 100~300s，靠并发压总时长）──
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # 用 append 模式续写（done 的已在文件里，跳过）
    mode = "a" if (out_file.exists() and not args.overwrite) else "w"
    todo = [c for c in targets if args.overwrite or c not in done]
    failed = []
    t_all = time.time()
    processed = 0
    print(f"[run] 待跑 {len(todo)} 专业（跳过已完成 {len(done)}），并发 {args.workers}\n")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def run_one(code: str):
        """跑一个专业（含重试），返回 (code, 种子obj|None, 最后错误)。"""
        row = freq_all[code]
        user_prompt = build_user_prompt(row)
        last_err = ""
        for _attempt in range(MAX_RETRY):
            try:
                raw = call_llm(client, args.backend, args.model, SYSTEM_PROMPT, user_prompt)
                obj = parse_one_major(raw, code)
                if obj:
                    return code, obj, ""
                last_err = ("返回为空(思考token挤空正文/截断)" if not raw
                            else f"解析失败, 返回前200字: {raw[:200]!r}")
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(2)
        return code, None, last_err

    with open(out_file, mode, encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, c): c for c in todo}
        # 写文件只发生在主线程（as_completed 循环），无交错风险
        for i, fut in enumerate(as_completed(futs), 1):
            code, obj, err = fut.result()
            row = freq_all[code]
            if obj:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fout.flush()
                processed += 1
                seeds_preview = "、".join(obj["典型岗位名称"][:4])
                print(f"  [{i}/{len(todo)}] ✅ {code} {row['专业名称']} → "
                      f"{len(obj['典型岗位名称'])}种子: {seeds_preview}")
            else:
                failed.append((code, row["专业名称"], err))
                print(f"  [{i}/{len(todo)}] ❌ {code} {row['专业名称']}: {err}")

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"完成 {processed} 专业  失败 {len(failed)}  耗时 {time.time()-t_all:.0f}s")
    if failed:
        print(f"失败列表（可重跑，断点续跑会跳过已成功的）:")
        for code, name, err in failed[:10]:
            print(f"  {code} {name}: {err[:80]}")
        if len(failed) > 10:
            print(f"  ...还有 {len(failed)-10} 个")
    print(f"{'='*60}")
    print(f"\n输出: {out_file}")
    if not failed:
        print("下一步: python run_classify.py --range " + rng + "  （Stage 4 归类）")


if __name__ == "__main__":
    main()
