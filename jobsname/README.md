# jobsname — 典型岗位归类

> 接收 `majors_data/{月份范围}/majors_data_dwd_all/` 的按专业分组的岗位数据，将每个专业下的杂乱岗位名归纳到**种子典型岗位**集合 + 「其他」。
>
> **归类模型**：Qwen3-Embedding-4B（MRL 缩维 1024d），KNN 种子锚点匹配。

---

## 两件事，别混淆

本目录做两件不同的事，分两个阶段：

| 阶段 | 做什么 | 用什么 | 产物 | 谁消费 |
|------|--------|--------|------|--------|
| **① 种子构建** | 把杂乱岗位名归纳成若干个典型岗位种子（数量随职业颗粒度，不设上限） | **纯 LLM 看词频榜**（`name_clusters.py`，固定 prompt + temperature 0.3 + 逐专业 API，可复现） | `models/典型岗位名称_gt100_{范围}.jsonl` | Stage 4 归类 |
| **② 岗位归类（Stage 4）** | 把每条岗位归到某个种子或「其他」 | Qwen3-Embedding-4B KNN 锚点匹配 | `output/datarow_gt100/{范围}/` | 下游交付 |

> 📌 **种子构建**是归类的前置依赖。没有种子，Stage 4 无锚点可匹配。
> 种子构建方法见下方「种子构建方法」章节；归类逻辑见「数据流」「脚本说明」章节。

---

## 种子构建方法（重点 —— 2026-08-16 定稿：纯 LLM 看词频榜）

### 方法演进（为什么现在是纯 LLM）

| 版本 | 方法 | 结局 |
|------|------|------|
| v1 | 纯 LLM **手工对话**归纳（人工贴频次榜给 Claude 聊） | 不可复现、质量看运气，弃用 |
| v2 | **聚类（BGE+HDBSCAN）定结构 + LLM 定名字** | 2026-08-16 对照 eval 证明失效，退役；脚本已删（2026-08-17 清理） |
| **v3（现行）** | **纯 LLM 看词频榜，脚本化**：固定 prompt + temperature 0.3 + 逐专业 API 调用 + 断点续跑 | ✅ 兼得 LLM 辨向能力与可复现性 |

**v2 退役依据**（3 专业 × 同一 Qwen KNN 管线、同 θ=0.65 对照）：同类岗位家族（护士/口腔护士/护士长/护工）在 embedding 空间余弦 >0.9，HDBSCAN 聚成巨簇 →「一簇一种子」→ 护理只给 3 个种子、覆盖率 84%；纯 LLM 看词频榜给 13 个种子、覆盖率 **96%**。聚类"定结构"在最需要细分的同质专业上恰好失效——辨方向本来就是 LLM 的强项，词频榜 + 合并/边界规则足够它稳定发挥。

### 架构

```
每个专业（≥100 个匹配岗位的专业）:
  岗位词频统计_{range}.jsonl 里该专业的 top-150 高频岗位名（岗位名×频次，降序）
    │
    ▼  name_clusters.py（脚本，逐专业调 LLM）
    │  · User Prompt = 专业代码 + 专业名称 + 词频榜 top-150
    │  · System Prompt 固化在脚本里（典型名定义 / 怎么读词频榜 /
    │    归纳规则 / 合并去重规则 / 边界自检，详见 种子构建说明.md）
    │  · GLM-5.2（Coding Plan）· temperature 0.3 · 6 路并发 · 断点续跑
    │
    │  种子数量不设上下限 —— 有几个边界清晰的职业颗粒度就给几个
    │
    ▼
  models/典型岗位名称_gt100_{范围}.jsonl
    {"专业代码":"...","专业名称":"...","典型岗位名称":[...]}
```

### 具体做法

**脚本**：`scripts/name_clusters.py`（GLM / 兼容 OpenAI 格式的任意 LLM）

- 输入两份（均为上游现成产物）：门槛名单 `专业匹配岗位统计_{range}.xlsx`（筛 ≥100 专业）+ 词频榜 `岗位词频统计_{range}.jsonl`
- **逐专业**调 LLM：User Prompt = 该专业词频榜 top-150（`TOP_FREQ_FOR_LLM`，eval 实测足够辨向）；脏标题原样进榜，系统提示词已教 LLM 剥杂质归方向、不据其立种子
- 套用「合并去重规则」（见 [种子构建说明.md](种子构建说明.md)）：字面包含只触发审查、合并依据是"同一岗位不同叫法"、保留依据是"不同岗位"
- **种子数量不设上下限，唯一标准是合理性**：实质相似的合并、真实不同的保留，输出前两两名过一遍边界自检
- 输出：`models/典型岗位名称_gt100_{范围}.jsonl`（格式不变，下游零改）
- **断点续跑**：已成功的专业跳过；单专业失败不中断，记失败列表可重跑

> **LLM 接入**：默认 `coding` 后端，走智谱 **GLM Coding Plan 套餐**的 OpenAI 协议端点，model 默认 `glm-5.2`。key 从环境变量 `ZHIPUAI_API_KEY` 读。也可：`--backend anthropic`（套餐的 Anthropic 协议端点，Claude Code 同款）；`--backend zhipu`（按量计费，`zhipuai` SDK，需余额）；`--backend openai`（OpenAI 兼容，接本地/其他云端）。**GLM-5.2 官方已不支持关闭思考**（disabled/reasoning_effort=none 均被端点忽略），应对 = max_tokens 16384 + 6 路并发（`--workers`）。详见 [种子构建说明.md](种子构建说明.md)。

### 种子构建命令

```bash
source ~/workspace/venv/bin/activate
cd models/jobsname/scripts

# ── 前置：生成两份输入（门槛名单 + 词频榜），见 种子构建说明.md ──
# python ../../majors_data/scripts/stat_by_major.py --range 2026(01-06)
# python ../../majors_data/scripts/stat_job_freq.py  --range 2026(01-06)

# 纯 LLM 看词频榜（调 GLM Coding Plan，逐专业归纳）
export ZHIPUAI_API_KEY="你的key"
python name_clusters.py --range 2026(01-06) --dry-run        # 先看 prompt 全文
python name_clusters.py --range 2026(01-06) --major 530701   # 单专业试跑看效果
python name_clusters.py --range 2026(01-06)                  # 全量（断点续跑，默认 6 路并发）
# → models/典型岗位名称_gt100_2026(01-06).jsonl

# 验收（推荐）：合并质量检查，见 种子构建说明.md 末尾
```

---

## 命名规范（重要）

产物按**月份范围**分目录，与上游 majors_data 的月份范围一一对应：

```
jobsname/
├── scripts/
│   ├── qwen/                        # Stage 4 归类（Qwen3-Embedding-4B 版，唯一主力）
│   │   ├── run_classify.py          #   全量归类（按 --range 指定月份范围）
│   │   └── calibrate_theta.py       #   θ 校准工具（一次性，换模型时重跑）
│   ├── name_clusters.py             # 种子构建：纯 LLM 看词频榜归纳（GLM，逐专业 API）
│   └── output/                      #   历史草稿产物目录（聚类版草稿已清理）
├── output/
│   └── datarow_gt100/               #   Stage 4 归类结果根目录（gt100 = 只分析岗位数≥100的专业）
│       ├── {月份范围}/              #     如 2026(01-06)，与 majors_data 对应
│       │   └── {专业代码}_{专业名称}.jsonl
│       └── old/                     #   旧版产物（归档保留）
├── 种子构建说明.md                  #   种子构建 prompt + 流程（纯 LLM 词频榜）
└── README.md
```

> **目录名含义**：`datarow_gt100` 的 `gt100` = 只对**有效匹配岗位数 ≥ 100**的专业做归类。理由：<100 的专业样本太少，归纳出的种子不可靠；种子文件 `典型岗位名称_gt100_{范围}.jsonl` 覆盖所有 ≥100 的专业。
>
> ⚠️ **100 这个阈值可能随甲方需求变化**，改动需同时：① 改 `run_classify.py` 顶部 `MIN_JOBS_PER_MAJOR`；② 同步种子文件覆盖范围。

---

## 数据流（Stage 4 岗位归类）

```
majors_data/{月份范围}/majors_data_dwd_all/{专业代码}_{专业名称}.jsonl
    │
    │  每行: {岗位唯一标识, 原始岗位名称, 岗位全文, 匹配专业代码, ...}
    │
    ▼  种子文件: models/典型岗位名称_gt100_{范围}.jsonl (≥100专业)
    ▼  run_classify.py --range {月份范围}
    │
    │  对于每个专业:
    │    ① 读 dwd JSONL → 岗位唯一标识 + 原始岗位名称
    │    ② 岗位数 < 100 → 跳过（MIN_JOBS_PER_MAJOR）
    │    ③ 去重 + 频次统计（按清洗后名去重，同名只编码一次；输出仍写原始名）
    │    ④ Qwen3-Embedding-4B 编码种子名 + 岗位名（MRL 1024d, normalize=True；清洗后名）
    │    ⑤ 余弦匹配:
    │         max_sim ≥ 0.65 → 归入对应种子
    │         max_sim < 0.65 → "其他"
    │    ⑥ 输出: {岗位唯一标识, 岗位名称, 典型岗位名称}
    │
    └─→ output/datarow_gt100/{月份范围}/{专业代码}_{专业名称}.jsonl
```

---

## 种子文件

| 文件 | 专业数 | 来源 |
|------|:--|------|
| `models/典型岗位名称_gt100_{范围}.jsonl` | **≥100 的专业**（种子数随颗粒度，不设上限） | 纯 LLM 看词频榜归纳（`name_clusters.py`）；旧版（手工对话 / 聚类+LLM）已归档 |

每行格式：
```json
{"专业代码": "530701", "专业名称": "电子商务", "典型岗位名称": ["电商运营", "跨境电商运营", "电商客服", "电商美工/视觉设计", "网络销售", "直播/新媒体运营"]}
```

> 📌 **新范围种子如何构建**：见 [种子构建说明.md](种子构建说明.md) —— 纯 LLM 看词频榜归纳，门槛为 xlsx 中岗位数 ≥ 100 的专业，输出 `典型岗位名称_gt100_{范围}.jsonl`。

---

## 输出格式

每行 3 列：
```json
{"岗位唯一标识": "JOBL20002000202606291", "岗位名称": "电商客服", "典型岗位名称": "电商客服"}
```

归不上的 → `"典型岗位名称": "其他"`。

---

## 脚本说明

### `scripts/name_clusters.py` — 种子构建：纯 LLM 看词频榜归纳

```bash
source ~/workspace/venv/bin/activate
cd models/jobsname/scripts
export ZHIPUAI_API_KEY="你的key"

python name_clusters.py --range 2026(01-06) --dry-run        # 不调 LLM，看 prompt
python name_clusters.py --range 2026(01-06) --major 530701   # 单专业试跑
python name_clusters.py --range 2026(01-06)                  # 全量（断点续跑，默认 6 路并发）
python name_clusters.py --range 2026(01-06) --model glm-4.6  # 换模型
python name_clusters.py --range 2026(01-06) --workers 8      # 调并发数
```

- `--range`：月份范围（必填），读 `majors_data/岗位词频统计_{range}.jsonl`（+ 门槛名单 xlsx 筛 ≥100 专业），写 `models/典型岗位名称_gt100_{range}.jsonl`
- 每专业发词频榜 top-150 给 LLM（`TOP_FREQ_FOR_LLM`），System Prompt 固化在脚本里
- `--backend`：`coding`（默认，智谱 Coding Plan 套餐端点）/ `anthropic`（套餐 Anthropic 协议端点）/ `zhipu`（按量计费，`zhipuai` SDK）/ `openai`（兼容格式，接本地/其他云端）
- `--model`：模型名，默认 `glm-5.2`（Coding Plan 支持 glm-5.2/5.1 等）
- API key：`--api-key` 或环境变量 `ZHIPUAI_API_KEY`（coding/zhipu/anthropic）
- **断点续跑**：已输出的专业跳过；`--overwrite` 强制重跑
- JSON 解析容错：自动剥离 markdown 包裹/解释文字，校验专业代码 + 种子非空

### `scripts/qwen/run_classify.py` — Stage 4 全量归类（按月份范围）

```bash
source ~/workspace/venv/bin/activate
cd models/jobsname/scripts/qwen

# 指定处理的月份范围（必填）
python run_classify.py --range 2026(01-06)            # 该范围全量大专业
python run_classify.py --range 2026(01-06) --major 530701   # 单个专业
```

- `--range`：月份范围标识（必填），读 `majors_data/{range}/majors_data_dwd_all/`，写 `output/datarow_gt100/{range}/`
- 岗位数 < `MIN_JOBS_PER_MAJOR(100)` 的专业自动跳过
- Qwen3-Embedding-4B 编码（MRL 缩维 1024d）
- θ1=0.65，KNN 种子锚点匹配

### `scripts/qwen/calibrate_theta.py` — θ 校准（一次性）

```bash
python calibrate_theta.py
```

- 拿电子商务 530701 算 job→seed 余弦分布
- 定 θ1（Qwen 短文本余弦中位 0.728，取 0.65 覆盖 ~74% 去重名）
- 换模型后重跑

---

## 关键参数

| 参数 | 值 | 阶段 | 说明 |
|------|:--|:--|------|
| **种子构建** | | | |
| 方法 | 纯 LLM 看词频榜 | 种子 | `name_clusters.py`；聚类版已退役并删除（2026-08-17） |
| 词频榜 top-N | **150** | 种子 | 每专业发给 LLM 的高频名数（eval 实测足够辨向，护理 13 方向 / 新能源汽车 18 方向全覆盖） |
| LLM | GLM-5.2（Coding Plan）| 种子 | temperature 0.3；默认 coding 端点，兼容 anthropic/zhipu/openai；思考关不掉 → max_tokens 16384 + 并发 6 |
| **岗位归类** | | | |
| 归类模型 | Qwen3-Embedding-4B | Stage4 | MRL 缩维 1024d，短文本区分度优于 BGE |
| **MIN_JOBS_PER_MAJOR** | **100** | Stage4 | **只归类岗位数≥100的专业；可随甲方需求调整** |
| θ1 | 0.65 | Stage4 | 余弦匹配阈值（校准得出） |
| 标题清洗 | clean_name() | Stage4 | embedding/匹配前剥"(急招)"【直聘】前缀、公司名段；输出保留原始名（eval 实测护理 +26pt） |
| EMBED_BATCH | 128 | Stage4 | 编码批大小 |
| 算法 | KNN 种子锚点匹配 | Stage4 | 每个岗位归入余弦最近且 ≥θ 的种子 |

---

## 环境

```bash
source /home/cl-715/workspace/venv/bin/activate
# sentence-transformers, transformers, torch, numpy
# 种子构建: openai (默认 coding 后端)，可选 anthropic / zhipuai (按量计费用)
```

上游：[majors_data/README.md](../../majors_data/README.md)
推理层：[models_scripts/README.md](../models_scripts/README.md)
全链路：[工作流方案.md](../工作流方案.md)
