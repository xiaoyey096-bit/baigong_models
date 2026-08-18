# models_scripts — 岗位→专业匹配推理脚本

> 本目录是**岗位数据平台**的模型推理层，将爬取的招聘岗位文本匹配到 744 个高职专科专业，输出 Top3 专业推荐 + 匹配分数。

---

## 目录结构

```
models_scripts/
├── scr/                         # 核心脚本
│   ├── config.py                # 共用配置（路径/参数/字段映射/学历筛选/工具函数）
│   ├── run_rule_match.py        # Stage 0: 规则匹配（专业标准名子串）→ 快路径 Top3
│   ├── run_embed.py             # Phase 1: BGE Embedding 编码 → FAISS 粗筛 Top10
│   ├── run_rerank.py            # Phase 2: Qwen3-Reranker-0.6B 精排 → Top3
│   ├── merge_results.py         # 合并规则 + 语义 Top3 结果
│   └── stats_major_jobs.py      # 统计每个专业匹配的岗位数（≥0.6）
├── majors_usedtoembed/          # 专业数据（用于 Embedding 编码 + 规则匹配）
│   ├── majors_optimized.jsonl   # 744 专业（major_code/name/embedding_text）
│   └── majors_optimized.xlsx
├── usedto_top3_data/            # 预筛单文件输入（专科岗位数据，--range 模式读这里）
├── cache/                       # 向量缓存
│   ├── majors/majors_744.index  # 744 专业 1024d FAISS 索引
│   └── jobs_{range}/single/     # --range 模式岗位向量缓存
│       ├── vectors.npy          # N × 1024 float32
│       └── candidates.npy       # N × 10 int64 (FAISS Top10 专业编号)
├── outputs_rule_{range}/        # 规则匹配产物（{range}_rule.jsonl + _unmatched.jsonl）
├── outputs_json_{range}/        # 语义匹配产物（{range}.jsonl 或 {range}_semantic.jsonl）
├── outputs_merged_{range}/      # 合并后最终结果（{range}_merged.jsonl，供 Stage3 读）
└── progress_{range}.json        # --range 模式断点续跑进度
```

---

## 做什么

### 双路径匹配管道（规则分流 + 语义）

岗位先走**规则匹配**快路径：用 744 专业标准名在「任职要求 ∪ 专业要求」做子串匹配，命中的岗位直接产 Top3（分=1.0），不再走模型；**未命中的**才走语义路径（Embed+Rerank）。两条路径结果最后合并。

> **为什么加规则路径**：实测约 11.9% 岗位在「任职要求」里明示了专业（如"市场营销相关专业优先"），这部分用规则精确命中即可，不必让 Reranker 舍近求远去猜，省约 12% 的精排版算力。
> **为什么只用这两个字段**：实测「岗位职责」里的专业词多为动作描述（皮肤护理/健康管理），误命中多，不用。「任职要求」命中密度 16.5% 且多为真命中；「专业要求」虽 99.9% 空，但非空即标准专业名原文，精度极高。

```
data/new/*.jsonl (18 数据源, 568万条岗位)
    │
    ▼  学历筛选（大专/空/不限/中技/中专/专科/无学历要求 → 专科岗位）
    │
    ▼  config.extract_fields: 原始岗位名称(≤20字) + 岗位全文(≤400字)
    │
    ├──────────────────┬─────────────────────────────┐
    │ Stage 0 规则匹配  │     未命中 → 语义路径         │
    │ run_rule_match   │                              │
    │ 任职要求∪专业要求 │ ┌──────────────────────────┐ │
    │ 子串匹配 744 名   │ │ Phase 1: BGE Embedding    │ │
    │ 命中→Top3(分1.0) │ │ + FAISS 粗筛 Top10        │ │
    │ 不走语义          │ └────────────┬─────────────┘ │
    │                  │              │ Top10 候选     │
    │                  │              ▼               │
    │                  │ ┌──────────────────────────┐ │
    │                  │ │ Phase 2: Qwen3-Reranker  │ │
    │                  │ │ CrossEncoder 精排 Top3   │ │
    │                  │ └────────────┬─────────────┘ │
    └──────────┬───────┴──────────────┘               │
               ▼                                       ▼
         merge_results.py（按 job_id 合并，规则优先）
               │
               ▼
        outputs_merged_{range}/{range}_merged.jsonl (12列)
```

### 输出格式（12 列）

```
岗位唯一标识 | 原始岗位名称 | 岗位全文 |
专业1代码 | 专业1名称 | 专业1匹配分数 |
专业2代码 | 专业2名称 | 专业2匹配分数 |
专业3代码 | 专业3名称 | 专业3匹配分数
```

- 匹配分数：sigmoid 0.0000~1.0000，**≥0.6 = 匹配**，≥0.9 = 强匹配
- **规则路径命中的专业分数固定为 1.0**（与语义分数同尺度，下游 Stage3 的 ≥0.6 阈值自然通过）

---

## 各脚本说明

### `config.py` — 共用配置

| 内容 | 说明 |
|------|------|
| 路径 | 模型/数据/缓存/输出目录定位 |
| `EMBED_MODEL` | `bge-large-zh-v1.5`（326M, 1024d, MIT） |
| `EMBED_BATCH=512` | 编码批大小 |
| `TOP_K_STAGE1=10` | FAISS 粗筛候选数 |
| `TOP_K_STAGE2=3` | 最终输出 TopK |
| `RERANK_BATCH=64` | Reranker 批大小 |
| `RERANK_MAX_LEN=384` | Reranker 单 pair 最大 token（提速 29%，质量≥95%） |
| `extract_fields()` | 原始 JSON → 岗位名称(≤20字) + 岗位全文(≤400字) |
| `is_zhuanke()` | 学历筛选：大专/空/不限/中技/中专/专科/无学历要求 → True |
| `load_majors()` | 读取 744 专业 jsonl |

### `run_rule_match.py` — Stage 0 规则匹配（快路径）

```bash
python run_rule_match.py --range "2026(01-06)_专科"                 # 标准用法
python run_rule_match.py --range "2026(01-06)_专科" --limit 10000   # 限量
python run_rule_match.py --range test --input ../outputs_json_old/  # 测试任意目录
```

- 用 `majors_optimized.jsonl` 的 744 专业标准名，在「任职要求 ∪ 专业要求」子串匹配
- 专业名按长度降序匹配（长名优先，避免"国际贸易"抢占"国际经济与贸易"）
- 命中的岗位 → 12 列输出（分=1.0）；未命中的 → 原样写入 `_unmatched.jsonl`
- 输出：`outputs_rule_{range}/{range}_rule.jsonl` + `_unmatched.jsonl`
- 预期命中率 ~11.9%（实测 30 万采样）

### `run_embed.py` — Phase 1 Embedding

```bash
python run_embed.py                     # 全量 18 源（多源模式）
python run_embed.py --source 51job      # 单源
python run_embed.py --list-sources      # 列数据源
python run_embed.py --range "2026(01-06)_专科"                    # 单文件预筛模式
python run_embed.py --range "2026(01-06)_专科" --rule-unmatched   # 只编码规则未命中岗位
```

- 构建/加载 744 专业 FAISS 索引
- `--range`：读 `usedto_top3_data/{range}岗位.jsonl` 单文件，缓存写 `cache/jobs_{range}/single/`
- `--rule-unmatched`：配合 `--range`，读 `outputs_rule_{range}/{range}_unmatched.jsonl`（只编码规则没覆盖的岗位，省算力）
- 断点续跑（按 vectors.npy 文件大小判断已编码条数）

### `run_rerank.py` — Phase 2 Reranker

```bash
python run_rerank.py                    # 全量（升序，跳过已完成）
python run_rerank.py --source 华为招聘   # 单源
python run_rerank.py --range "2026(01-06)_专科"                    # 单文件模式
python run_rerank.py --range "2026(01-06)_专科" --rule-unmatched   # 只重排规则未命中岗位
```

- 读缓存向量 + 岗位原文
- Qwen3-Reranker-0.6B CrossEncoder 打分（max_len=384）
- sigmoid → Top3；`--range` 输出 `outputs_json_{range}/{range}.jsonl`
- `--rule-unmatched`：读未命中缓存，输出 `{range}_semantic.jsonl`（仅未命中岗位的语义结果）
- 升序处理（小源先出结果），`progress.json` / `progress_{range}.json` 断点续跑

### `merge_results.py` — 合并规则 + 语义结果

```bash
python merge_results.py \
    --rule    ../outputs_rule_2026\(01-06\)_专科/ \
    --semantic ../outputs_json_2026\(01-06\)_专科/ \
    --range   "2026(01-06)_专科"
```

- 按「岗位唯一标识」合并：规则命中的用规则 Top3（分 1.0），未命中的用语义 Top3
- 同一 job_id 两边都有时**规则优先**（正常流程应无重叠：规则命中的不该再走语义）
- 输出：`outputs_merged_{range}/{range}_merged.jsonl`（全量 12 列，供 Stage3 build_by_major.py 读）

### `stats_major_jobs.py` — 统计

统计每个专业匹配的岗位数（分数 ≥0.6），输出 `models/统计_专业匹配岗位数.jsonl` + `.xlsx`。

---

## 关键参数

| 参数 | 值 | 说明 |
|------|:--|------|
| Embedding 模型 | bge-large-zh-v1.5 | 326M, 快，中文优化 |
| Reranker 模型 | Qwen3-Reranker-0.6B | CrossEncoder, C-MTEB 71.31 |
| FAISS 粗筛 | Top10 | IndexFlatIP 内积 |
| Reranker 精排 | Top3 | sigmoid ≥0.6 = 匹配 |
| 学历筛选 | 专科口径 | 大专/空/不限/中技/中专/专科/无学历要求 |

---

## 运行环境

```bash
source /home/cl-715/workspace/venv/bin/activate
# PyTorch 2.11 + CUDA 12.8
# sentence-transformers 5.6 / transformers 5.12 / faiss-cpu 1.14
```

详见 [dgx配置.md](../../dgx配置.md)
