# 10 万岗位模型标注数据集

本目录与 `new_top` 原有全量生产方案完全隔离。原方案仅用于字段口径和专业目录对齐；本目录用于构建新的 10 万岗位模型标注测试集。

## 目录结构

```text
gold_dataset_100k/
├── README.md
├── data/
│   └── 10wdata.csv
├── docs/
│   └── 10wdata_quality_report.md
├── output/
└── scripts/
    ├── select_quality_100k.py
    ├── analyze_10wdata.py
    └── run_embedding_top20.py
```

## 当前固定流程

```text
dwd.v_job_core
  → 五类核心字段筛选 100,000 个岗位
  → BGE 纯 Embedding 从 744 个专业中取 Top20
  → 大模型对 Top20 逐项判断并输出 Top10
```

五类岗位字段：

- `job_title_nlp`（为空时由 `job_title_std` 兜底）
- `job_description_std`
- `job_requirement_std`
- `ability_tags`
- `major_requirement_raw`

Embedding Top20 阶段不使用 Reranker、专业规则、模糊映射或 LLM。

## 纯 Embedding Top20

默认输入：`data/10wdata.csv`

默认专业目录：`../../majors_usedtoembed/new_majors.jsonl`

默认模型：`../../../bge-large-zh-v1.5`

默认输出：`output/embedding_top20.csv`

输出字段：

```text
job_id,embedding_rank,major_code,major_name,embedding_score
```

先做无 GPU 数据校验：

```bash
/home/cl-715/workspace/venv/bin/python scripts/run_embedding_top20.py --validate-only
```

得到确认后先运行 1,000 岗位样本：

```bash
/home/cl-715/workspace/venv/bin/python scripts/run_embedding_top20.py \
  --max-jobs 1000 \
  --output output/embedding_top20_sample_1000.csv \
  --overwrite
```

样本审核通过后再运行完整 10 万岗位。未经人工确认，不启动全量计算。
