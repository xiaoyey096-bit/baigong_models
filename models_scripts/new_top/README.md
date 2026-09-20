# 岗位—专业固定 Top10 匹配方案

本目录是当前唯一有效的岗位—专业匹配方案及其运行产物。新 agent 接手时，以本 README 和 `HANDOFF_QA.md` 为准，不需要参考其他旧方案。

## 1. 已定稿算法

```text
PostgreSQL dwd.v_job_core
  → SQL 筛选 education_level_std IN ('大专','不限')
  → Python 再次校验学历
  → 构建岗位 Embedding 文本
  → BGE 生成岗位向量
  → 与 744 个专业 embedding_text 计算相似度
  → 每岗取固定 Embedding Top10
  → 只扫描 major_requirement_raw 做标准专业名称精确匹配
  → 仅在 Top10 内把规则命中的专业提前
  → 输出每岗固定 10 条最终结果
```

确定的业务规则：

- Embedding 决定 Top10 候选集合。
- 每个岗位最终固定 10 条，不允许规则命中 Top10 外专业插入。
- 规则只扫描 `major_requirement_raw`（原始专业要求）。
- 一个岗位命中多个专业时，全部记录。
- Top10 内同时被 Embedding 和精确规则命中的专业优先；同一层按 `embedding_score` 降序。
- 不使用 Reranker（重排序模型）、模糊映射、别名映射、小类展开或实时 LLM 判断。

## 2. 输入字段

### 岗位字段

| 字段                      | 中文含义             | 用途                     |
| ------------------------- | -------------------- | ------------------------ |
| `job_id`                | 岗位唯一标识         | 主键                     |
| `job_title_nlp`         | NLP 处理后的岗位名称 | Embedding                |
| `job_description_std`   | 标准化岗位职责       | Embedding                |
| `job_requirement_std`   | 标准化任职要求       | Embedding                |
| `ability_tags`          | 技能/能力标签        | Embedding                |
| `major_requirement_raw` | 原始专业要求         | Embedding + 唯一规则输入 |
| `education_level_std`   | 标准化学历           | 数据筛选                 |
| `crawl_date`            | 抓取日期             | 可选时间筛选             |

生产数据从 PostgreSQL View `dwd.v_job_core` 读取，使用服务端游标分批处理，不导出岗位源数据作为中间 JSON/CSV。

### 专业字段

`new_majors.jsonl` 包含 744 个专业：

| 字段               | 中文含义                                    |
| ------------------ | ------------------------------------------- |
| `major_code`     | 专业代码                                    |
| `major_name`     | 专业名称                                    |
| `embedding_text` | 专业向量文本，专业 Embedding 唯一使用此字段 |

## 3. 输出字段

`output/final_top10.csv` 是当前离线最终结果格式，每个岗位严格 10 行：

| 字段                            | 中文含义                                                |
| ------------------------------- | ------------------------------------------------------- |
| `job_id`                      | 岗位唯一标识                                            |
| `major_code` / `major_name` | 专业代码 / 专业名称                                     |
| `embedding_score`             | 岗位与专业的向量相似度，不是概率                        |
| `semantic_rank`               | 纯 Embedding 排名                                       |
| `final_rank`                  | 精确规则内部重排后的最终排名，1—10                     |
| `rule_match_type`             | `0` 未命中；`1` 命中标准专业名称                    |
| `rule_evidence`               | 原文中实际命中的专业词                                  |
| `fusion_priority`             | `1` Embedding 与规则同时命中；`2` 仅 Embedding 命中 |
| `fusion_source`               | `embedding_and_rule` 或 `embedding_only`            |
| `embedding_text`              | 对应专业实际使用的向量文本                              |

## 4. 正式脚本

```text
script/run_db_top10.py          # 生产入口：数据库直读、Embedding、规则、Top10
script/run_embedding_top10.py   # CSV 离线 Embedding Top10
script/apply_major_rules.py     # major_requirement_raw 精确规则
script/merge_final_top10.py     # Top10 内部重排
script/evaluate_top_schemes.py  # 离线评测脚本，不参与生产计算
```

离线文件流程：

```bash
cd /home/cl-715/workspace/models/models_scripts/new_top
/home/cl-715/workspace/venv/bin/python script/run_embedding_top10.py --device auto --overwrite
/home/cl-715/workspace/venv/bin/python script/apply_major_rules.py --overwrite
/home/cl-715/workspace/venv/bin/python script/merge_final_top10.py --overwrite
```

数据库小样本流程：

```bash
export TOPN_DB_HOST=192.168.111.40
export TOPN_DB_PORT=5432
export TOPN_DB_NAME=baigongdb
export TOPN_DB_USER=baigong
export TOPN_DB_PASSWORD='由Secret注入，不写入文件'

/home/cl-715/workspace/venv/bin/python script/run_db_top10.py \
  --max-jobs 5 --batch-size 5 --print-jobs 1 \
  --model /home/cl-715/workspace/models/bge-large-zh-v1.5 --device auto
```

正式全量运行必须显式添加 `--confirm-full-run`。当前 `run_db_top10.py` 只计算并打印抽样结果，尚未写入生产结果表；目标表结构和平台集成合同确认后再增加写库 sink。

## 5. 当前离线产物

| 路径                                | 内容                                                 |
| ----------------------------------- | ---------------------------------------------------- |
| `output/embedding_top10.csv`      | 纯 Embedding Top10                                   |
| `output/rule_matches.csv`         | 所有精确规则命中记录，包括 Top10 外命中              |
| `output/final_top10.csv`          | 规则只在 Top10 内重排后的最终结果                    |
| `evaluation/llm_blind_labels.csv` | 50 个岗位、704 条候选的 LLM 初步盲审标签，不是金标准 |
| `evaluation/evaluation_report.md` | 当前离线质量检查报告                                 |
| `HANDOFF_QA.md`                   | 面向新 agent 的完整问答交接记录                      |

已完成的一次全量离线检查：90,807 个岗位、908,070 条最终结果；每岗 10 条；候选集合与纯 Embedding Top10 一致；9,220 个岗位发生了内部顺序调整。以上是当前运行基线，不是人工金标准结论。

## 6. 数据库与平台边界

- 源 View：`dwd.v_job_core`。
- 当前测试连接：PostgreSQL `192.168.111.40:5432`，数据库 `baigongdb`，用户 `baigong`。
- 密码只能由环境变量或平台 Secret 注入。
- 平台集成需要另行确定 Run、Shard、租约、心跳、重试、幂等、结果表和发布指针；算法层不得改变固定 Top10 和精确规则口径。
