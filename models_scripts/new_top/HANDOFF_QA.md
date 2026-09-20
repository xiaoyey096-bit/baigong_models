# 新 agent 交接问答：当前进度与测试集方案

本文件记录已经确认的结论。新 agent 先阅读本文件和同目录 `README.md`，再修改代码。

## 一、当前方案是什么？

问：最终采用什么 Top 策略？

答：每个岗位先用 BGE Embedding 从 744 个专业中取固定 Top10。然后只扫描 `major_requirement_raw`，对标准专业名称做精确匹配。若规则命中的专业已经在这 10 条中，就把它提前；候选集合不变，最终仍固定 10 条。

问：规则命中 Top10 外的专业怎么办？

答：保留在 `rule_matches.csv` 作为审计记录，但不插入最终 Top10。

问：一个岗位命中多个专业时只留一个吗？

答：不是。原始要求中命中多少个标准专业，就记录多少个专业；最终排序时，Top10 内的命中项全部提前。

问：是否扫描 `job_requirement_std` 做专业规则？

答：不扫描。`job_requirement_std` 只参与岗位 Embedding；精确规则唯一输入是 `major_requirement_raw`。

问：是否使用模糊映射、LLM 实时判断或 Reranker？

答：全部不使用。当前方案只有 BGE Embedding 和标准专业名称精确规则。

## 二、当前代码和数据到哪一步？

问：正式代码在哪里？

答：全部在 `script/`：

- `run_db_top10.py`：数据库直读入口；
- `run_embedding_top10.py`：Embedding Top10；
- `apply_major_rules.py`：精确规则；
- `merge_final_top10.py`：内部重排；
- `evaluate_top_schemes.py`：离线评测。

问：目前是否已经能从数据库读取？

答：可以。已验证从 `dwd.v_job_core` 读取、执行“大专/不限”筛选、分批生成 Embedding Top10。当前脚本只输出计算结果和抽样，不写生产表。

问：目前是否已经确定生产结果表？

答：还没有。老师旧版 TOPN 提供过一套 Run/Shard/Candidate 表参考，但其字段和当前算法不一致。需要拿到新的平台集成文档和目标结果表结构后，再实现写库和发布。

## 三、目前的评测是不是金牌数据集？

问：`evaluation/llm_blind_labels.csv` 是金标准吗？

答：不是。它有 50 个岗位、704 条岗位—专业候选记录，是 LLM 盲审形成的初步评测集，没有逐条人工确认，不能作为正式金牌数据集。

问：`rule_matches.csv` 是金标准吗？

答：也不是。它只能把岗位明确写出的标准专业作为规则基准，不能覆盖岗位职责中隐含但未写出的相关专业。

## 四、正式测试集如何构建？

问：测试集是否要人工制作一个有序 Top10？

答：不需要。金标准不预先规定唯一的 Top10 顺序，而是标注每个岗位有哪些相关专业及相关等级。系统运行后产生有序 Top10，再用标签评估召回和排序。

问：样本抽多少？

答：第一版抽 300 个岗位：

- 100 个 `explicit`：`major_requirement_raw` 明确写出专业；
- 100 个 `fuzzy`：计算机类、管理类、相关专业等模糊要求；
- 100 个 `unlimited`：专业不限或为空。

其中 200 个为 `validation`，用于检查和修订标注流程；100 个为 `test`，方案定稿后只用于最终报告。

问：测试集保存哪些文件？

答：两个输入文件加一个程序结果文件：

```text
eval_jobs.csv
 gold_major_labels.csv
 system_final_top10.csv
```

`eval_jobs.csv` 字段：

| 字段 | 含义 |
|---|---|
| `job_id` | 岗位唯一标识 |
| `job_title_nlp` | 岗位名称 |
| `job_description_std` | 岗位职责 |
| `job_requirement_std` | 任职要求 |
| `ability_tags` | 能力标签 |
| `major_requirement_raw` | 原始专业要求 |
| `sample_type` | `explicit` / `fuzzy` / `unlimited` |
| `dataset_split` | `validation` / `test` |

`gold_major_labels.csv` 字段：

| 字段 | 含义 |
|---|---|
| `job_id` | 岗位唯一标识 |
| `major_code` | 专业代码 |
| `major_name` | 专业名称 |
| `relevance_label` | `2`核心匹配；`1`合理相关；`0`不相关 |
| `label_evidence` | 岗位原文中的支持证据 |
| `label_reason` | 简短标注理由 |
| `review_status` | `llm_draft` 或 `human_confirmed` |

正式金标准只使用 `human_confirmed`。文件只需保存标签为 1 或 2 的专业；未列出的专业在评测时按 0 处理。不要把模型的 `semantic_rank` 或 `final_rank` 写成金标准标签。

问：标签需要给出人工 Top1、Top2 顺序吗？

答：不需要。标签 2 自动优先于标签 1；同一标签内部不强制规定顺序。这样才能评估多个合理专业同时存在的岗位。

问：LLM 在测试集构建中做什么？

答：LLM 只做初步候选推荐和理由生成；人工负责补充遗漏、删除错误、修改 1/2 等级并最终确认。LLM 初标不能直接当金标准。

## 五、测试指标和下一步

主要指标固定为：

- `Recall@10`：首要指标，Top10 找回多少相关专业；
- `Recall@5`：核心前五名召回；
- `Precision@10`：Top10 准确率；
- `NDCG@10`：标签 2 的专业是否排在前面；
- `Precision@5`、`NDCG@5`：核心层排序质量。

下一步顺序：

1. 从 `dwd.v_job_core` 抽取并分层确定 300 个评测岗位；
2. 根据 744 个专业目录生成 LLM 初标；
3. 人工确认 `gold_major_labels.csv`；
4. 使用 `run_db_top10.py` 或离线流程生成 `system_final_top10.csv`；
5. 编写针对金标准的评测脚本并生成最终报告；
6. 获取平台目标结果表和调度文档后，再实现批量写库、Run/Shard、重试和发布。
