# TOPN 平台对接要点（任务书消化版）

> 来源：《招聘数据TOPN算法外部实现与师融百工集成研发任务书_20260818.md》+ TOPN/ 参考实现。
> 用途：算法做完后按此对接进平台，免重读原文。只保留和算法侧相关的内容，平台侧职责略。

---

## 1. 一句话理解

算法跑在外部（咱们 40/145/172 的 GPU），但**运行控制、结果落库、质量门禁、发布**全部走平台：PostgreSQL 是唯一权威，Ray 按 96 分片租约式执行，页面只看平台 API。咱们交付的是"算法端口"——合同和测试留参考实现的，替换召回/排序/推理实现。

## 2. 红线（对接时逐条自查）

1. **绝对不写** `dwd.dwd_job_detail_t` 的历史 Top3 九字段（match_prof1~3 的 id/name/weight）——只读
2. 不改生产发布指针（发布只能走平台 DG API 的原子切换）
3. 不在 Git/镜像/日志/文档存真实密码、Token、DSN
4. 不把专业数量写死在代码（专业集合来自版本化目录快照 `meta.dim_major_t`，运行中 SHA256 不得变）
5. 不物理删除已发布 Run

## 3. 数据口径

- **源**：`dwd.dwd_job_detail_t`，只读，筛选 `education_level_std IN ('大专','不限') AND crawl_date >= DATE '2026-07-27'`
- **分片**：`hash(inference_key) % shard_count`（默认 96），分片 SQL 用 `FOR UPDATE SKIP LOCKED` 租约认领，90s 心跳
- **去重单元**：`duplicate_group_id` 有效 → `DG:{id}` 一个单元只推理一次；否则 `JOB:{job_id}`；代表岗按质量分确定性选取
- **缓存**：规范语义 `semantic_sha256` + 模型/特征/目录版本组成 `inference_key`，跨单元复用

## 4. 必须产出（数据库合同）

| 表 | 写什么 |
|---|---|
| `dg.job_major_topn_run_t` | Run 控制（幂等键、top_n 1~20、recall_top_k、算法/模型版本、制品摘要） |
| `dg.job_major_topn_shard_t` | 分片状态/心跳/检查点 |
| `dwd.job_major_topn_unit_t` + `..._unit_member_t` | 匹配单元及成员（demand_weight 合计 1） |
| `dwd.job_major_topn_candidate_t` | **候选主表**：rank 1..top_n、major_code/name、raw_scores(jsonb)、relevance_score、match_probability(可空)、attribution_weight(单元内合计1)、confidence_level(H/M/L)、reason_codes、evidence、各版本号、status(CANDIDATE/REPORTABLE/REVIEW_REQUIRED) |
| `dg.job_major_topn_pointer_t` | 发布指针（只有平台能写） |

字段分离铁律：`raw_scores` 只审计；`relevance_score` 是排序刻度**不是概率**；概率需独立校准+ECE/Brier；`attribution_weight` 是守恒归因。

## 5. 内部 API（外部引擎提供，供平台 DG/Compute 调）

`/internal/v1/job-major-topn/*`：health、capabilities、estimate、runs(创建)、shards/claim、heartbeat、results:batch(幂等)、complete/fail/retry、validate、request-publish。错误合同带 code/retryable/trace_id。参考实现的 `api/openapi.yaml` 可直接用。

## 6. 状态机

```
DRAFT→QUEUED→RUNNING→VALIDATING→GATE_PASSED→PUBLISHED
                 |            └→ FAILED（重试仅重跑失败分片，同 run_id）
                 └→ FAILED
```

## 7. 质量门禁（平台决策、咱们算指标）

- 完整率、rank 连续、单元内专业不重、attribution_weight 合计=1（误差 <1e-6）
- **Gold 指标**：Top1 P / Hit@3 / Recall@5 / MRR / NDCG@5、覆盖率、拒识率、集中度
- 换代门槛：Top1 95% 置信区间下界不低于 Champion（历史 Top3），重点专业不显著回退
- Gold Dataset：全专业覆盖、正例+近邻难例+负例、按重复组切分防泄漏、双人标注+仲裁、允许 ABSTAIN

## 8. 咱们算法 → 合同的映射表（对接时照此改）

| 咱们现有 | 合同对应 | 改造 |
|---|---|---|
| BGE 编码（run_embed） | 步骤4 Small Dense 全量编码 | I/O 改 SQL 读源表；产出写 semantic_sha256 缓存 |
| top20 落盘（run_topn） | 步骤5 全专业候选召回 recall_top_k=20 | 候选写 `job_major_topn_candidate_t`，cosine → raw_scores.small + relevance_score（跨路径校准） |
| 规则 AC 自动机（run_rule_match_v2） | 步骤5 显式规则路径 + 证据 | 命中写 raw_scores.rule + evidence(source_field, signal)；1.0/0.9 分进校准层 |
| merge 并集 | 步骤8 排序校准融合 | 语义+规则分路径归一后融合；补 attribution_weight（softmax 或按分归一）+ confidence_level |
| margin | LOW 原因（LOW_MARGIN / LOW_SCORE） | margin 从"截断数量"改为置信分级参数；灵活 N 输出对应 confidence_level=HIGH 子集 |
| judge_eval 金标 | `job_major_gold_*` 三张表 | LLM 裁判→独立标注层，人工→仲裁层；格式转表 |
| eval_topn | 门禁 Gold 指标计算 | 补 MRR/NDCG（合同要排名指标，因它有 rank 列）；P/R 直接复用 |

## 9. 部署顺序（145 测试环境）

本地合同测试 → Codeup MR → CI 制品 → 145 DB preflight/expand → 部署外部 API/Worker → 注册 Compute 模板 `JOB_MAJOR_SEMANTIC_MATCH_TOPN_V1` → 1k smoke → 1万 Gold/性能 → 10万故障恢复 → 全量 Challenger → 页面验收 → 灰度。

代码归属：`baigong-platform/src/03_kn/03_knowledge_extraction/job_major_match/`（algorithm/contracts/migrations/ops/tests/artifacts 结构）。

## 10. 算法角色权限（DB 建角色时）

- 源表：仅 SELECT
- TOPN staging 四表：SELECT/INSERT/UPDATE
- pointer 表：无写权限
- 运行前后跑任务书 §12.3 的审计摘要 SQL，证明九字段未动
