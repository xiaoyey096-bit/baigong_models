# 专业匹配流程（Embedding、规则、固定Top10）

当前目录实现四个正式步骤：

1. 对岗位和744个专业进行Embedding，生成每岗语义Top10。
2. 扫描完整`major_requirement_raw（原始专业要求）`，生成精确规则与高置信模糊规则候选。
3. 融合Embedding与规则候选，输出每岗固定最终Top10。
4. 使用全量精确专业召回基准和LLM语义盲审，对比Top5、Top10及不同融合方案。

当前不使用Reranker（重排序模型），也不建立Top20候选池。

## 目录

```text
new_top/
├── README.md
├── script/
│   ├── run_embedding_top10.py
│   ├── apply_major_rules.py
│   ├── merge_final_top10.py
│   └── evaluate_top_schemes.py
├── mapping/
│   └── fuzzy_major_mapping.json
├── evaluation/
│   ├── llm_blind_labels.csv
│   └── evaluation_report.md
└── output/
    ├── embedding_top10.csv
    ├── rule_matches.csv
    └── final_top10.csv
```

## 常用英文术语

| 英文术语 | 中文意思 |
|---|---|
| `Embedding` | 向量化/语义向量：把岗位或专业文本转换成可计算相似度的数字向量 |
| `Top10` | 排名前10的候选专业 |
| `NLP` | 自然语言处理；字段名中的`nlp`表示文本经过了自然语言处理 |
| `raw` | 原始值，基本保留数据源内容 |
| `std` | 标准化后的值 |
| `rank` | 排名，数字越小排名越靠前 |
| `score` | 分数；当前指Embedding向量相似度，数值越大表示语义越相似 |
| `rule` | 专业名称规则或模糊映射规则 |
| `mapping` | 映射：把岗位中的专业说法对应到744个标准专业 |
| `Reranker` | 重排序模型；当前流程不使用 |
| `Eval` / `Evaluation` | 评测：用独立真实标签检验Top5、Top10的效果 |
| `Recall` | 召回率：应当找出的正确专业中，实际找出了多少 |
| `Precision` | 准确率：实际返回的专业中，有多少确实正确 |

## 第一步：Embedding Top10

岗位文件：`models/hbb/高质量10w样本_专科清洗.csv`

岗位Embedding只使用：

1. `major_requirement_raw（原始专业要求）`
2. `job_title_nlp（岗位名称）`
3. `job_description_std（岗位职责）`
4. `job_requirement_std（任职要求）`
5. `ability_tags（能力标签）`

`major_requirement_raw`会排除“不限专业、无专业要求”等空信息；`ability_tags`会排除沟通、团队、责任心、抗压等通用软素质。

专业文件：`models/models_scripts/majors_usedtoembed/new_majors.xlsx`。专业Embedding唯一输入为`embedding_text（专业向量文本）`，模型为`models/bge-large-zh-v1.5`。

运行：

```bash
cd /home/cl-715/workspace/models/models_scripts/new_top
/home/cl-715/workspace/venv/bin/python script/run_embedding_top10.py --device auto --overwrite
```

输出：`output/embedding_top10.csv`，共90,807个岗位、908,070行，每岗固定10行。

### `embedding_top10.csv`字段说明

| 英文字段 | 中文名称 | 具体含义 |
|---|---|---|
| `job_id` | 岗位唯一标识 | 用于区分岗位，同一个岗位的10行结果使用相同ID |
| `job_title_nlp` | 处理后的岗位名称 | 经过NLP清洗、归一后的岗位名称 |
| `major_requirement_raw` | 原始专业要求 | 岗位数据原本填写的专业要求，也是第二步规则匹配的依据 |
| `semantic_rank` | Embedding语义排名 | 仅按向量相似度得到的原始排名，取值1—10 |
| `major_code` | 专业代码 | 744专业表中的标准专业代码 |
| `major_name` | 专业名称 | 744专业表中的标准专业名称 |
| `embedding_score` | Embedding相似度 | 岗位向量与专业向量的余弦相似度，越大表示语义越相似；它不是概率 |


## 第二步：精确规则与模糊规则

输入为第一步结果、`models/majors_intro.jsonl`和`mapping/fuzzy_major_mapping.json`。规则扫描完整原始专业要求，而不只检查Embedding Top10。

匹配口径：

1. 与标准专业名称完整相等：`rule_match_type=1（精确专业名）`。
2. 与标准小类名称完整相等：展开小类专业，`rule_match_type=2（模糊映射）`。
3. 命中审核过的别名：展开目标小类或专业，`rule_match_type=2（模糊映射）`。
4. 不使用子串包含判断，避免“艺术设计”误命中“数字媒体艺术设计”。
5. 同一岗位、同一专业重复命中时只保留一行，类型1优先于类型2，并合并同级证据。

运行：

```bash
/home/cl-715/workspace/venv/bin/python script/apply_major_rules.py --overwrite
```

输出：`output/rule_matches.csv`。映射表只保留高置信规则，脚本会拒绝重复别名、重复目标、无效小类和无效专业。

### `rule_matches.csv`字段说明

| 英文字段 | 中文名称 | 具体含义 |
|---|---|---|
| `job_id` | 岗位唯一标识 | 表示该规则候选来自哪个岗位 |
| `major_requirement_raw` | 原始专业要求 | 未替换为标准专业名的岗位原始专业要求 |
| `rule_match_type` | 规则命中类型 | `1`表示精确专业名称命中；`2`表示标准小类或模糊映射命中 |
| `rule_evidence` | 规则证据 | 真正在原始专业要求中触发匹配的词，例如“机械工程”；多项证据用中文分号连接 |
| `mapping_source` | 映射来源 | 说明候选通过标准专业名、标准小类还是审核别名得到，取值见下表 |
| `mapping_confidence` | 映射置信等级 | `exact`表示标准专业名直接相等；`high`表示高置信标准小类或审核别名映射 |
| `major_code` | 专业代码 | 规则找到的标准专业代码 |
| `major_name` | 专业名称 | 规则找到的标准专业名称 |

### `mapping_source`英文取值说明

| 英文取值 | 中文名称 | 举例与解释 |
|---|---|---|
| `standard_major_name` | 标准专业名称命中 | 原始要求中的词与744专业表的专业名称完全相同。例如“软件技术”直接命中“软件技术” |
| `standard_category` | 标准专业小类命中 | 原始要求中的词与标准小类名称完全相同，然后展开小类包含的专业。例如“计算机类”展开为该小类的多个专业 |
| `curated_alias` | 审核过的专业别名映射 | 原始说法不是现行标准名称，由本次LLM辅助构建并审核的映射表转换。例如“软件工程”映射到“软件技术”等高职专业 |

如果同一岗位—专业通过多个同等级来源命中，`mapping_source`和`rule_evidence`可能用分号合并多个值。

### `mapping_confidence`英文取值说明

| 英文取值 | 中文名称 | 具体含义 |
|---|---|---|
| `exact` | 精确 | 原始专业词与标准专业名称完整相等，不是模型推测 |
| `high` | 高置信 | 来自标准小类展开或已审核的高置信别名映射；它是规则可信等级，不是Embedding分数或概率 |

### `fuzzy_major_mapping.json`字段说明

| 英文字段 | 中文名称 | 具体含义 |
|---|---|---|
| `schema_version` | 结构版本 | 映射表JSON结构的版本号 |
| `mapping_version` | 映射内容版本 | 用于标识当前使用的是哪一版模糊映射规则 |
| `created_by` | 创建者 | 说明映射表由谁或通过什么方式创建 |
| `description` | 映射表说明 | 当前映射表的用途和口径 |
| `rules` | 规则列表 | 所有模糊映射规则组成的列表 |
| `aliases` | 别名列表 | 岗位中可能出现的非标准专业说法，例如“机械工程” |
| `target_categories` | 目标专业小类 | 别名要展开到哪些标准小类；会产生小类下的专业候选 |
| `target_major_names` | 目标专业名称 | 别名直接对应的一个或多个标准专业名称 |
| `confidence` | 规则置信等级 | 当前正式规则使用`high（高置信）` |
| `note` | 规则备注 | 解释为什么这样映射，方便后续人工复核 |

## 第三步：融合并固定最终Top10

输入为`embedding_top10.csv`、`rule_matches.csv`和`new_majors.jsonl`。

融合优先级严格固定为：

1. `embedding_and_rule`：同时被Embedding Top10与规则命中，按`embedding_score（向量相似度）`降序。
2. `rule_only`：只被规则命中，即原本位于Embedding Top10外；精确规则优先于模糊规则，其余使用映射来源、规则证据和专业代码稳定排序。
3. `embedding_only`：只被Embedding召回，按`embedding_score`降序。

取排序后的前10个专业。规则命中Top10外的专业会插入并挤掉Embedding尾部，但每个岗位始终固定10行。由于第一步只计算Embedding Top10，`rule_only`候选没有Embedding分数，这是当前已确认口径，不会伪造分数。

运行：

```bash
/home/cl-715/workspace/venv/bin/python script/merge_final_top10.py --overwrite
```

输出：`output/final_top10.csv`，共90,807个岗位、908,070行，每岗固定10行。

### `final_top10.csv`字段说明

| 英文字段 | 中文名称 | 具体含义 |
|---|---|---|
| `job_id` | 岗位唯一标识 | 同一岗位固定有10行最终候选 |
| `job_title_nlp` | 处理后的岗位名称 | 经过NLP清洗、归一后的岗位名称 |
| `major_requirement_raw` | 原始专业要求 | 岗位数据原本填写的专业要求 |
| `final_rank` | 最终融合排名 | 融合规则与Embedding后的排名，取值1—10；这是最终使用的排名 |
| `major_code` | 专业代码 | 最终候选专业的标准代码 |
| `major_name` | 专业名称 | 最终候选专业的标准名称 |
| `fusion_priority` | 融合优先级 | `1`同时被Embedding和规则命中；`2`仅规则命中；`3`仅Embedding命中 |
| `fusion_source` | 融合来源 | `embedding_and_rule`、`rule_only`或`embedding_only`，中文解释见下表 |
| `semantic_rank` | 原Embedding排名 | 专业原本在Embedding Top10中的排名；`rule_only`候选原本不在Top10内，所以为空 |
| `embedding_score` | 原Embedding相似度 | 原本的向量相似度；`rule_only`候选没有计算Top10外分数，所以为空 |
| `rule_match_type` | 规则命中类型 | `0`未命中规则；`1`精确专业名称；`2`标准小类或模糊映射 |
| `rule_evidence` | 规则证据 | 触发规则的原始专业词；未命中规则时为空 |
| `mapping_source` | 映射来源 | 取值为`standard_major_name`、`standard_category`、`curated_alias`；未命中规则时为空 |
| `mapping_confidence` | 映射置信等级 | `exact`或`high`；未命中规则时为空 |
| `embedding_text` | 专业向量文本 | 专业进行Embedding时使用的精简语义描述；保留此字段便于人工评测 |

### `fusion_source`英文取值说明

| 英文取值 | 中文名称 | 排序方式 |
|---|---|---|
| `embedding_and_rule` | Embedding与规则同时命中 | 融合优先级1，排在最前；同组按`embedding_score`从高到低排序 |
| `rule_only` | 仅规则命中 | 融合优先级2；专业原本不在Embedding Top10中，由规则插入并挤掉尾部候选 |
| `embedding_only` | 仅Embedding命中 | 融合优先级3；排在规则候选之后，同组按`embedding_score`从高到低排序 |

### 数字类型字段速查

| 字段和值 | 中文意思 |
|---|---|
| `rule_match_type=0` | 没有命中任何专业规则，只由Embedding召回 |
| `rule_match_type=1` | 原始专业词与标准专业名称精确相等 |
| `rule_match_type=2` | 命中标准专业小类或审核过的模糊别名 |
| `fusion_priority=1` | Embedding与规则同时命中，最高优先级 |
| `fusion_priority=2` | 仅规则命中，第二优先级 |
| `fusion_priority=3` | 仅Embedding命中，第三优先级 |

本次全量融合有29,201个岗位插入了Top10外规则候选，共插入120,140行并挤掉相同数量的Embedding尾部候选。

## 第四步：评测Top5、Top10及融合方案

评测同时使用两类证据：

1. 全量精确专业召回基准：检查`major_requirement_raw（原始专业要求）`中明确写出的标准专业是否进入Top5/Top10。
2. LLM语义盲审：对50个分层抽样岗位、704个候选专业独立判断相关性；标注时隐藏方案来源、排名、分数和规则类型，再按全量岗位分层占比加权。

运行：

```bash
cd /home/cl-715/workspace/models/models_scripts/new_top
/home/cl-715/workspace/venv/bin/python script/evaluate_top_schemes.py
```

正式报告：`evaluation/evaluation_report.md`。

盲审标签：`evaluation/llm_blind_labels.csv`。该文件采用长表结构，每行表示一个“岗位—候选专业”，不会展开为每岗80多个字段。

### `llm_blind_labels.csv`字段说明

| 英文字段 | 中文名称 | 具体含义 |
|---|---|---|
| `sample_id` | 抽样编号 | 盲审样本的顺序编号 |
| `stratum` | 抽样分层 | 岗位按规则插入数量及Embedding第1名是否被挤出划分的风险层 |
| `job_id` | 岗位唯一标识 | 用于关联岗位原始数据和各方案结果 |
| `job_title` | 岗位名称 | 供盲审判断的处理后岗位名称 |
| `major_requirement_raw` | 原始专业要求 | 招聘信息中填写的专业要求 |
| `job_description_std` | 标准化岗位职责 | 清洗后的岗位工作内容 |
| `job_requirement_std` | 标准化任职要求 | 清洗后的学历、经验、技能等要求 |
| `ability_tags` | 能力标签 | 从岗位中提取的技能或能力词 |
| `candidate_id` | 候选匿名编号 | 盲审阶段用于隐藏候选来源的编号 |
| `major_code` | 专业代码 | 被评测候选专业的标准代码 |
| `major_name` | 专业名称 | 被评测候选专业的标准名称 |
| `category` | 所属专业小类 | 候选专业所在的标准小类 |
| `role_summary` | 专业就业方向摘要 | 由专业向量文本提取的典型岗位或工作方向 |
| `llm_relevance_label` | LLM相关性标签 | `0`不相关；`1`合理相关；`2`高度匹配 |
| `label_meaning` | 标签中文含义 | 与数值标签对应的“不相关、合理相关、高度匹配” |
| `semantic_rank` | 纯Embedding排名 | 候选在纯Embedding Top10中的排名；未入选时为空 |
| `final_rank` | 当前融合排名 | 候选在当前融合Top10中的排名；未入选时为空 |

### 评测分层英文取值说明

| `stratum`取值 | 中文意思 |
|---|---|
| `A_no_injection` | 没有Top10外规则专业插入 |
| `B_low_injection_rank1_kept` | 插入1—4个规则专业，原Embedding第1名仍保留 |
| `C_low_injection_rank1_dropped` | 插入1—4个规则专业，原Embedding第1名被挤出 |
| `D_heavy_injection_rank1_dropped` | 插入5—10个规则专业，原Embedding第1名被挤出 |
| `E_heavy_injection_rank1_kept` | 插入5—10个规则专业，原Embedding第1名仍保留 |

### 评测指标中文说明

| 英文指标 | 中文意思 |
|---|---|
| `Precision@5/10` | Top5或Top10准确率：返回候选中相关专业所占比例 |
| `Recall@5/10` | Top5或Top10召回率：评测目标中的相关专业被找回的比例 |
| `NDCG@5/10` | 排序质量：不仅判断是否相关，还奖励高度相关专业排在更前面，越接近1越好 |
| `micro_recall` | 目标专业微平均召回率：把所有明确岗位—专业目标合在一起计算召回 |
| `full_job_recall` | 岗位全部召回率：一个岗位写出的所有明确专业均被找回才算成功 |

### 当前评测结论

- 数据库仍建议固定保存Top10；`final_rank=1—5`可作为核心匹配层，`final_rank=6—10`作为扩展召回层。
- `rule_match_type=1（精确专业名称）`允许从Embedding Top10外插入，能明显提升明确专业召回。
- `rule_match_type=2（标准小类或模糊映射）`不宜从Top10外批量插入；推荐仅在该专业已经位于Embedding Top10内时提升排序。
- 当前`output/final_top10.csv`仍保留原融合逻辑，评测步骤不会自动覆盖正式结果。采用推荐方案后，需要修改第三步脚本并重新生成Top10。
