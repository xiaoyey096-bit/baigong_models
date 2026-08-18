# majors_data — 按专业分组的岗位库

> 接收 `models_scripts/outputs_json_new/` 的 Top3 匹配结果，**反透视**成「每个专业 → 一批岗位」，并做质量分层、频次统计。是后续典型岗位归类（`jobsname/`）的输入源。

---

## 命名规范（重要）

产物按**月份范围**分目录，每个月份范围下挂 4 类结果库：

```
majors_data/
├── {月份范围}/                       # 首目录 = 累积月份范围，如 2026(01-06)
│   ├── majors_data_dwd_all/         # ① 有效匹配库（分数≥0.6 且全文非空）
│   │   └── {专业代码}_{专业名称}.jsonl
│   ├── majors_data_review_low/      # ② 低分复核库（分数<0.6 且全文非空）
│   ├── majors_data_review_empty/    # ③ 空全文异常
│   └── majors_data_review_nomatch/  # ④ 完全无匹配（Top3 全<0.6）
├── old/                             # 旧版扁平结构产物（归档保留）
├── scripts/                         # 数据构建与统计脚本
│   ├── build_by_major.py            # 按专业分组构建 4 个结果库（支持范围段合并）
│   ├── stat_by_major.py             # 统计每专业匹配岗位数 → xlsx（按 --range）
│   └── stat_job_freq.py             # 岗位名频次统计 → jsonl（按 --range，供种子归纳）
├── 专业匹配岗位统计_old.xlsx         # 旧版统计（归档）
└── 岗位词频统计_old.jsonl            # 旧版频次（归档）
```

> **两层范围区别**（详见 [工作流方案.md](../工作流方案.md)）：
> - `outputs_json_new/{范围段}/`：忠实对应原始爬取数据目录名（如 `2026-08-03(01-03)`），不累积。
> - `majors_data/{月份范围}/`：累积汇总标识，只标月份、不带爬取日期（如 `2026(01-06)`）。可由多个范围段合并而来。

---

## 数据流

```
models_scripts/outputs_json_new/2026-08-03(01-03)/*.jsonl  (一岗→Top3, 12列)
models_scripts/outputs_json_new/2026-08-03(04-06)/*.jsonl
    │
    ▼  build_by_major.py --input ... --output-range 2026(01-06)
    │  (反透视 + 质量分层 + 跨文件 job_id 去重)
    │
    └─→ majors_data/2026(01-06)/
        ├─ majors_data_dwd_all/      （有效，724文件）
        ├─ majors_data_review_low/   （低分复核）
        ├─ majors_data_review_empty/ （空全文异常）
        └─ majors_data_review_nomatch/（无匹配）
```

---

## 数据格式

### majors_data_dwd_all/（有效匹配库）

每行 6 列：
```json
{
  "岗位唯一标识": "JOBL20002000202606291",
  "原始岗位名称": "电商客服",
  "岗位全文": "负责电商平台售前咨询接待...",
  "匹配专业代码": "530701",
  "匹配专业名称": "电子商务",
  "匹配专业分数": 0.8597
}
```

### 复核库（low/empty/nomatch）

7 列 = 上述 6 列 + `复核原因`（低分 / 岗位全文为空 / 完全无匹配）。

---

## 各脚本说明

### `build_by_major.py` — 按专业分组构建（支持范围段合并）

```bash
# 筛选单个范围段 → 只处理 1-3 月
python build_by_major.py --input 2026-08-03(01-03) --output-range 2026(01-03)

# 合并多个范围段 → 1-3月 + 4-6月 汇总成 1-6月（跨文件按 job_id 去重）
python build_by_major.py \
    --input 2026-08-03(01-03) \
    --input 2026-08-03(04-06) \
    --output-range 2026(01-06)
```

- `--input`：outputs_json_new 下的范围段目录名，**可多次指定以合并**（必填）
- `--output-range`：输出的月份范围标识，产物写到 `majors_data/{此值}/`（必填）
- 读 outputs_json_new/{input}/*.jsonl → 反透视 + 质量分层
- 合并多范围段时**自动按 job_id 去重**，避免同一岗位重复计数
- 阈值 `THRESHOLD = 0.6`（代码顶部，可调）

### `stat_by_major.py` — 岗位数统计（按月份范围）

```bash
python stat_by_major.py --range 2026(01-06)
# → 专业匹配岗位统计_2026(01-06).xlsx
```

- 扫描 `majors_data/{range}/majors_data_dwd_all/` 下所有 jsonl
- 统计每专业匹配岗位数 → `专业匹配岗位统计_{range}.xlsx`（文件名带范围段，避免覆盖）

### `stat_job_freq.py` — 岗位名频次（按月份范围）

```bash
python stat_job_freq.py --range 2026(01-06)                  # 全量
python stat_job_freq.py --range 2026(01-06) --major 530701   # 单专业
# → 岗位词频统计_2026(01-06).jsonl
```

- 清洗岗位名（去括号/薪资/促销词）
- 按专业统计完整岗位名频次（全量不截断）
- 输出 `岗位词频统计_{range}.jsonl`（文件名带范围段），供 `jobsname/` 归纳典型岗位种子

> ⚠️ 后续 LLM 归纳种子时需自行抽样（累计覆盖90%或top-N），不要全量喂。
> 旧版文件已归档：`岗位词频统计_old.jsonl`、`专业匹配岗位统计_old.xlsx`。

---

## 数据规模（2026(01-06) 实测）

| 指标 | 数值 |
|------|:--|
| 输入岗位行（去重后） | 335,758 |
| ① 有效匹配 | 257,814 条（724 专业文件） |
| ② 低分复核 | 170,646 条（716 专业文件） |
| ③ 空全文异常 | 5,292 条 |
| ④ 完全无匹配 | 573,522 条 |

---

## 下游

`majors_data/{月份范围}/majors_data_dwd_all/` → `jobsname/` 典型岗位归类
