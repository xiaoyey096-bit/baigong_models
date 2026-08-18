"""
师融百工 — 全链路工作流（Prefect）
====================================
四阶段: Embedding → Reranker → 按专业分组 → 典型岗位归类
三机并行: DGX-40(大源) + DGX-2(中源) + DGX-3(小源)

用法:
    pip install prefect
    prefect server start              # 启动 Web UI (localhost:4200)
    python flow.py                    # 跑工作流（另开终端）

或一键:
    bash serve.sh
"""

from prefect import flow, get_run_logger
from config import NODES, SOURCE_ASSIGN
from tasks import distribute, run_embed, run_rerank, collect, run_stage3, run_stage4


@flow(name="师融百工_全链路", log_prints=True)
def main_flow():
    log = get_run_logger()
    log.info("════════════════════════════════════════════")
    log.info("  师融百工 全链路工作流 启动")
    log.info("════════════════════════════════════════════")

    local_dir = NODES["dgx40"]["workdir"]

    # ── 1. 分发代码+数据到 worker ──
    log.info("▼ 分发阶段")
    for node_name in ["dgx2", "dgx3"]:
        node = NODES[node_name]
        distribute(node_name, node["host"], node["user"], node["workdir"], local_dir)

    # ── 2. Stage 1+2 并行（三台同时跑）──
    log.info("▼ Stage 1+2 并行（三机）")
    all_rerank_futures = []

    for node_name, sources in SOURCE_ASSIGN.items():
        node = NODES[node_name]
        host = node["host"]
        user = node["user"]
        workdir = node["workdir"]

        for src in sources:
            # submit = 异步提交，不等待
            embed_future = run_embed.submit(host, user, workdir, src)
            rerank_future = run_rerank.submit(host, user, workdir, src, wait_for=[embed_future])
            all_rerank_futures.append(rerank_future)

    # 等全部 Reranker 完成
    for f in all_rerank_futures:
        f.wait()
    log.info("✅ Stage 1+2 全部完成")

    # ── 3. 回收结果 ──
    log.info("▼ 回收结果")
    for node_name in ["dgx2", "dgx3"]:
        node = NODES[node_name]
        collect(node["host"], node["user"], node["workdir"], local_dir)
    log.info("✅ 结果回收完成")

    # ── 4. Stage 3+4 本地 ──
    log.info("▼ Stage 3: 按专业分组")
    run_stage3(local_dir)
    log.info("✅ Stage 3 完成")

    log.info("▼ Stage 4: 典型岗位归类")
    run_stage4(local_dir)
    log.info("✅ Stage 4 完成")

    log.info("════════════════════════════════════════════")
    log.info("  🎉 全链路工作流完成")
    log.info("  分组结果: models/majors_data/majors_data_dwd_all/")
    log.info("  归类结果: models/jobsname/output/qwen_datarow_gt100/")
    log.info("════════════════════════════════════════════")


if __name__ == "__main__":
    main_flow()
