"""各阶段任务（Prefect @task）"""

from prefect import task, get_run_logger
from remote import ssh_run, ssh_check, rsync_push, rsync_pull
import subprocess


@task(name="分发代码数据", retries=1)
def distribute(node_name: str, host: str, user: str, workdir: str, local_dir: str):
    """rsync 代码 + 数据到远程节点"""
    log = get_run_logger()
    log.info(f"分发到 {node_name} ({host})")

    # 代码
    rsync_push(host, user, f"{local_dir}/models/", f"{workdir}/models/")
    # 原始数据
    rsync_push(host, user, f"{local_dir}/data/old_data/", f"{workdir}/data/old_data/")
    # 种子文件（现役命名 典型岗位名称_gt100_{range}.jsonl，按范围构建；这里通配分发已构建的种子）
    rsync_push(host, user, f"{local_dir}/models/典型岗位名称_gt100_*.jsonl", f"{workdir}/models/")

    # 检查模型
    if not ssh_check(host, user, f"{workdir}/models/bge-large-zh-v1.5/pytorch_model.bin"):
        log.warning(f"{node_name} 模型不存在，请远程执行 install_models.sh")
    else:
        log.info(f"  ✅ {node_name} 模型就绪")


@task(name="Stage1_Embedding", retries=1, timeout_seconds=86400)
def run_embed(host: str, user: str, workdir: str, source: str):
    """跑 Embedding（本地或远程）"""
    log = get_run_logger()
    venv = f"{workdir}/venv/bin/activate"
    scr = f"{workdir}/models/models_scripts/scr"
    cmd = f"cd {scr} && source {venv} && python run_embed.py --source {source}"

    if host == "localhost":
        log.info(f"  [本地] Embedding: {source}")
        p = subprocess.run(cmd, shell=True)
        if p.returncode != 0:
            raise RuntimeError(f"Embedding 失败: {source}")
    else:
        log.info(f"  [{host}] Embedding: {source}")
        rc, out, err = ssh_run(host, user, cmd)
        if rc != 0:
            raise RuntimeError(f"Embedding 失败: {source}\n{err}")


@task(name="Stage2_Reranker", retries=1, timeout_seconds=86400)
def run_rerank(host: str, user: str, workdir: str, source: str):
    """跑 Reranker"""
    log = get_run_logger()
    venv = f"{workdir}/venv/bin/activate"
    scr = f"{workdir}/models/models_scripts/scr"
    cmd = f"cd {scr} && source {venv} && python run_rerank.py --source {source}"

    if host == "localhost":
        log.info(f"  [本地] Reranker: {source}")
        p = subprocess.run(cmd, shell=True)
        if p.returncode != 0:
            raise RuntimeError(f"Reranker 失败: {source}")
    else:
        log.info(f"  [{host}] Reranker: {source}")
        rc, out, err = ssh_run(host, user, cmd)
        if rc != 0:
            raise RuntimeError(f"Reranker 失败: {source}\n{err}")


@task(name="回收结果", retries=1)
def collect(host: str, user: str, workdir: str, local_dir: str):
    """rsync 回收远程 outputs_json"""
    log = get_run_logger()
    if host == "localhost":
        return
    log.info(f"回收 {user}@{host}")
    remote_path = f"{workdir}/models/models_scripts/outputs_json/*.jsonl"
    local_path = f"{local_dir}/models/models_scripts/outputs_json/"
    rsync_pull(host, user, remote_path, local_path)
    log.info(f"  ✅ 回收完成")


@task(name="Stage3_分组", retries=1)
def run_stage3(workdir: str):
    """按专业分组"""
    log = get_run_logger()
    log.info("Stage 3: 按专业分组")
    venv = f"{workdir}/venv/bin/activate"
    cmd = f"cd {workdir}/models/majors_data/scripts && source {venv} && python build_by_major.py"
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError("Stage 3 失败")


@task(name="Stage4_归类", retries=1, timeout_seconds=7200)
def run_stage4(workdir: str):
    """典型岗位归类"""
    log = get_run_logger()
    log.info("Stage 4: 典型岗位归类")
    venv = f"{workdir}/venv/bin/activate"
    cmd = f"cd {workdir}/models/jobsname/scripts/qwen && source {venv} && python run_classify.py"
    p = subprocess.run(cmd, shell=True)
    if p.returncode != 0:
        raise RuntimeError("Stage 4 失败")
