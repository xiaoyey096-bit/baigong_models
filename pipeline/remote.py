"""SSH / rsync 远程操作工具"""

import subprocess
from config import SSH_PORT


def ssh_run(host: str, user: str, cmd: str, timeout: int = 86400) -> tuple:
    """SSH 执行远程命令，返回 (returncode, stdout, stderr)"""
    full_cmd = f"ssh -p {SSH_PORT} -o StrictHostKeyChecking=no {user}@{host} '{cmd}'"
    p = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def ssh_check(host: str, user: str, filepath: str) -> bool:
    """检查远程文件是否存在"""
    rc, _, _ = ssh_run(host, user, f"test -f '{filepath}' && echo ok", timeout=10)
    return rc == 0


def rsync_push(host: str, user: str, local: str, remote: str):
    """rsync 本地 → 远程"""
    cmd = f"rsync -avz -e 'ssh -p {SSH_PORT} -o StrictHostKeyChecking=no' {local} {user}@{host}:{remote}"
    subprocess.run(cmd, shell=True, check=True)


def rsync_pull(host: str, user: str, remote: str, local: str):
    """rsync 远程 → 本地"""
    cmd = f"rsync -avz -e 'ssh -p {SSH_PORT} -o StrictHostKeyChecking=no' {user}@{host}:{remote} {local}"
    subprocess.run(cmd, shell=True, check=True)
