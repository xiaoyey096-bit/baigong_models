#!/bin/bash
# =============================================================================
# Prefect 工作流启动脚本
# =============================================================================
# 用法:
#   bash serve.sh              # 启动 Prefect Server + 跑工作流
#   bash serve.sh server       # 仅启动 Web UI
#   bash serve.sh run          # 仅跑工作流（Server 已启动）
# =============================================================================
set -uo pipefail

GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 找 venv
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"  # workspace/
if [ -f "$WORKSPACE/venv/bin/activate" ]; then
    source "$WORKSPACE/venv/bin/activate"
elif [ -f "$SCRIPT_DIR/../../venv/bin/activate" ]; then
    source "$SCRIPT_DIR/../../venv/bin/activate"
fi

# 装 Prefect
pip install prefect -q 2>/dev/null
info "Prefect 版本: $(python -c 'import prefect; print(prefect.__version__)')"

ACTION="${1:-all}"

start_server() {
    info "启动 Prefect Server (Web UI: localhost:4200)..."
    prefect server start --host 0.0.0.0 &
    SERVER_PID=$!
    info "Server PID: $SERVER_PID"
    sleep 5
    info "✅ Web UI: http://localhost:4200"
}

run_flow() {
    info "启动工作流..."
    cd "$SCRIPT_DIR"
    python flow.py
    info "✅ 工作流完成"
}

case "$ACTION" in
    all)
        start_server
        run_flow
        info "工作流结束，Web UI 继续运行 (Ctrl+C 停止)"
        wait $SERVER_PID
        ;;
    server)
        start_server
        info "Web UI 运行中，另开终端执行: bash serve.sh run"
        wait $SERVER_PID
        ;;
    run)
        run_flow
        ;;
    *)
        echo "用法: bash serve.sh [all|server|run]"
        exit 1
        ;;
esac
