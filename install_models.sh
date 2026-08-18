#!/bin/bash
# =============================================================================
# 一键安装三个模型（ModelScope 国内源）
# =============================================================================
# BGE-large-zh-v1.5       (1.3GB)  Stage 1 岗位向量编码
# Qwen3-Reranker-0.6B     (1.2GB)  Stage 2 Top3 精排
# Qwen3-Embedding-4B      (7.6GB)  Stage 4 典型岗位归类
#
# 用法:
#   bash install_models.sh                          # 自动检测路径
#   bash install_models.sh /opt/models /opt/venv   # 手动指定
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 路径自动检测 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# MODELS_DIR: 优先用参数 → 否则用脚本所在目录（models/）→ 否则报错
if [ -n "${1:-}" ]; then
    MODELS_DIR="$1"
elif basename "$SCRIPT_DIR" | grep -qi "models"; then
    MODELS_DIR="$SCRIPT_DIR"
else
    MODELS_DIR="$SCRIPT_DIR/models"
fi

# VENV: 优先用参数 → 否则向上找 venv/ → 否则报错
if [ -n "${2:-}" ]; then
    VENV="$2"
elif [ -d "$SCRIPT_DIR/../../venv" ]; then
    VENV="$SCRIPT_DIR/../../venv"
elif [ -d "$SCRIPT_DIR/../venv" ]; then
    VENV="$SCRIPT_DIR/../venv"
elif [ -d "$HOME/workspace/venv" ]; then
    VENV="$HOME/workspace/venv"
else
    warn "未找到 venv，将在 $MODELS_DIR 下创建新虚拟环境"
    VENV="$MODELS_DIR/../venv"
    python3 -m venv "$VENV"
fi

# ── 系统依赖（triton 编译需要 Python.h）──
if ! dpkg -s python3-dev >/dev/null 2>&1; then
    info "安装系统依赖 python3-dev（triton 编译需要）..."
    sudo apt install -y python3-dev || warn "python3-dev 安装失败，若验证报错请手动执行: sudo apt install python3-dev"
fi

# ── 检查 ──
[ -f "$VENV/bin/python" ] || err "venv Python 不存在: $VENV/bin/python"
source "$VENV/bin/activate"
info "安装 Python 依赖（首次较慢，请耐心等待）..."
pip install --upgrade pip
pip install modelscope sentence-transformers transformers torch numpy huggingface_hub

info "MODELS_DIR: $MODELS_DIR"
info "VENV:       $VENV"
mkdir -p "$MODELS_DIR"

# ══════════════════════════════════════════════════════════════
# 1. BGE-large-zh-v1.5 (1.3GB)
# ══════════════════════════════════════════════════════════════
BGE_DIR="$MODELS_DIR/bge-large-zh-v1.5"
if [ -f "$BGE_DIR/pytorch_model.bin" ] || [ -f "$BGE_DIR/model.safetensors" ]; then
    info "✅ BGE-large-zh-v1.5 已存在，跳过"
else
    info "下载 BGE-large-zh-v1.5 (1.3GB)..."
    python -c "
from modelscope import snapshot_download
snapshot_download('BAAI/bge-large-zh-v1.5', local_dir='$BGE_DIR')
print('done')
"
    [ -f "$BGE_DIR/pytorch_model.bin" ] && info "✅ BGE 完成" || err "BGE 下载失败"
fi

# ══════════════════════════════════════════════════════════════
# 2. Qwen3-Reranker-0.6B (1.2GB) + CrossEncoder 配置补全
# ══════════════════════════════════════════════════════════════
RERANKER_DIR="$MODELS_DIR/Qwen3-Reranker-0.6B"
if [ -f "$RERANKER_DIR/model.safetensors" ] && [ -f "$RERANKER_DIR/modules.json" ]; then
    info "✅ Qwen3-Reranker-0.6B 已存在且配置完整，跳过"
else
    if [ ! -f "$RERANKER_DIR/model.safetensors" ]; then
        info "下载 Qwen3-Reranker-0.6B (1.2GB)..."
        python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-Reranker-0.6B', local_dir='$RERANKER_DIR')
print('done')
"
        [ -f "$RERANKER_DIR/model.safetensors" ] && info "✅ Reranker 权重完成" || err "Reranker 下载失败"
    fi

    # 补 CrossEncoder 配置（ModelScope 缺这 5 个文件，从 HF 镜像补）
    info "补全 Reranker CrossEncoder 配置..."
    export HF_ENDPOINT=https://hf-mirror.com
    python -c "
from huggingface_hub import hf_hub_download
files = ['modules.json', '1_LogitScore/config.json', 'config_sentence_transformers.json',
         'sentence_bert_config.json', 'chat_template.jinja']
for f in files:
    try:
        hf_hub_download('Qwen/Qwen3-Reranker-0.6B', f, local_dir='$RERANKER_DIR')
    except Exception as e:
        print(f'  ⚠️ {f}: {e}')
print('done')
"
    [ -f "$RERANKER_DIR/modules.json" ] && info "✅ CrossEncoder 配置补全" || warn "配置补全可能不完整，但可继续"
fi

# ══════════════════════════════════════════════════════════════
# 3. Qwen3-Embedding-4B (7.6GB)
# ══════════════════════════════════════════════════════════════
EMBED_DIR="$MODELS_DIR/Qwen3-Embedding-4B"
if [ -f "$EMBED_DIR/model.safetensors" ] || [ -f "$EMBED_DIR/model-00001-of-00002.safetensors" ]; then
    info "✅ Qwen3-Embedding-4B 已存在，跳过"
else
    info "下载 Qwen3-Embedding-4B (7.6GB, 最慢)..."
    python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-Embedding-4B', local_dir='$EMBED_DIR')
print('done')
"
    [ -f "$EMBED_DIR/model.safetensors" ] || [ -f "$EMBED_DIR/model-00001-of-00002.safetensors" ] \
        && info "✅ Embedding 完成" || err "Embedding 下载失败"
fi

# ══════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════
info "验证安装..."
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'设备: {dev}')

# BGE
m1 = SentenceTransformer('$BGE_DIR', device=dev)
v = m1.encode(['测试'])
print(f'✅ BGE dim={v.shape[1]}')

# Reranker
m2 = CrossEncoder('$RERANKER_DIR', device=dev, trust_remote_code=True)
import numpy as np
s = m2.predict([['测试岗位','测试专业']])
s = float(np.atleast_1d(s)[0])
print(f'✅ Reranker score={s:.4f}')

# Embedding 4B
m3 = SentenceTransformer('$EMBED_DIR', device=dev, model_kwargs={'torch_dtype': torch.float16})
v2 = m3.encode(['测试'], truncate_dim=1024)
print(f'✅ Qwen-Embedding-4B dim={v2.shape[1]}')

print()
print('三个模型全部就绪 ✅')
"

info "模型大小:"
du -sh "$BGE_DIR" "$RERANKER_DIR" "$EMBED_DIR" 2>/dev/null
