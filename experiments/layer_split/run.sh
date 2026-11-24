#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
model=${MODEL:-"$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"}
cuda_python=${CUDA_PYTHON:-"$HOME/.venvs/official0111-cuda-20260903-121127/bin/python"}
npu_python=${NPU_PYTHON:-"$HOME/.venvs/official0111-rbln-20260903-121127/bin/python"}
bundle=${BUNDLE:-"$PWD/build/suffix"}
export CUDA_VISIBLE_DEVICES=0 RBLN_DEVICES=0 HF_HUB_OFFLINE=1
export RBLN_RUNTIME_TIMER=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p logs
run_logs=$(mktemp -d "$PWD/logs/run-XXXXXX")
echo "Logs: $run_logs"

# Recheck the two physical devices before allocating either.
nvidia-smi -i 0 > "$run_logs/idle-gpu.log"
if [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU 0 is busy" >&2
    exit 2
fi
rbln-stat -d 0 -j > "$run_logs/idle-npu.json"
"$cuda_python" - "$run_logs/idle-npu.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as src:
    status = json.load(src)
if status["contexts"] or int(status["devices"][0]["memory"]["used"]):
    sys.exit("NPU 0 is busy")
PY

"$cuda_python" -u gpu.py --model "$model" --out "$run_logs/reference.pt" 2>&1 | tee "$run_logs/reference.log"
if [[ ! -d "$bundle" ]]; then
    "$npu_python" -u suffix.py "$model" --bundle "$bundle" 2>&1 | tee "$run_logs/compile.log"
fi
"$cuda_python" -u compare.py --model "$model" --npu-python "$npu_python" --bundle "$bundle" \
    --reference "$run_logs/reference.pt" --logs "$run_logs" 2>&1 | tee "$run_logs/split.log"
