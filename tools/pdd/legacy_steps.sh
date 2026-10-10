#!/usr/bin/env bash
# Internal implementations preserved from the original PDD shell scripts.
# runner.py validates the state and holds the run/launch locks before dispatch.
set -Eeuo pipefail

STEP=${1:-}
if (( $# > 0 )); then
    shift
fi
case "$STEP" in
    prepare)
# BEGIN prepare
# Internal preflight; invoked only by runner.py under the run and launch locks.
umask 077
EXPECTED_BRANCH=0.11.1-2_native_dtype
REQUIRED_LAYOUT_COMMIT=972ce20cf53b0d2d50c4155f9d44be6879ede966
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

missing_python() {
    local name=$1
    printf 'ERROR: %s is required.\n' "$name" >&2
    printf 'Example:\n' >&2
    printf 'CUDA_PY=$HOME/.venvs/pdd-cuda/bin/python3 \\\n' >&2
    printf 'RBLN_PY=$HOME/.venvs/pdd-rbln/bin/python3 \\\n' >&2
    printf './start_pdd.sh\n' >&2
    exit 1
}

absolute_executable() {
    local supplied=$1
    local directory basename
    if [[ "$supplied" != */* ]]; then
        supplied=$(command -v -- "$supplied") || return 1
    fi
    directory=$(dirname -- "$supplied")
    basename=$(basename -- "$supplied")
    directory=$(cd -- "$directory" >/dev/null 2>&1 && pwd -P) || return 1
    printf '%s/%s\n' "$directory" "$basename"
}

write_env() {
    local destination=$1
    local name=$2
    local value=$3
    "$PDD_CONTROL_PY" -c 'import shlex, sys; print(sys.argv[1] + "=" + shlex.quote(sys.argv[2]))' "$name" "$value" >>"$destination"
}

[[ -n "${CUDA_PY:-}" ]] || missing_python CUDA_PY
[[ -n "${RBLN_PY:-}" ]] || missing_python RBLN_PY
[[ -d "$REPO" ]] || die "repository directory is missing: $REPO"
REPO=$(cd -- "$REPO" >/dev/null 2>&1 && pwd -P) || \
    die "cannot resolve repository path: $REPO"
git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "not a Git repository: $REPO"

CUDA_PY=$(absolute_executable "$CUDA_PY") || die "cannot resolve CUDA_PY"
RBLN_PY=$(absolute_executable "$RBLN_PY") || die "cannot resolve RBLN_PY"
[[ -f "$CUDA_PY" && -x "$CUDA_PY" ]] || \
    die "CUDA Python is not an executable file: $CUDA_PY"
[[ -f "$RBLN_PY" && -x "$RBLN_PY" ]] || \
    die "RBLN Python is not an executable file: $RBLN_PY"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
RESULT_ROOT="${RESULT_ROOT:-$HOME/vllm-rbln-profile-results}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RBLN_DEVICE="${RBLN_DEVICE:-0}"
EXPECTED_COMPILER="${EXPECTED_COMPILER:-0.11.1.post2.dev2+g2995098f.prod}"
EXPECTED_CUDA_VLLM="${EXPECTED_CUDA_VLLM:-0.22.0}"
EXPECTED_RBLN_VLLM="${EXPECTED_RBLN_VLLM:-0.22.0+cpu}"
OFFLINE="${OFFLINE:-0}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-$MODEL_DTYPE}"
VLLM_RBLN_PDD_LAYOUT_REORDER="${VLLM_RBLN_PDD_LAYOUT_REORDER:-0}"
UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"
PDD_RUN_TOKEN="${PDD_RUN_TOKEN:-}"
PROXY_SCRIPT="$REPO/pdd_profile_proxy.py"

[[ "$CUDA_DEVICE" =~ ^[0-9]+$ ]] || die "CUDA_DEVICE must be a nonnegative integer"
[[ "$RBLN_DEVICE" =~ ^[0-9]+$ ]] || die "RBLN_DEVICE must be a nonnegative integer"
[[ "$OFFLINE" == 0 || "$OFFLINE" == 1 ]] || die "OFFLINE must be 0 or 1"
[[ "$VLLM_RBLN_PDD_LAYOUT_REORDER" == 0 || \
   "$VLLM_RBLN_PDD_LAYOUT_REORDER" == 1 ]] || \
    die "VLLM_RBLN_PDD_LAYOUT_REORDER must be 0 or 1"
[[ "$MODEL_DTYPE" == float16 || "$MODEL_DTYPE" == bfloat16 ]] || \
    die "MODEL_DTYPE must be float16 or bfloat16"
[[ "$KV_CACHE_DTYPE" == float16 || "$KV_CACHE_DTYPE" == bfloat16 ]] || \
    die "KV_CACHE_DTYPE must be float16 or bfloat16"
if [[ -n "$PDD_RUN_TOKEN" && ! "$PDD_RUN_TOKEN" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "PDD_RUN_TOKEN may contain only letters, digits, dot, underscore, and hyphen"
fi
if [[ "$MODEL_PATH" == /* ]]; then
    [[ -d "$MODEL_PATH" ]] || die "local MODEL_PATH directory is missing: $MODEL_PATH"
    MODEL_PATH=$(realpath -e -- "$MODEL_PATH") || die "cannot resolve MODEL_PATH"
fi
[[ -f "$PROXY_SCRIPT" ]] || die "repository proxy is missing: $PROXY_SCRIPT"
command -v curl >/dev/null 2>&1 || die "curl is required for readiness checks"
command -v setsid >/dev/null 2>&1 || die "setsid is required for process-group isolation"
command -v flock >/dev/null 2>&1 || die "flock is required for launch serialization"
command -v git >/dev/null 2>&1 || die "git is required"

CURRENT_COMMIT=$(git -C "$REPO" rev-parse HEAD) || die "cannot resolve current commit"
BRANCH=$(git -C "$REPO" branch --show-current) || die "cannot resolve current branch"
if [[ -n "$BRANCH" && "$BRANCH" != "$EXPECTED_BRANCH" ]]; then
    die "branch mismatch: expected $EXPECTED_BRANCH or detached HEAD, got $BRANCH"
fi
git -C "$REPO" merge-base --is-ancestor \
    "$REQUIRED_LAYOUT_COMMIT" "$CURRENT_COMMIT" || \
    die "required layout commit is not an ancestor: $REQUIRED_LAYOUT_COMMIT"
[[ -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" ]] || \
    die "repository working tree is not clean"

if ! COMPILER_VERSION=$(
    "$RBLN_PY" -c '
import importlib.metadata as md
print(md.version("rebel-compiler"))
'
); then
    die "cannot read the installed rebel-compiler distribution from RBLN_PY"
fi
CUDA_VLLM_VERSION=$(
    "$CUDA_PY" -c 'import importlib.metadata as md; print(md.version("vllm"))'
) || die "cannot read CUDA vLLM version"
RBLN_VLLM_VERSION=$(
    "$RBLN_PY" -c 'import importlib.metadata as md; print(md.version("vllm"))'
) || die "cannot read RBLN vLLM version"
[[ "$COMPILER_VERSION" == "$EXPECTED_COMPILER" ]] || \
    die "compiler mismatch: expected $EXPECTED_COMPILER, got $COMPILER_VERSION"
[[ "$CUDA_VLLM_VERSION" == "$EXPECTED_CUDA_VLLM" ]] || \
    die "CUDA vLLM mismatch: expected $EXPECTED_CUDA_VLLM, got $CUDA_VLLM_VERSION"
[[ "$RBLN_VLLM_VERSION" == "$EXPECTED_RBLN_VLLM" ]] || \
    die "RBLN vLLM mismatch: expected $EXPECTED_RBLN_VLLM, got $RBLN_VLLM_VERSION"

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
CUDA_GPU_LINE=$(nvidia-smi -i "$CUDA_DEVICE" \
    --query-gpu=index,uuid,name --format=csv,noheader,nounits) || \
    die "CUDA device $CUDA_DEVICE is unavailable"
CUDA_GPU_UUID=$(printf '%s\n' "$CUDA_GPU_LINE" | cut -d, -f2 | tr -d ' ')
[[ -n "$CUDA_GPU_UUID" ]] || die "CUDA device query returned an unexpected result"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | \
    tr -d ' ' | grep -Fxq "$CUDA_GPU_UUID"; then
    die "CUDA device $CUDA_DEVICE already has a compute process; no existing process was changed"
fi

command -v rbln-smi >/dev/null 2>&1 || die "rbln-smi is unavailable"
RBLN_STATE=$(rbln-smi --json) || die "RBLN device query failed"
printf '%s' "$RBLN_STATE" | "$CUDA_PY" -c '
import json
import sys

device_id = sys.argv[1]
doc = json.load(sys.stdin)
devices = [d for d in doc.get("devices", []) if str(d.get("npu")) == device_id]
if len(devices) != 1 or devices[0].get("status") != "normal":
    raise SystemExit(f"RBLN device {device_id} is unavailable")
for context in doc.get("contexts", []):
    marker = str(context.get("npu", context.get("device", ""))).lower()
    if marker in {device_id, f"rbln{device_id}"}:
        raise SystemExit(f"RBLN device {device_id} already has a context")
' "$RBLN_DEVICE" || \
    die "RBLN device $RBLN_DEVICE is not available for an isolated run"

STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -- "$RESULT_DIR/cache" "$RESULT_DIR/cache/cuda" "$RESULT_DIR/cache/rbln"
read -r PREFILL_PORT DECODE_PORT PROXY_PORT PREFILL_SIDE_PORT DECODE_SIDE_PORT < <(
    "$CUDA_PY" -c '
import socket

sockets = []
try:
    for _ in range(5):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        sockets.append(sock)
    ports = [sock.getsockname()[1] for sock in sockets]
    if len(set(ports)) != 5:
        raise RuntimeError("duplicate port allocation")
    print(*ports)
finally:
    for sock in sockets:
        sock.close()
'
)
for port in "$PREFILL_PORT" "$DECODE_PORT" "$PROXY_PORT" \
            "$PREFILL_SIDE_PORT" "$DECODE_SIDE_PORT"; do
    [[ "$port" =~ ^[0-9]+$ ]] || die "free-port allocation failed"
done

PREFILL_ENGINE_ID="cuda-prefill-$RUN_ID"
DECODE_ENGINE_ID="rbln-decode-$RUN_ID"
PRODUCER_CONFIG=$(printf \
    '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_buffer_device":"cuda","kv_load_failure_policy":"fail","engine_id":"%s","kv_connector_extra_config":{"kv_recompute_threshold":0}}' \
    "$PREFILL_ENGINE_ID")
CONSUMER_CONFIG=$(printf \
    '{"kv_connector":"RblnNixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","engine_id":"%s","kv_connector_extra_config":{"kv_recompute_threshold":0}}' \
    "$DECODE_ENGINE_ID")

RUNTIME_FILE="$RESULT_DIR/runtime.env.tmp"
PIDS_FILE="$RESULT_DIR/pids.env.tmp"
: >"$RUNTIME_FILE"
: >"$PIDS_FILE"
for item in \
    "RESULT_DIR=$RESULT_DIR" \
    "RUN_ID=$RUN_ID" \
    "RESULT_ROOT=$RESULT_ROOT" \
    "REPO=$REPO" \
    "BRANCH=$BRANCH" \
    "COMMIT=$CURRENT_COMMIT" \
    "REQUIRED_LAYOUT_COMMIT=$REQUIRED_LAYOUT_COMMIT" \
    "CUDA_PY=$CUDA_PY" \
    "RBLN_PY=$RBLN_PY" \
    "CUDA_VLLM_VERSION=$CUDA_VLLM_VERSION" \
    "RBLN_VLLM_VERSION=$RBLN_VLLM_VERSION" \
    "COMPILER_VERSION=$COMPILER_VERSION" \
    "MODEL_PATH=$MODEL_PATH" \
    "MODEL_NAME=$MODEL_NAME" \
    "MODEL_DTYPE=$MODEL_DTYPE" \
    "KV_CACHE_DTYPE=$KV_CACHE_DTYPE" \
    "OFFLINE=$OFFLINE" \
    "PDD_RUN_TOKEN=$PDD_RUN_TOKEN" \
    "CUDA_DEVICE=$CUDA_DEVICE" \
    "CUDA_GPU_UUID=$CUDA_GPU_UUID" \
    "CUDA_GPU_LINE=$CUDA_GPU_LINE" \
    "RBLN_DEVICE=$RBLN_DEVICE" \
    "PREFILL_PORT=$PREFILL_PORT" \
    "DECODE_PORT=$DECODE_PORT" \
    "PROXY_PORT=$PROXY_PORT" \
    "PREFILL_SIDE_PORT=$PREFILL_SIDE_PORT" \
    "DECODE_SIDE_PORT=$DECODE_SIDE_PORT" \
    "PREFILL_ENGINE_ID=$PREFILL_ENGINE_ID" \
    "DECODE_ENGINE_ID=$DECODE_ENGINE_ID" \
    "PRODUCER_CONFIG=$PRODUCER_CONFIG" \
    "CONSUMER_CONFIG=$CONSUMER_CONFIG" \
    "VLLM_RBLN_PDD_LAYOUT_REORDER=$VLLM_RBLN_PDD_LAYOUT_REORDER" \
    "VLLM_KV_CACHE_LAYOUT=HND" \
    "UCX_NET_DEVICES=$UCX_NET_DEVICES" \
    "PRODUCER_KV_BUFFER_DEVICE=cuda" \
    "CONSUMER_KV_BUFFER_DEVICE=cpu" \
    "PRODUCER_LOCAL_MEMORY_TYPE=VRAM" \
    "CONSUMER_LOCAL_MEMORY_TYPE=DRAM" \
    "CONSUMER_REMOTE_MEMORY_TYPE=VRAM" \
    "TRANSFER_PATH=CUDA_VRAM_TO_RBLN_HOST_DRAM" \
    "PRODUCER_USE_HOST_BUFFER=false" \
    "PRODUCER_APPLICATION_D2H_EXPECTED=0" \
    "PYTHON_BIT_TRANSFORM_EXPECTED=0" \
    "DTYPE_CAST_EXPECTED=0" \
    "BLOCK_SIZE=128" \
    "MAX_NUM_BATCHED_TOKENS=128" \
    "STARTED_AT=$STAMP"; do
    write_env "$RUNTIME_FILE" "${item%%=*}" "${item#*=}"
done
write_env "$PIDS_FILE" RESULT_DIR "$RESULT_DIR"

mv -- "$RUNTIME_FILE" "$RESULT_DIR/runtime.env"
# runner.py publishes pids.env after sealing the runtime settings.
rm -- "$PIDS_FILE"
# END prepare
        ;;
    commands)
# BEGIN commands
# The command and environment arrays below are preserved from start_pdd.sh.
PROXY_SCRIPT="$REPO/pdd_profile_proxy.py"
COMMON_SERVER_ARGS=(
    "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host 127.0.0.1
    --dtype "$MODEL_DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len 1024
    --max-num-batched-tokens 128
    --max-num-seqs 1
    --tensor-parallel-size 1
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --seed 0
)

PREFILL_CMD=(
    "$CUDA_PY" -m vllm.entrypoints.cli.main serve
    "${COMMON_SERVER_ARGS[@]}"
    --port "$PREFILL_PORT"
    --block-size 128
    --num-gpu-blocks-override 32
    --kv-transfer-config "$PRODUCER_CONFIG"
)
DECODE_CMD=(
    "$RBLN_PY" -m vllm.entrypoints.cli.main serve
    "${COMMON_SERVER_ARGS[@]}"
    --port "$DECODE_PORT"
    --block-size 128
    --num-gpu-blocks-override 32
    --kv-transfer-config "$CONSUMER_CONFIG"
)
PROXY_CMD=(
    "$CUDA_PY" "$PROXY_SCRIPT"
    --host 127.0.0.1
    --port "$PROXY_PORT"
    --prefill-host 127.0.0.1
    --prefill-port "$PREFILL_PORT"
    --decode-host 127.0.0.1
    --decode-port "$DECODE_PORT"
)

OFFLINE_ENV=()
if [[ "$OFFLINE" == 1 ]]; then
    OFFLINE_ENV+=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
fi

CUDA_ENV=(env)
while IFS= read -r inherited_name; do
    if [[ "$inherited_name" == RBLN_* || "$inherited_name" == VLLM_RBLN_* ]]; then
        CUDA_ENV+=(-u "$inherited_name")
    fi
done < <(compgen -e)
CUDA_ENV+=(
    -u VLLM_PLUGINS
    -u VLLM_RBLN_ENFORCE_MODEL_FP32
    -u HF_HUB_OFFLINE
    -u TRANSFORMERS_OFFLINE
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    PYTHONPATH="$REPO"
    TOKENIZERS_PARALLELISM=false
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    VLLM_KV_CACHE_LAYOUT=HND
    UCX_NET_DEVICES="$UCX_NET_DEVICES"
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
    VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_PORT"
    VLLM_CACHE_ROOT="$RESULT_DIR/cache/cuda"
    "${OFFLINE_ENV[@]}"
)
RBLN_ENV=(env)
while IFS= read -r inherited_name; do
    if [[ "$inherited_name" == RBLN_* || "$inherited_name" == VLLM_RBLN_* ]]; then
        RBLN_ENV+=(-u "$inherited_name")
    fi
done < <(compgen -e)
RBLN_ENV+=(
    -u VLLM_PLUGINS
    -u VLLM_RBLN_ENFORCE_MODEL_FP32
    -u HF_HUB_OFFLINE
    -u TRANSFORMERS_OFFLINE
    PYTHONNOUSERSITE=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONHASHSEED=0
    PYTHONPATH="$REPO"
    TOKENIZERS_PARALLELISM=false
    # UCX must see the source GPU to read the producer's remote VRAM while
    # the RBLN connector still registers its local receive buffers as DRAM.
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
    RBLN_DEVICES="$RBLN_DEVICE"
    VLLM_KV_CACHE_LAYOUT=HND
    UCX_NET_DEVICES="$UCX_NET_DEVICES"
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1
    VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_PORT"
    VLLM_CACHE_ROOT="$RESULT_DIR/cache/rbln"
    VLLM_RBLN_USE_VLLM_MODEL=1
    VLLM_RBLN_USE_DEVICE_TENSOR=1
    VLLM_RBLN_COMPILE_MODEL=1
    VLLM_RBLN_COMPILE_STRICT_MODE=1
    VLLM_RBLN_SAMPLER=0
    VLLM_RBLN_PDD_LAYOUT_REORDER="$VLLM_RBLN_PDD_LAYOUT_REORDER"
    RBLN_USE_CUSTOM_KERNEL=0
    RBLN_ROOT_IP=127.0.0.1
    RBLN_LOCAL_IP=127.0.0.1
    "${OFFLINE_ENV[@]}"
)


case "$1" in
    PREFILL) exec "${CUDA_ENV[@]}" "${PREFILL_CMD[@]}" ;;
    DECODE) exec "${RBLN_ENV[@]}" "${DECODE_CMD[@]}" ;;
    PROXY) exec "${CUDA_ENV[@]}" "${PROXY_CMD[@]}" ;;
    *) echo "ERROR: invalid internal server role" >&2; exit 2 ;;
esac
# END commands
        ;;
    correctness)
# BEGIN correctness
# Internal implementation; the public runner validates and loads run state.

PROMPT_ID="${PROMPT_ID:-2}"
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}
[[ "$PROMPT_ID" == 2 || "$PROMPT_ID" == 4 ]] || \
    die "PROMPT_ID must be 2 or 4"

[[ "${PRODUCER_KV_BUFFER_DEVICE:-}" == cuda ]] || die "producer buffer is not CUDA"
[[ "${CONSUMER_KV_BUFFER_DEVICE:-}" == cpu ]] || die "consumer buffer is not CPU"
[[ "${PRODUCER_LOCAL_MEMORY_TYPE:-}" == VRAM ]] || die "producer memory is not VRAM"
[[ "${CONSUMER_LOCAL_MEMORY_TYPE:-}" == DRAM ]] || die "consumer local memory is not DRAM"
[[ "${CONSUMER_REMOTE_MEMORY_TYPE:-}" == VRAM ]] || die "consumer remote memory is not VRAM"
[[ "${TRANSFER_PATH:-}" == CUDA_VRAM_TO_RBLN_HOST_DRAM ]] || die "transfer path mismatch"
[[ "${PRODUCER_USE_HOST_BUFFER:-}" == false ]] || die "producer host buffer must be disabled"
[[ "${VLLM_RBLN_PDD_LAYOUT_REORDER:-}" == 0 || \
   "${VLLM_RBLN_PDD_LAYOUT_REORDER:-}" == 1 ]] || die "invalid layout reorder mode"
[[ "${PROXY_PORT:-}" =~ ^[0-9]+$ ]] || die "invalid proxy port"
[[ -x "${CUDA_PY:-}" ]] || die "recorded CUDA_PY is unavailable"
for pid in "${PREFILL_PID:-}" "${DECODE_PID:-}" "${PROXY_PID:-}"; do
    [[ "$pid" =~ ^[0-9]+$ ]] || die "pids.env contains an invalid PID"
    kill -0 "$pid" 2>/dev/null || die "recorded server PID $pid is not running"
done

successful_transfer_reports_before=$(
    awk \
        '/KV Transfer metrics: Num successful transfers=[1-9][0-9]*/ {count++}
         END {print count + 0}' \
        "$RESULT_DIR/decode.log"
)

export RESULT_DIR PROXY_PORT MODEL_NAME MODEL_DTYPE PROMPT_ID
"$CUDA_PY" - <<'PY'
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

result_dir = os.environ["RESULT_DIR"]
proxy_port = int(os.environ["PROXY_PORT"])
model = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
model_dtype = os.environ.get("MODEL_DTYPE", "unknown")
prompt_id = int(os.environ["PROMPT_ID"])

if prompt_id == 2:
    prompt = "Complete the sequence: 1, 1, 2, 3, 5,"
    prompt_sha256 = "99c6c5a7e43544bac9573914b469b0391ad40d529fb1ad137276bef173dc7116"
    expected_text = " 8, 13, 21, 34, 55, 89, 144, 233"
    expected_ids = [
        220, 23, 11, 220, 16, 18, 11, 220,
        17, 16, 11, 220, 18, 19, 11, 220,
        20, 20, 11, 220, 23, 24, 11, 220,
        16, 19, 19, 11, 220, 17, 18, 18,
    ]
else:
    prompt = (
        "A clear glass of water rests on the wooden table. " * 29
        + "Continue with the next three words only:"
    )
    prompt_sha256 = "57242b91f80632b4eec84b172bfe170fa52b0d33940af0b59a9a43be51889921"
    expected_text = None
    expected_ids = [
        279, 1790, 2326, 4244, 525, 1112, 30, 576,
        4226, 374, 330, 1782, 1790, 2326, 4244, 525,
        1112, 3263, 576, 4226, 374, 330, 1782, 1790,
        2326, 4244, 525, 1112, 3263, 576, 4226, 374,
    ]

actual_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
if actual_prompt_sha256 != prompt_sha256:
    raise SystemExit("embedded prompt checksum mismatch")

output_path = os.path.join(
    result_dir, f"correctness_{model_dtype}_prompt{prompt_id}_response.json"
)
body = {
    "model": model,
    "prompt": prompt,
    "temperature": 0,
    "top_p": 1,
    "seed": 0,
    "max_tokens": 32,
    "logprobs": 5,
    "n": 1,
    "return_token_ids": True,
    "return_tokens_as_token_ids": True,
    "stream": False,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{proxy_port}/v1/completions",
    data=json.dumps(body).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
        "X-Request-Id": f"pdd-correctness-p{prompt_id}-{os.getpid()}",
        "X-Prompt-Sha256": prompt_sha256,
    },
    method="POST",
)

status = 0
payload = None
try:
    with urllib.request.urlopen(request, timeout=900) as response:
        status = response.status
        raw = response.read()
except urllib.error.HTTPError as exc:
    status = exc.code
    raw = exc.read()
except Exception as exc:
    raw = b""
    payload = {"transport_error": f"{type(exc).__name__}: {exc}"}

if payload is None:
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {
            "http_status": status,
            "non_json_body": raw.decode("utf-8", errors="replace"),
        }

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")

errors = []
if not 200 <= status < 300:
    errors.append(f"HTTP status was {status}")
choices = payload.get("choices") if isinstance(payload, dict) else None
choice = choices[0] if isinstance(choices, list) and choices else {}
actual_text = choice.get("text") if isinstance(choice, dict) else None
actual_ids = choice.get("token_ids") if isinstance(choice, dict) else None
if not isinstance(actual_ids, list) or not all(
    isinstance(token_id, int) and not isinstance(token_id, bool)
    for token_id in actual_ids
):
    errors.append("choices[0].token_ids is not an integer array")
    actual_ids = []
if not actual_ids or actual_ids[0] != expected_ids[0]:
    errors.append(f"first token was {actual_ids[0] if actual_ids else None!r}")
if actual_ids != expected_ids:
    matching = sum(a == b for a, b in zip(actual_ids, expected_ids))
    errors.append(f"token IDs were not exact ({matching}/32 positions matched)")
if expected_text is not None and actual_text != expected_text:
    errors.append(f"text mismatch: {actual_text!r}")

if errors:
    for error in errors:
        print(f"CORRECTNESS_FAIL: {error}", file=sys.stderr)
    raise SystemExit(1)

print("CORRECTNESS_PASS")
print(f"PROMPT_ID={prompt_id}")
print(f"FIRST_TOKEN={expected_ids[0]}")
print("TOKEN_EXACT=32/32")
if expected_text is not None:
    print("TEXT_EXACT=PASS")
print(f"RESPONSE_FILE={output_path}")
PY

transfer_deadline=$((SECONDS + 30))
transfer_completed=0
while (( SECONDS < transfer_deadline )); do
    successful_transfer_reports_after=$(
        awk \
            '/KV Transfer metrics: Num successful transfers=[1-9][0-9]*/ {count++}
             END {print count + 0}' \
            "$RESULT_DIR/decode.log"
    )
    if (( successful_transfer_reports_after > successful_transfer_reports_before )); then
        transfer_completed=1
        break
    fi
    sleep 1
done
(( transfer_completed == 1 )) || \
    die "decode log does not confirm a successful NIXL transfer"

if grep -nEi \
    'NIXL_ERR_NOT_FOUND|DRAM_SEG lookup for CUDA|remote VRAM registration.*fail|backend mismatch|failed receive|recompute fallback|NIXL transfer failure' \
    "$RESULT_DIR/prefill.log" "$RESULT_DIR/decode.log" "$RESULT_DIR/proxy.log"; then
    die "transfer logs contain a prohibited error"
fi
printf 'NIXL_TRANSFER_COMPLETION=PASS\n'
# END correctness
        ;;
    stop)
# BEGIN stop
# Internal implementation; the public runner holds the run and launch locks.

proc_fields() {
    local pid=$1
    local stat_line tail
    [[ -r "/proc/$pid/stat" ]] || return 1
    IFS= read -r stat_line 2>/dev/null <"/proc/$pid/stat" || return 1
    tail=${stat_line##*) }
    set -- $tail
    printf '%s %s %s %s %s\n' "$1" "$2" "$3" "$4" "${20}"
}

recorded_pid_active() {
    local pid=$1
    local expected_starttime=$2
    local state ppid pgid sid starttime
    read -r state ppid pgid sid starttime < <(proc_fields "$pid") || return 1
    [[ "$starttime" == "$expected_starttime" ]] || return 1
    [[ "$state" != Z && "$state" != X ]] || return 1
}

group_exists() {
    local pgid=$1
    # Zombie processes retain a group but cannot receive signals or own devices.
    ps -eo pgid=,stat= | awk -v pgid="$pgid" '
        $1 == pgid && $2 !~ /^[ZX]/ { found = 1 }
        END { exit(found ? 0 : 1) }
    '
}

group_session_valid() {
    local pgid=$1
    local sid=$2
    ps -eo pgid=,sid= | awk -v pgid="$pgid" -v sid="$sid" '
        $1 == pgid { found = 1; if ($2 != sid) bad = 1 }
        END { exit(found && !bad ? 0 : 1) }
    '
}

is_recorded_descendant() {
    local candidate=$1
    local current=$candidate
    local state ppid pgid sid starttime
    local depth role auth_name pid_name
    for ((depth = 0; depth < 128; depth++)); do
        for role in PREFILL DECODE PROXY; do
            auth_name="${role}_AUTHORIZED"
            pid_name="${role}_PID"
            if [[ "${!auth_name:-0}" == 1 && "$current" == "${!pid_name:-}" ]]; then
                return 0
            fi
        done
        read -r state ppid pgid sid starttime < <(proc_fields "$current") || return 1
        [[ "$ppid" =~ ^[0-9]+$ && "$ppid" -gt 1 && "$ppid" != "$current" ]] || return 1
        current=$ppid
    done
    return 1
}

snapshot_unrelated() {
    local destination=$1
    local pid pgid args state ppid actual_pgid sid starttime
    local role auth_name pgid_name owned
    : >"$destination"
    while read -r pid pgid args; do
        [[ "$pid" =~ ^[0-9]+$ && "$pgid" =~ ^[0-9]+$ ]] || continue
        case "$args" in
            *vllm.entrypoints.cli.main*serve*|*pdd_profile_proxy.py*) ;;
            *) continue ;;
        esac
        owned=0
        for role in PREFILL DECODE PROXY; do
            auth_name="${role}_AUTHORIZED"
            pgid_name="${role}_PGID"
            if [[ "${!auth_name:-0}" == 1 && "$pgid" == "${!pgid_name:-}" ]]; then
                owned=1
                break
            fi
        done
        if (( owned == 1 )); then
            continue
        fi
        if is_recorded_descendant "$pid"; then
            continue
        fi
        read -r state ppid actual_pgid sid starttime < <(proc_fields "$pid") || continue
        [[ "$state" != Z && "$state" != X ]] || continue
        printf '%s\t%s\t%s\t%s\n' "$pid" "$starttime" "$pgid" "$args" >>"$destination"
    done < <(ps -eo pid=,pgid=,args=)
}

validate_role() {
    local role=$1
    local pid_name="${role}_PID"
    local pgid_name="${role}_PGID"
    local sid_name="${role}_SID"
    local start_name="${role}_STARTTIME"
    local pid=${!pid_name:-}
    local pgid=${!pgid_name:-}
    local sid=${!sid_name:-}
    local expected_starttime=${!start_name:-}
    local state ppid actual_pgid actual_sid actual_starttime
    local self_pgid

    printf -v "${role}_AUTHORIZED" '%s' 0
    if [[ -z "$pid" && -z "$pgid" && -z "$sid" && -z "$expected_starttime" ]]; then
        printf -v "${role}_RECORDED" '%s' 0
        return 0
    fi
    printf -v "${role}_RECORDED" '%s' 1
    [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]] || {
        printf 'ERROR: invalid %s PID record\n' "$role" >&2
        return 1
    }
    [[ "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 ]] || {
        printf 'ERROR: invalid %s PGID record\n' "$role" >&2
        return 1
    }
    [[ "$sid" =~ ^[0-9]+$ && "$sid" -gt 1 ]] || {
        printf 'ERROR: invalid %s SID record\n' "$role" >&2
        return 1
    }
    [[ "$expected_starttime" =~ ^[0-9]+$ ]] || {
        printf 'ERROR: invalid %s start-time record\n' "$role" >&2
        return 1
    }
    [[ "$pid" == "$pgid" && "$pid" == "$sid" ]] || {
        printf 'ERROR: %s was not recorded as an isolated session\n' "$role" >&2
        return 1
    }
    self_pgid=$(ps -o pgid= -p $$ | tr -d '[:space:]')
    [[ "$pgid" != "$self_pgid" ]] || {
        printf 'ERROR: refusing to signal the stop shell process group\n' >&2
        return 1
    }

    if read -r state ppid actual_pgid actual_sid actual_starttime < <(proc_fields "$pid"); then
        if [[ "$actual_starttime" != "$expected_starttime" ]]; then
            printf 'ERROR: %s PID %s was reused; its process group will not be signaled\n' \
                "$role" "$pid" >&2
            return 1
        fi
        [[ "$actual_pgid" == "$pgid" && "$actual_sid" == "$sid" ]] || {
            printf 'ERROR: %s process identity no longer matches its recorded group/session\n' \
                "$role" >&2
            return 1
        }
    fi

    if group_exists "$pgid"; then
        group_session_valid "$pgid" "$sid" || {
            printf 'ERROR: %s group/session membership is inconsistent\n' "$role" >&2
            return 1
        }
        printf -v "${role}_AUTHORIZED" '%s' 1
    fi
}

signal_role() {
    local role=$1
    local signal_name=$2
    local auth_name="${role}_AUTHORIZED"
    local pgid_name="${role}_PGID"
    local sid_name="${role}_SID"
    local pid_name="${role}_PID"
    local start_name="${role}_STARTTIME"
    local authorized=${!auth_name:-0}
    local pgid=${!pgid_name:-}
    local sid=${!sid_name:-}
    local pid=${!pid_name:-}
    local expected_starttime=${!start_name:-}
    local state ppid actual_pgid actual_sid actual_starttime

    [[ "$authorized" == 1 ]] || return 0
    group_exists "$pgid" || return 0
    if read -r state ppid actual_pgid actual_sid actual_starttime < <(proc_fields "$pid"); then
        if [[ "$actual_starttime" != "$expected_starttime" || \
              "$actual_pgid" != "$pgid" || "$actual_sid" != "$sid" ]]; then
            printf 'ERROR: refusing %s for reused or changed %s process identity\n' \
                "$signal_name" "$role" >&2
            return 1
        fi
    fi
    group_session_valid "$pgid" "$sid" || {
        printf 'ERROR: refusing %s for changed %s group/session\n' \
            "$signal_name" "$role" >&2
        return 1
    }
    if ! kill "-$signal_name" -- "-$pgid" 2>/dev/null; then
        group_exists "$pgid" || return 0
        printf 'ERROR: failed to send %s to recorded %s process group %s\n' \
            "$signal_name" "$role" "$pgid" >&2
        return 1
    fi
}

verify_unrelated() {
    local source=$1
    local pid expected_starttime pgid args
    local state ppid actual_pgid sid actual_starttime
    local failed=0
    while IFS=$'\t' read -r pid expected_starttime pgid args; do
        [[ -n "$pid" ]] || continue
        if ! read -r state ppid actual_pgid sid actual_starttime < <(proc_fields "$pid"); then
            printf 'ERROR: unrelated process %s is no longer present\n' "$pid" >&2
            failed=1
            continue
        fi
        if [[ "$actual_starttime" != "$expected_starttime" || "$state" == Z || "$state" == X ]]; then
            printf 'ERROR: unrelated process identity %s did not survive unchanged\n' "$pid" >&2
            failed=1
        fi
    done <"$source"
    return "$failed"
}

FAILURE=0
if [[ -n "${PDD_STATE_ERRORS:-}" ]]; then
    printf 'ERROR: unsafe state records were excluded: %s\n' "$PDD_STATE_ERRORS" >&2
    FAILURE=1
fi
validate_role PROXY || FAILURE=1
validate_role PREFILL || FAILURE=1
validate_role DECODE || FAILURE=1

UNRELATED_BEFORE="$RESULT_DIR/unrelated_processes_before_stop.txt"
UNRELATED_AFTER="$RESULT_DIR/unrelated_processes_after_stop.txt"
# Preserve diagnostics from every cleanup attempt, including failed attempts.
DIAGNOSTIC_SUFFIX=
if [[ -e "$UNRELATED_BEFORE" || -e "$UNRELATED_AFTER" ]]; then
    DIAGNOSTIC_SUFFIX=".$(date +%s).$$"
    UNRELATED_BEFORE="$RESULT_DIR/unrelated_processes_before_stop${DIAGNOSTIC_SUFFIX}.txt"
    UNRELATED_AFTER="$RESULT_DIR/unrelated_processes_after_stop${DIAGNOSTIC_SUFFIX}.txt"
fi
snapshot_unrelated "$UNRELATED_BEFORE"

signal_role PROXY TERM || FAILURE=1
signal_role PREFILL TERM || FAILURE=1
signal_role DECODE TERM || FAILURE=1

TERM_DEADLINE=$((SECONDS + 30))
while (( SECONDS < TERM_DEADLINE )); do
    ANY_GROUP=0
    for role in PROXY PREFILL DECODE; do
        auth_name="${role}_AUTHORIZED"
        pgid_name="${role}_PGID"
        if [[ "${!auth_name:-0}" == 1 ]] && group_exists "${!pgid_name}"; then
            ANY_GROUP=1
            break
        fi
    done
    (( ANY_GROUP == 0 )) && break
    sleep 0.2
done

signal_role PROXY KILL || FAILURE=1
signal_role PREFILL KILL || FAILURE=1
signal_role DECODE KILL || FAILURE=1

KILL_DEADLINE=$((SECONDS + 5))
while (( SECONDS < KILL_DEADLINE )); do
    ANY_GROUP=0
    for role in PROXY PREFILL DECODE; do
        auth_name="${role}_AUTHORIZED"
        pgid_name="${role}_PGID"
        if [[ "${!auth_name:-0}" == 1 ]] && group_exists "${!pgid_name}"; then
            ANY_GROUP=1
            break
        fi
    done
    (( ANY_GROUP == 0 )) && break
    sleep 0.1
done

for role in PROXY PREFILL DECODE; do
    pid_name="${role}_PID"
    start_name="${role}_STARTTIME"
    auth_name="${role}_AUTHORIZED"
    pgid_name="${role}_PGID"
    recorded_name="${role}_RECORDED"
    [[ "${!recorded_name:-0}" == 1 ]] || continue
    pid=${!pid_name:-}
    expected_starttime=${!start_name:-}
    pgid=${!pgid_name:-}
    if [[ "$pid" =~ ^[0-9]+$ && "$expected_starttime" =~ ^[0-9]+$ ]] && \
       recorded_pid_active "$pid" "$expected_starttime"; then
        printf 'ERROR: recorded %s PID %s is still active\n' "$role" "$pid" >&2
        FAILURE=1
    fi
    if [[ "${!auth_name:-0}" == 1 && "$pgid" =~ ^[0-9]+$ ]] && group_exists "$pgid"; then
        printf 'ERROR: recorded %s process group %s is still active\n' \
            "$role" "$pgid" >&2
        FAILURE=1
    fi
done

snapshot_unrelated "$UNRELATED_AFTER"
verify_unrelated "$UNRELATED_BEFORE" || FAILURE=1

# An interrupted run may never have launched decode. GPU/proxy-only cleanup
# must not require the NPU SDK or a functioning CUDA Python environment.
RBLN_CLEAR=not_applicable
if [[ "${DECODE_RECORDED:-0}" == 1 ]]; then
    RBLN_CLEAR=0
    RBLN_STATE_FILE="$RESULT_DIR/rbln_smi_after_stop${DIAGNOSTIC_SUFFIX}.json"
    if [[ ! "${RBLN_DEVICE:-}" =~ ^[0-9]+$ ]]; then
        printf 'ERROR: RBLN_DEVICE is missing or invalid for post-stop verification\n' >&2
        FAILURE=1
    elif ! command -v rbln-smi >/dev/null 2>&1; then
        printf 'ERROR: rbln-smi is unavailable for post-stop verification\n' >&2
        FAILURE=1
    elif [[ ! -x "${PDD_CONTROL_PY:-}" ]]; then
        printf 'ERROR: PDD_CONTROL_PY is unavailable for post-stop verification\n' >&2
        FAILURE=1
    else
        for ((attempt = 0; attempt < 10; attempt++)); do
            if rbln-smi --json >"$RBLN_STATE_FILE" 2>/dev/null && \
               "$PDD_CONTROL_PY" - "${DECODE_PID:-}" "$RBLN_STATE_FILE" "$RBLN_DEVICE" <<'PY'
import json
import sys

decode_pid = str(sys.argv[1])
device_id = str(sys.argv[3])
with open(sys.argv[2], encoding="utf-8") as handle:
    doc = json.load(handle)
for context in doc.get("contexts", []):
    marker = str(context.get("npu", context.get("device", ""))).lower()
    serialized = json.dumps(context, sort_keys=True)
    if marker in {device_id, f"rbln{device_id}"} or (
        decode_pid and decode_pid in serialized
    ):
        raise SystemExit(1)
PY
            then
                RBLN_CLEAR=1
                break
            fi
            sleep 1
        done
    fi
    if [[ "$RBLN_CLEAR" != 1 ]]; then
        printf 'ERROR: could not verify RBLN device-%s contexts cleared after cleanup\n' \
            "${RBLN_DEVICE:-unknown}" >&2
        FAILURE=1
    fi
fi

if (( FAILURE != 0 )); then
    exit 1
fi

printf 'PID_SCOPED_CLEANUP=PASS\n'
printf 'UNRELATED_PROCESSES_UNTOUCHED=PASS\n'
if [[ "$RBLN_CLEAR" == 1 ]]; then
    printf 'RBLN_CONTEXT_CLEAR=PASS\n'
else
    printf 'RBLN_CONTEXT_CLEAR=NOT_APPLICABLE (decode was not started)\n'
fi
printf 'STOP_COMPLETE\n'
printf 'RESULT_DIR=%s\n' "$RESULT_DIR"
# END stop
        ;;
    *)
        printf 'ERROR: unknown internal PDD step: %s\n' "$STEP" >&2
        exit 2
        ;;
esac
