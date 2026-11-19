#!/usr/bin/env bash
# Capture the negotiated PCI links, device identity and packages used by the two engines.
set -euo pipefail
out=${1:?usage: environment.sh OUTPUT_DIR}
mkdir -p "$out"
git rev-parse HEAD > "$out/source.txt"
git status --porcelain >> "$out/source.txt"
date --iso-8601=seconds > "$out/date.txt"
rbln-smi -j > "$out/rbln-smi.json"
rbln-smi -t > "$out/numa.txt"
lscpu >> "$out/numa.txt"
rg 'Cpus_allowed_list|Mems_allowed_list|VmLck' /proc/self/status >> "$out/numa.txt"
rg 'Max locked memory' /proc/self/limits >> "$out/numa.txt"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,vbios_version,pstate,memory.used --format=csv > "$out/nvidia-smi.csv"
GPU_PCI=$(nvidia-smi --id=0 --query-gpu=pci.bus_id --format=csv,noheader | sed 's/^00000000:/0000:/' | tr A-F a-f)
NPU_PCI=$("$RBLN_PY" -c 'import json,sys; print(next(d for d in json.load(open(sys.argv[1]))["devices"] if d["npu"]==0)["pci"]["bus_id"])' "$out/rbln-smi.json")
for pci in "$GPU_PCI" "$NPU_PCI"; do
  (
    cd "/sys/bus/pci/devices/$pci"
    for field in current_link_speed current_link_width max_link_speed max_link_width numa_node local_cpulist; do
      echo "$field=$(cat "$field")"
    done
  ) > "$out/pci-$pci.txt"
done
for backend in CUDA_PY RBLN_PY; do
  "${!backend}" - > "$out/$backend-packages.json" <<'PY'
import json, platform, sys
from importlib.metadata import PackageNotFoundError, version
packages = {}
for name in ("torch", "torch-rbln", "vllm", "vllm-rbln", "rebel-compiler",
             "numpy", "nixl", "nixl-cu12", "transformers", "tokenizers", "triton"):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        pass
print(json.dumps(dict(python=sys.version, kernel=platform.release(), packages=packages), indent=2))
PY
done
sha256sum "$(command -v rblnBandwidthLatencyTest)" > "$out/tool-sha256.txt"
