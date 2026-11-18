# 공식 llm-d로 이종 PDD 실행하기

이 브랜치의 실험 진입점은 `python -m llmd`이다. 한 배포에서 하나의 모델 revision과 tokenizer를 고정하고, 외부 클라이언트는 같은 `/v1/completions` 주소로 요청한다. 모델을 바꿀 때는 새 실행 디렉터리에서 다음 실험을 시작한다. [공식 llm-d v1.0.0](https://github.com/llm-d/llm-d/releases/tag/v1.0.0)

```text
client → Envoy → ext_proc ↔ llm-d EPP runner (native plugins and discovery)
           ├─ full → selected vLLM engine (consumer Pod uses its sidecar)
           └─ PDD  → selected decoder's official PD sidecar
                         ├─ prefill HTTP → producer engine → KV lease
                         └─ decode HTTP  → consumer engine
                                           └─ NIXL READ → host staging → device KV → decode
```

EPP 이미지는 외부 Go 플러그인을 등록한 후 upstream runner를 시작한다. 릴리스에 고정한 공식 sidecar 소스에는 요청 ID, 생산자, 토큰 및 블록 범위를 검사하고 선택된 preload를 전달하는 패치가 적용된다. 빌드는 공식 저장소를 지정된 커밋으로 가져오고 패치 회귀 테스트를 실행한다. 기존 제출용 셸 스크립트는 원래 실험에서 계속 사용할 수 있다. [빌드와 upstream 출처](../llmd/serving/README.md) · [정책 구현과 비용표](../llmd/epp/README.md) · [RBLN Host 수신 및 실행 제어](rbln_host_staging.md)

## 로컬에서 실행하기

Linux에서 두 가상환경을 준비하고 `CUDA_PY`, `RBLN_PY`에 각 환경의 Python 실행 파일을 지정한다. 이 저장소의 `docker/requirements-prefill.txt`와 `docker/requirements-decode.txt`가 해당 엔진의 패키지 잠금이다. RBLN 가상환경에는 이 저장소의 platform entry point가 필요하다. 지정된 버전, 실제 장치, compiler, 모델 snapshot을 시작할 때 검사한다. 로컬에서는 Python 3.10, RTX PRO 6000 Blackwell, RBLN-CA22, vLLM 0.22.0, PyTorch 2.11.0, NIXL 1.4.1 및 RBLN SDK 0.11.1 조합을 사용했다. [엔진별 버전 목록](../llmd/engines.py)

```bash
export LLMD_BIN_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hetero-pdd/llmd-v1/bin"
llmd/serving/build.sh "$LLMD_BIN_DIR" --images
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
hf download Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --cache-dir "$HF_HUB_CACHE"
run=$(mktemp -d)
"$CUDA_PY" -m llmd validate experiments/qwen2-fixed.yaml
"$CUDA_PY" -m llmd render experiments/qwen2-fixed.yaml --work-dir "$run/render"
"$CUDA_PY" -m llmd start experiments/qwen2-baseline.yaml --work-dir "$run/baseline" --timeout 2400
"$CUDA_PY" -m llmd ready --work-dir "$run/baseline"
"$CUDA_PY" -m llmd verify --work-dir "$run/baseline"
"$CUDA_PY" -m llmd stop --work-dir "$run/baseline"
"$CUDA_PY" -m llmd start experiments/qwen2-fixed.yaml --work-dir "$run/pdd"
"$CUDA_PY" -m llmd verify --work-dir "$run/pdd" --reference "$run/baseline/verification.json"
"$CUDA_PY" -m llmd stop --work-dir "$run/pdd"
```

`start`가 성공하면 서비스는 독립 프로세스로 유지된다. `ready`는 해당 실행이 실제 listener를 소유하는지, 엔진과 Envoy의 ext-proc upstream이 준비됐는지 검사한다. 시작 기본 timeout은 2,400초다. 성공 종료 코드는 0, 검증·설정·시작 실패는 1, 인터럽트는 130이다. 시작 실패 시 run ID가 일치하는 프로세스 트리를 회수한다. 같은 호스트에서 동시에 여러 배포를 실행하려면 장치와 listener를 각각 할당한다.

## 실험과 호환성

| 입력 | 의미 |
| --- | --- |
| `apiVersion: hetero-pdd/v2` | 실험 스키마. 예제와 필드 검사는 `experiments/`, `llmd/config.py`에 있다. |
| `model` | 모델 YAML 또는 객체. `name`, 40자리 `revision`, 선택 항목 `tokenizer`, `tokenizer_revision`, `path`, `tokenizer_path`. 모델과 별도 tokenizer를 쓰면 양쪽 revision을 지정한다. |
| `profile` | 공식 `llm-d.ai/v1 EndpointPickerConfig` YAML 경로. 플러그인 인스턴스, 데이터 소스, scorer 가중치와 profile 조합은 native YAML이 정한다. |
| `endpoints[]` | 항목 하나가 discovery 대상 서비스 인스턴스 하나다. ID, 역할 목록, backend, node, host, HTTP/KV/sidecar port, 물리 장치, 자원 이름, nodeSelector 및 엔진별 설정을 갖는다. |
| `engine` | 공통 기본값과 인스턴스별 override. dtype, KV dtype/layout, block size/count, TP/PP/DP, context, token budget, 동시 실행 한도, compile options와 엔진 CLI argv를 지정한다. |
| `verify` | 모델 tokenizer로 생성할 입력 토큰 수, 강제 출력 길이, 요청 수와 동시성. 동일한 입력 토큰열·생성 조건과 독립 GPU full 출력을 비교한다. 선택 필드 `paths`는 시험에서 관측할 경로 집합이다. |
| `instrumentation` | `rbln_events` 및 선택적 엔진 `otlp_endpoint`. `run.json`에 metrics, 설정, argv, 디바이스 UUID, 버전, 바이너리 해시, 로그와 이벤트 경로가 남는다. |
| `cache_root` | 엔진별 compile-cache 루트. 환경 변수 `HETERO_COMPILE_CACHE` 또는 기본 `~/.cache/hetero-pdd/compiled`를 사용한다. |

역할 수는 endpoint 수로, 인스턴스의 물리 장치 수는 TP × PP × DP로 표현한다. 같은 장치의 full/prefill 또는 full/decode는 같은 엔진에 역할을 함께 부여한다. 별도 인스턴스가 동일한 node/backend/device를 중복 예약하면 검증이 실패한다. 예제 `baseline-hetero.yaml`은 GPU와 NPU full 후보를 제공하고, `pool-2p2d.yaml`은 생산자 2개와 소비자 2개를 제공한다. `cuda-pair-tp2.yaml`은 서로 다른 노드에서 동일 TP2 shard 구성을 쓰는 CUDA 생산자·소비자의 배포 예다.

| Backend / 경로 | 제공 범위와 검사 |
| --- | --- |
| CUDA full / prefill / decode | vLLM 엔진과 논리 장치 배치. PDD는 양쪽의 KV dtype/layout, block size, 모델 revision, TP가 맞아야 한다. 현재 NIXL lease 매핑은 PP1/DP1로 제한한다. |
| RBLN full | 엔진이 지원하는 모델 및 TP 배치. RBLN 어댑터의 PP/DP 배치는 구현하지 않았다. |
| CUDA → RBLN | 동일 노드에서 생산자 GPU를 볼 수 있는 UCX/NIXL 소비자. TP1, native FP16 또는 BF16, HND, dense attention을 검사한다. 각 valid producer ID를 소비자의 peer label로 게시한다. 하드웨어 검증의 dtype은 FP16이다. |
| TikTok | 위 RBLN 경로에서 한정된 Host 수신 slot 및 byte budget을 별도로 예약한다. 실행 가능 수는 실제 모델의 `max_num_seqs` 및 `active_decode_limit`로 제한한다. |
| 추가 장치 / 변환 | `config.py`의 backend capability 및 pair 검사를 추가하고 `engines.py`에 실행·환경 검증을 연결한다. shard 재분배, 타 노드 RBLN host import, RBLN producer는 현재 adapter에 구현되어 있지 않다. |

새 모델은 pinned snapshot과 모델 YAML을 추가하고 동일한 GPU 기준 실행부터 검증한다. 저장소의 모델 두 개가 새로운 아키텍처의 정확성을 보증하지 않는다. 엔진의 모델 등록, 실제 컴파일, KV 기하와 출력 비교가 모델별로 필요하다. 캐시 키는 모델·tokenizer revision, config, 엔진 옵션, 장치 UUID/driver, 패키지와 로컬 RBLN 소스 해시를 포함한다. 같은 키의 빌드는 잠그고, 모델 서빙 준비가 확인된 세대만 원자적으로 게시한다. 일시적인 offload 파일은 실행 디렉터리에 둔다.

연구 trace의 정적 배정을 재생할 때는 `experiments/trace-allocation.yaml`과 요청별 `x-pdd-request-index`를 사용한다. 온라인 비용 기반의 `selective` 실험과 구별해서 기록한다.

새 정책은 Go의 공개 factory를 등록하고 native profile에서 참조한다. 예제 `example-weighted.yaml`은 두 scorer의 가중치로 full 엔진을 선택한다. `fixed-pdd.yaml`의 `pair-handler.parameters`에 `producer`와 `consumer` ID를 지정하면 한 쌍을 고정한다. 선택 실패, 손상된 KV handle, decode 실패는 오류로 반환한다. [플러그인 추가 절차와 실험 규칙](../llmd/epp/README.md)

## Kubernetes

기본 구성은 공식 standalone chart의 Envoy/EPP, namespaced RBAC, Service와 Pod discovery를 사용한다. 노드에는 GPU/RBLN device plugin, CUDA IPC 접근이 가능한 container runtime, 모델과 컴파일 캐시용 PVC가 필요하다. `experiments/cluster-site.yaml.example`에서 사이트의 image, PVC, runtimeClass, CPU/메모리 요청과 peer GPU visibility를 지정한다. GPU→RBLN은 producer와 consumer를 같은 노드 및 IPC namespace에 배치한다. 전용 노드 예제의 `all`을 사용하면 같은 노드 GPU 전체가 수신용 컨테이너에 보인다. 사이트에서 범위를 제한할 때는 실제 생산자 UUID를 지정한다. 하드웨어 resource 이름은 endpoint별로 바꿀 수 있다. [RBLN 0.11.1 device plugin](https://docs.rbln.ai/v0.11.1/software/system_management/kubernetes/device_plugin.html)

```bash
docker build --build-context wheels=/opt/wheels -f docker/Dockerfile -t hetero-pdd-engine:local .
docker load -i "$LLMD_BIN_DIR/epp.tar"
docker load -i "$LLMD_BIN_DIR/pd-sidecar.tar"
export ns=hetero-pdd
# 기존 namespace에 이 배포의 Pod가 읽을 pdd-models, pdd-compiled PVC를 준비한다.
work=$(mktemp -d)
"$CUDA_PY" -m llmd render experiments/qwen2-fixed.yaml --target kubernetes \
  --namespace "$ns" --site experiments/cluster-site.yaml.example --work-dir "$work"
"$CUDA_PY" -m llmd.kube start --work-dir "$work" --timeout 2400
"$CUDA_PY" -m llmd.kube ready --work-dir "$work"
"$CUDA_PY" -m llmd.kube verify --work-dir "$work" --reference "$run/baseline/verification.json"
"$CUDA_PY" -m llmd.kube stop --work-dir "$work"
```

노드별 container runtime에 이미지를 배포하거나 사이트 설정에 registry digest를 입력한다. 모델 PVC의 루트는 HF hub 캐시 구조이고, 여러 호스트의 동시 사용에는 적합한 access mode가 필요하다. 인스턴스당 1 replica를 생성한다. Kubernetes의 wire engine ID는 실제 Pod 이름이다. 생산자 Pod를 교체하면 새 엔진 ID가 붙는다. 같은 Pod에서 NIXL 프로세스가 재시작되면 이전 CUDA IPC 매핑을 재사용하지 않도록 시작을 거부한다. 이 경우 해당 Pod를 교체하면 Deployment가 새 이름의 Pod를 생성한다. Kubelet이 배정한 장치와 Pod UID, node, 실제 container imageID를 `kube-run.json`에 기록한다. 검증 중에는 kubectl port-forward를 통해 해당 배포의 Gateway와 metrics에 접근하고 종료 시 포워드를 닫는다. 엔진 이미지 빌드에는 저장소가 고정한 compiler wheel이 필요하다. [엔진 이미지 정의](../docker/Dockerfile)

## 검증 결과 해석

`verification.json`에는 `smoke`, `transfer`, `correctness`가 각각 기록된다. GPU full 실행에서 reference를 만들면 `smoke=passed`, `correctness=not_evaluated`이다. 실행마다 모델 가중치와 tokenizer 설정의 SHA-256을 기록한다. PDD 요청에서는 EPP decision과 gateway header, producer/request ID 및 NIXL byte/count 증가를 검사한다. RBLN 이벤트를 활성화한 경우에는 소비자 worker에서 같은 lease의 remote/local 블록 수, H2D 블록 수 및 비어 있지 않은 import를 입력 범위와 대조한다. 요청별 import 계측이 없으면 합산 전송만 `transfer=aggregate_observed`로 기록한다. `correctness=passed`는 독립 GPU full 실행과 각 요청의 전체 출력 token ID 열이 일치했음을 뜻한다. 모델의 일반적 품질이나 처리량 지표는 별도 실험이 필요하다.

프로토콜 fixture와 실제 장비 실험은 독립적으로 실행할 수 있다.

```bash
"$CUDA_PY" -m llmd.serving.protocol_test --work "$run/protocol"
"$CUDA_PY" -m llmd.epp.tests.policy_process --work "$run/policies"
"$CUDA_PY" -m unittest llmd.test_engines llmd.test_runtime
```

실측 기록과 남은 검증 범위는 [검증 기록](llmd-validation.md)에 정리했다.
