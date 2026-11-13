# 프로그램 목록

문서 갱신: 2026-11-11. 아래 시험·자료 확인일은 2026-11-10이다.

아래 목록은 이 브랜치의 코드 기준이다.
줄 수는 이번 문서 작성 시 `wc -l`로 직접 측정했으며 빈 줄·주석을 포함한다.
엔진은 수정분이 아닌 파일 전체 줄 수다. CLI가 없는 모듈에 가상의 실행 명령을 부여하지 않는다.
모든 명령은 저장소 루트 기준이다. 결과·모델·venv 경로는 외부 위치로 설정한다.

## 실행 환경 기호

| 기호 | 환경·용도 |
|---|---|
| CUDA_PY | CUDA venv Python; 온라인 하네스·프로파일러·분석; transformers, prometheus_client 필요 |
| RBLN_PY | RBLN venv Python; NPU 서버·오프라인 NPU LLM·조건부 import 시험 |
| PDD_TOOLS_PY | 도구 venv Python; Perfetto 패키지 및 도구 관련 작업 |
| PERFETTO_TP_BIN | 고정한 trace_processor_shell; operator_stats의 PB SQL 파싱 |
| python3 | 시스템 Python; unittest와 bash 설치 스크립트의 준비 작업 |
| ALLSMI_BIN / OTELCOL_BIN | 별도 고정 바이너리; Python venv로 실행하지 않음 |

환경변수에는 사용자가 설치한 실행 파일을 지정한다. 모델·결과 경로 예시는
MODEL_PATH, RUN_DIR, SWEEP_DIR, OP_OUT 같은 실행 시 변수로 표현한다.
현재 전체 시험에는 prometheus_client import가 필요하므로 표준 라이브러리만으로
시험 환경 전체가 완성된다고 해석하지 않는다.

## 하네스와 실행 프로그램

아래 경로는 `tools/hetero/` 기준이다. 명령 번호는 뒤의 예제와 연결된다.

| 파일 | 줄 수 | 목적 | 실행 환경·방법 |
|---|---:|---|---|
| orchestrate.py | 690 | B0/PDD 서버 기동·정지, 역할 argv/env, clean·버전 검사 | CUDA_PY; start/stop (A); 자식은 CUDA_PY/RBLN_PY |
| loadgen.py | 389 | 토큰 입력·SSE 요청, 중앙 FIFO, 원자료·응답 헤더 저장 | CUDA_PY; (B), run에서 호출 |
| finalize_results.py | 371 | 결과표·비교표·데이터셋·DATA_DICTIONARY | CUDA_PY; (F), sweep에서 호출 |
| profiler/run.py | 386 | 온라인 모드·서비스·workload 생명주기 | CUDA_PY; (C) |
| profiler/overhead.py | 296 | PER-001 OFF/ON 반복·통계·CPU·보고서 | CUDA_PY; (D) |
| profiler/sweep.py | 192 | 쌍 검사·모델 매핑·캐시·재개·finalize | CUDA_PY; (E) |
| profiler/analysis.py | 340 | 원자료·계측 검증 및 client/server/resource/transfer/energy 계산 | CUDA_PY; (G); --run-dir는 workload 폴더 |
| profiler/scrape.py | 70 | 1Hz endpoint 병렬 GET, 원문 JSONL·경계 함수 | CUDA_PY; (H), monitors에서 실행 |
| profiler/opprofile.py | 165 | 단일 장치 오프라인 LLM 수집, meta.json | GPU=CUDA_PY, NPU=RBLN_PY; (I) |
| profiler/operator_stats.py | 268 | Chrome/PB 무결성·카테고리별 top20·유형 비율·decode kernel step | PDD_TOOLS_PY + PERFETTO_TP_BIN; (J) |

## 내부 분석·수집 모듈

경로는 `tools/hetero/profiler/` 기준이다. 이 표의 Python 파일은 별도 CLI가 없고
호출 프로그램의 Python 환경에서 import한다.

| 파일 | 줄 수 | 목적·호출자 |
|---|---:|---|
| __init__.py | 1 | 패키지 표식 |
| common.py | 63 | JSONL 읽기, 지표 객체, 합집합·구간 분할·시간가중 적분; 분석 모듈 공용 |
| expected.py | 82 | 성공 판정 재사용, 서버·엔진 ID, 역할별 기대 집합, span 대기 |
| server.py | 94 | Prometheus 파싱, 경계 차분, span 지연, 게이지 평균; analysis 호출 |
| resources.py | 249 | 장치 UUID/PCI 대응, clock 변환, 상태, M14a/b, 자원·에너지; analysis 호출 |
| transfer.py | 184 | KV·H2D·PCIe·메모리 대역폭 추정과 H4/H5 게이트; analysis 호출 |
| monitors.py | 111 | Collector/all-smi/scrape 시작·준비·SIGTERM/kill/wait; run·overhead 호출 |
| tools.py | 52 | 모드별 바이너리 버전 검사, GPU UUID/PCI와 rbln-smi 원문 수집 |
| log.py | 12 | UTC·run_id·component 로그; 실행기·분석 호출 |
| otelcol.yaml | 23 | OTLP HTTP loopback 수신 → batch → file; Collector 설정 |

COLUMN_DOCS는 resources.py와 finalize_results.py에 있다. dataset/DATA_DICTIONARY.md 생성에
사용한다. 오프라인 operators.csv 추가 병합은 finalize의 --opprofile-dir 경로도 함께 참조한다.

## 고정 도구·설치·시나리오

| 저장소 상대 경로 | 줄 수 | 목적 | 실행 환경·방법 |
|---|---:|---|---|
| tools/hetero/third_party/make_offline_bundle.sh | 67 | 고정 산출물·wheel 다운로드, 해시·라이선스 묶음 | 인터넷 가능한 bash/python3; (K) |
| tools/hetero/third_party/install_offline.sh | 49 | SHA256 검증, 외부 venv·바이너리 설치 | bash/python3; (L) |
| tools/hetero/third_party/versions.env | 117 | 버전·릴리스 날짜·이유·산출물 SHA256 | 두 스크립트가 source; 직접 실행 안 함 |
| tools/hetero/scenarios/initial.json | 1527 | 모델 키·쌍·서비스·workload 매트릭스 | sweep의 --scenarios 입력; S9는 제거됨 |

번들·venv·설치 바이너리는 저장소 밖에 둔다. Python 대상 버전은 번들과 설치에서 일치해야 한다.
실제 설치 여부와 경로는 사용자의 환경에 따르며 문서에 고정하지 않는다.

## 엔진 변경 파일

| 저장소 상대 경로 | 전체 줄 수 | 계측 역할 | 실행 환경 |
|---|---:|---|---|
| vllm_rbln/v1/worker/metrics_v2.py | 562 | 게이트·이벤트 JSONL·flush·SDK report·step·thread-local 복사 합 | RBLN_PY의 vLLM worker |
| vllm_rbln/v1/worker/rbln_model_runner.py | 3109 | step 선택 필드, 계측 복사 함수와 원본 경로 분리 | 같은 worker; 직접 CLI 없음 |
| vllm_rbln/distributed/kv_transfer/kv_connector/v1/rbln_nixl/worker.py | 738 | kv_layout 및 xfer start/done·H2D 이벤트 | 같은 worker; 직접 CLI 없음 |

## 시험 기반

경로는 `tests/hetero/` 기준이다. 모두 시스템 unittest에서 사용하며
실제 장치 의미 검증을 대신하지 않는다.

| 파일 | 줄 수 | 용도·실행 |
|---|---:|---|
| __init__.py | 1 | 패키지 표식 |
| mock_protocol.py | 123 | mock의 SSE·metrics 가정과 확인/TODO 모음; import |
| mock_server.py | 160 | ThreadingHTTPServer, 임의 포트·SSE·지연·500·수집 원문; context manager |
| mock_proxy.py | 56 | prefill 뒤 decoder 선택, ID·target 헤더; context manager |
| fakes/__init__.py | 93 | 임시 PATH용 fake all-smi/otelcol 생성, FakePopen; install_fake_tools 호출 |

fake all-smi는 `api --port P --interval 1`, fake otelcol은 `--config FILE`을 받는다.
직접 제공되는 영구 fake 바이너리 파일은 없다. 테스트가 임시 디렉터리에 만든다.
fixtures/real은 출처·치환 규칙을 README에 두고 synthetic은 손계산 기대값을 동봉한다.

## 실행 예제

환경변수는 사전에 설정한다. SERVICE_ARGS는 모델과 서비스 설정을 담은 bash 배열이다.
예제는 실행 형식이며 이번 문서 작성에서 실제 서버를 기동하지 않았다.

```bash
SERVICE_ARGS=(--model-path "$MODEL_PATH" --model-name "$MODEL_NAME"
              --dtype bfloat16 --kv-cache-dtype bfloat16)
# A: 직접 하네스 실행
"$CUDA_PY" tools/hetero/orchestrate.py start --configuration pdd --run-dir "$SERVICE_DIR" "${SERVICE_ARGS[@]}"
"$CUDA_PY" tools/hetero/orchestrate.py stop --run-dir "$SERVICE_DIR"
# B: 직접 부하; B0는 --endpoint 대신 --worker ROLE=URL=SLOTS를 반복
"$CUDA_PY" tools/hetero/loadgen.py --endpoint "$ENDPOINT" --concurrency 2 --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" --workload sample --input-tokens 512 --output-tokens 16 --requests 4 --out-dir "$WORKLOAD_OUT"
# C: 온라인 실행기; WORKLOADS는 JSON 배열 파일
"$CUDA_PY" tools/hetero/profiler/run.py --configuration pdd --mode monitor --run-dir "$RUN_DIR" --workloads "$WORKLOADS" "${SERVICE_ARGS[@]}"
# D: WORKLOAD는 단일 객체 또는 요소 한 개인 목록 파일
"$CUDA_PY" tools/hetero/profiler/overhead.py --configuration pdd --run-dir "$OVERHEAD_DIR" --workload "$WORKLOAD" "${SERVICE_ARGS[@]}"
# E: --only는 반복 가능; 모델 키별 --model을 추가
"$CUDA_PY" tools/hetero/profiler/sweep.py --scenarios tools/hetero/scenarios/initial.json --sweep-dir "$SWEEP_DIR" --model "qwen25-3b=$MODEL_PATH" --only s2_b0 --only s2_pdd
# F: 오프라인 연산자 결과 병합은 선택
"$CUDA_PY" tools/hetero/finalize_results.py --sweep-dir "$SWEEP_DIR" --opprofile-dir "$OP_OUT"
# G: 서비스 루트가 아닌 그 아래 workload를 지정
"$CUDA_PY" tools/hetero/profiler/analysis.py --run-dir "$RUN_DIR/$WORKLOAD_NAME"
# H: endpoint NAME=URL을 반복
"$CUDA_PY" -m tools.hetero.profiler.scrape --endpoint "gpu=$M프로파일러CS_URL" --out "$M프로파일러CS_OUT"
# I: NPU는 RBLN_PY와 --device npu; phase는 prefill 또는 decode
"$CUDA_PY" tools/hetero/profiler/opprofile.py --device gpu --model-path "$MODEL_PATH" --model-name "$MODEL_NAME" --phase decode --out "$OP_OUT/gpu_decode"
# J: NPU PB에는 PERFETTO_TP_BIN 필요
"$PDD_TOOLS_PY" tools/hetero/profiler/operator_stats.py --input "$OP_OUT/gpu_decode" --out "$OP_OUT/gpu_decode/stats"
# K/L: 모든 설치·산출 위치는 저장소 밖
bash tools/hetero/third_party/make_offline_bundle.sh --out "$BUNDLE" --python-version "$PYTHON_VERSION"
bash tools/hetero/third_party/install_offline.sh --bundle "$BUNDLE" --venv "$TOOLS_VENV" --tools-dir "$TOOLS_DIR"
```

H6b에서 NPU PB 생성과 NEC·Neural DMA 트랙을 확인했다. prefill은 마지막 정상 모듈의 부분 집계이며,
decode 64/256토큰은 trace 무결성 문제로 not_collected다. GPU 두 phase는 집계했다.
전체 prefill 대표성과 NPU decode 정상 집계는 미확인(H6b)이다.

줄 수 재계산 명령:

```bash
wc -l tools/hetero/*.py tools/hetero/profiler/*.py tools/hetero/profiler/*.yaml tools/hetero/third_party/* tools/hetero/scenarios/initial.json
wc -l tests/hetero/*.py tests/hetero/fakes/*.py
wc -l vllm_rbln/v1/worker/metrics_v2.py vllm_rbln/v1/worker/rbln_model_runner.py vllm_rbln/distributed/kv_transfer/kv_connector/v1/rbln_nixl/worker.py
```

## 추가된 그림·CI·배포 프로그램

| 저장소 상대 경로 | 줄 수 | 목적 | 실행 환경·방법 |
|---|---:|---|---|
| tools/hetero/report_figures.py | 301 | 결과표·단계·전송·에너지·연산 유형·PER-001 그림 | PDD_TOOLS_PY; 아래 명령 |
| .github/workflows/hetero_unit_tests.yml | 34 | 수동/대상 push, 단위시험·변경 파일 pre-commit | GitHub ubuntu-22.04/Python3.10 |
| .github/actionlint.yaml | 13 | self-hosted 라벨 정적 검사 설정 | pre-commit/actionlint |

```bash
"$PDD_TOOLS_PY" tools/hetero/report_figures.py --sweep-dir "$SWEEP_DIR"   --opprofile-dir "$OP_OUT" --overhead "$PER_RUN_DIR/overhead/overhead.json" --out "$FIGURE_OUT"
```

함수・CLI 인자의 전체 목록은 [함수 참조](function_reference.md), 실제 실행은 [사용 안내](user_guide.md)를 따른다.
