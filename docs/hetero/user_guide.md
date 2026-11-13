# 사용자 안내서

문서 갱신: 2026-11-12. 아래 시험·자료 확인일은 2026-11-10이다.

현재 CLI의 실행 절차를 정리했다.
실행 전 `source "$HETERO_ENV_FILE"`로 운영자가 준비한 환경 파일을 읽는다.
모든 결과·캐시·모델 경로는 저장소 밖 변수로 받으며 아래 명령은 실제 장치를 사용한다.
장치 점유 확인과 clean 작업 트리 확보 후 실행한다.

## 단일 서비스 실행

run.py의 --run-dir는 서비스 실행 전체의 루트다. service/는 하네스가 직접 만든다.
--workloads에는 객체 목록 JSON을 전달하며 이름·필수 정수·허용 키를 검사한다.

```json
[{"name":"s1","input_tokens":512,"output_tokens":128,"requests":176,
  "seed":20260924,"warmup":1,"concurrency":2}]
```

위 예는 PDD다. B0는 concurrency 대신 slots를 사용한다(예: {"gpu":1,"npu0":1,"npu1":1}).

```bash
SERVICE_ARGS=(--model-path "$MODEL_PATH" --model-name qwen3-0.6b   --dtype float16 --max-model-len 1024 --max-num-seqs 1 --num-blocks 32)
"$CUDA_PY" -m tools.hetero.profiler.run --configuration pdd --mode monitor   --run-dir "$RUN_DIR" --workloads "$WORKLOADS_JSON" "${SERVICE_ARGS[@]}"
```

| 모드 | 수집·해석 |
|---|---|
| none | 클라이언트 원자료; 서버 자원 수집 없음 |
| monitor | all-smi, 서버 metrics 시계열·경계; 단계 귀속은 추정 |
| trace | monitor + OTEL span + RBLN 이벤트, 요청 ID 연결 |
| deep | trace + 온라인 torch 프로파일; NPU 연산자 분석은 아래 오프라인 경로 사용 |

시작한 프로세스는 finally에서 정리한다. 오류가 났으면 run.json의 실패 단계와 service/logs를 확인한다.
analysis_status가 INCOMPLETE이면 run_status PASS만으로 계측 성공이라 판단하지 않는다.

## S7 및 KV 레이아웃 지원 조건

GPU producer가 `VLLM_KV_CACHE_LAYOUT=HND`이면 `VLLM_RBLN_PDD_LAYOUT_REORDER=0`을 쓴다.
reorder 1은 producer가 실제로 NHD 배치일 때만 쓴다.
producer의 commands.sh 설정뿐 아니라 prefill 로그의 실제 KV cache layout도 확인한다.
S7의 HND+reorder 1은 **설정 호환성 문제 재현용**이며 정상 성능 최적화 조건이 아니다.
NHD 조건의 출력 정합성은 사용 전 별도 검증하며, 향후 핸드셰이크 상대 layout 기반 자동 선택을 제안한다.
근거: sweep4 S7 prefill의 HND·FLASH_ATTN 로그와 출력 토큰 대조.

## 시나리오 스윕과 재개

```bash
"$CUDA_PY" -m tools.hetero.profiler.sweep   --scenarios tools/hetero/scenarios/initial.json --sweep-dir "$SWEEP_DIR"   --model "qwen3-0.6b=$QWEN06_PATH" --model "qwen3-4b=$QWEN4_PATH"   --model "qwen25-3b=$QWEN25_PATH" --cache-root "$CACHE_ROOT"
```

`--only s1_b0 --only s1_pdd`처럼 반복하여 일부만 실행할 수 있다.
성공 기록은 스윕 폴더의 `scenarios.json`에 `attempts`(path·returncode)로 남는다.
재개 시 마지막 attempt의 종료코드0과 모든 workload summary의 run_status·analysis_status PASS를 함께 확인한다.
`runs/`만 다른 폴더로 복사해서는 재개되지 않는다. 기록 파일에 이전 성공 실행이 없으면 새 접미사로 실행한다.
시나리오 정의를 바꾸면 새 `--sweep-dir`가 필요하다. 제한 재측정에는 `--only`를 명시한다.
같은 pair의 요청 수·입출력·seed, equal_concurrency의 슬롯 합계/동시성이 같아야 한다.
성공한 시나리오는 재실행에서 건너뛰며 실패한 실행은 새 접미사 폴더에 남긴다.
모델 키에 연결한 실제 경로는 실행 인자로만 제공한다. 캐시 호환성은 [운영 안내](operator_guide.md)를 참조한다.
S6 pilot은 자동 수행하지 않는다. 외부 임시 JSON에서 16요청으로 두 구성을 실행한 뒤,
빠른 구성의 makespan으로 ceil(16×120/makespan), 상한 512를 계산해 시나리오를 확정한다.
본측정 시간은 각 실행의 summary에서 확인한다. 추정 요청 수만으로 120초 충족을 보장하지 않는다.

## PER-001 모니터링 부하

```bash
"$CUDA_PY" -m tools.hetero.profiler.overhead --configuration pdd   --run-dir "$PER_RUN_DIR" --workload "$ONE_WORKLOAD_JSON" "${SERVICE_ARGS[@]}"
```

서비스 인자는 측정 옵션 뒤에 잇는다. workload는 객체 하나 또는 요소 하나인 목록이다.
S2 조건 시험에서는 Qwen2.5-3B BF16, ctx 4352, batch 4, blocks 160,
i4096/o256·16요청·동시성 8을 사용했다. 위 S1 SERVICE_ARGS를 그대로 쓰지 않는다.
서비스 none 모드로 한 번 시작하고 OFF/ON 워밍업 뒤 OFF·ON·ON·OFF를 4블록 실행한다.
판정 경계이면 2블록을 한 번만 추가한다. 유효성 실패 블록은 1회 재실행하고 재실패 시 ABORTED다.
overhead/overhead.json에 원자료·재시도·CPU·판정을, overhead.md에 핵심 결과를 저장한다.
PASS는 5% 규칙의 통과이며 trace 모드 부하까지 보증하지 않는다.

## 오프라인 연산자 프로파일

온라인 서버를 내리고 장치별 venv로 각각 실행한다. --out은 저장소 밖 빈 폴더여야 한다.
모델은 예시 Qwen2.5-3B이며 실제 경로는 MODEL_PATH로 전달한다.

```bash
for phase in prefill decode; do
  "$CUDA_PY" -m tools.hetero.profiler.opprofile --device gpu --phase "$phase"     --model-path "$MODEL_PATH" --model-name qwen25-3b --out "$OP_OUT/gpu_$phase"
  "$RBLN_PY" -m tools.hetero.profiler.opprofile --device npu --phase "$phase"     --model-path "$MODEL_PATH" --model-name qwen25-3b --out "$OP_OUT/npu_$phase"
done
"$PDD_TOOLS_PY" -m tools.hetero.profiler.operator_stats   --input "$OP_OUT/gpu_prefill" --out "$OP_OUT/gpu_prefill"
```

prefill은 i4096/o1, decode는 i32/o256, 각각 4요청이고 최대 길이에 맞춰 제한한다.
`--decode-tokens 64`로 짧게 재측정할 수 있으나 H6b의 64·256토큰 NPU decode는 모두 무결성 문제였다.
NPU PB는 NEC·Neural DMA 트랙을 사용하며 현재 prefill 집계는 마지막 정상 모듈의 부분 결과다.
전체 trace와 실행별 PB를 중복 합산하지 않는다. 오류 slice를 버린 비율표를 정상 결과로 쓰지 않는다.
GPU kernel/cpu_op, NPU 명령은 분할 단위가 달라 비율을 직접 성능비로 해석하지 않는다.

## 분석·결과표·그림

```bash
"$CUDA_PY" -m tools.hetero.profiler.analysis --run-dir "$RUN_DIR/$WORKLOAD"
"$CUDA_PY" tools/hetero/finalize_results.py --sweep-dir "$SWEEP_DIR" --opprofile-dir "$OP_OUT"
"$PDD_TOOLS_PY" tools/hetero/report_figures.py --sweep-dir "$SWEEP_DIR"   --opprofile-dir "$OP_OUT" --overhead "$PER_RUN_DIR/overhead/overhead.json" --out "$SWEEP_DIR/figures"
```

| 산출물 | 보는 방법 |
|---|---|
| run.json / service/logs | 기동·측정·정리 단계, 실패 원인 |
| workload/profile/summary.json | run_status·analysis_status를 먼저 확인하고 각 value/status/reason 확인 |
| results.csv / comparison.csv | 시나리오별 수치와 유효한 쌍의 B0 대비 차이; 실패 행도 보존 |
| dataset/*.csv / DATA_DICTIONARY.md | 요청·단계 자원·시계열·전송·대역폭·에너지·연산자와 칼럼 정의 |
| operators.csv / operators_by_type.csv / operators.md | 장치·phase·category·scope를 확인한 뒤 연산/유형별 비율 사용 |
| figures/*.png | 최종 보고서용 그림; estimated와 자료 출처를 캡션과 함께 읽음 |

not_collected를 0으로 치환하지 않는다. T03·R09b, NPU T06의 하한 추정,
T02의 조건, 1토큰 요청 TPOT의 적용 여부는 [지표 정의](metrics_definition.md)를 따른다.
보고서·데이터셋만 전달할 때 원본 trace·모델·환경 파일은 포함하지 않는다.
