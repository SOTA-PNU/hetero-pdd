# 지표 정의서 초안

기준일: 2026-11-09. 대상: B0(GPU·NPU0·NPU1 독립 추론)와 PDD 1P:2D.
PLAN v6.2의 지표 번호를 유지하되, 이후 승인된 수정과 현재 구현을 구분한다.
M13은 PLAN에 정의되지 않았다. 없는 지표 번호를 새로 만들지 않는다.

## 근거와 읽는 순서

| 근거 | 적용 범위 |
|---|---|
| [analysis.py](../../tools/hetero/profiler/analysis.py) | 원자료 검증, 클라이언트 계산, 요청 연결 |
| [server.py](../../tools/hetero/profiler/server.py), [resources.py](../../tools/hetero/profiler/resources.py) | 경계 차분, 구간·상태·자원·에너지, M14a COLUMN_DOCS |
| [transfer.py](../../tools/hetero/profiler/transfer.py), [common.py](../../tools/hetero/profiler/common.py) | 전송·메모리 식, 정확도 객체, 시간가중 적분 |
| [finalize_results.py](../../tools/hetero/finalize_results.py)의 COLUMN_DOCS | 결과표·데이터셋 칸, 비교식, KV 논리 크기 |

과거 메모의 판정은 그 시점의 분석 코드 기준이다. 최신 코드의 정의는 아래와 같다.

## 공통 표기와 유효성

지표 객체는 `value`, `status`, `reason`, `source`, `unit`을 가진다.
시간가중 귀속에는 `coverage`가 추가된다. 값이 없으면 `value=null`이며 0으로 대체하지 않는다.
연산자 CSV의 호출 수·시간·비율은 평탄한 열이다. 모든 산출물이 이 객체 구조인 것은 아니다.

| status | 의미 | 예 |
|---|---|---|
| measured | 원자료의 관측값 또는 그 차분·집계 | TTFT, 서버 경과시간, T01 |
| estimated | 구간 귀속·바이트 접근량 등의 근사를 사용 | 1Hz 자원 평균, 에너지, T06 |
| not_collected | 적용 불가·결측·미검증·미구현으로 값 없음 | T03, 출력 1토큰의 TPOT |

`measured`는 원인이나 하드웨어 busy 시간을 입증한다는 뜻이 아니다.
`reason=not_applicable`과 `counter_reset`, `zero_count`, `missing_raw`는 서로 다른 사유다.
정합성 평가의 `not_evaluated`, 실행 판정 `run_status`·`analysis_status`는 지표 status와 별개다.
모델·실행 설정별 정확도 한계는 reason 및 이 문서의 한계 표를 함께 읽는다.

| 모드 | 현재 수집·분석 범위 | 미수집·주의 |
|---|---|---|
| none | 클라이언트 M01~M05, M14a 두 구간 근사 | analysis_status=NOT_COLLECTED; server/transfer/resource/energy는 빈 객체 |
| monitor | 클라이언트, 서버·NIXL 경계, 게이지, 1Hz 자원·에너지, M14a/b 근사 | 이벤트·요청별 서버 span 없음; T03·T06·R09에 필요한 장치 시간 없음 |
| trace | monitor + span·RBLN 이벤트, 상세 M14a/b, 조건부 T06 | H4/H5 게이트는 그대로 적용; 상세 표의 현재 한계 참조 |
| deep | trace + 온라인 torch 프로파일 수집·완성 대기 | 지연·에너지는 분석 전용; NPU 장치 연산 확보는 미확인(H6) |
| 오프라인 WP9 | GPU/NPU 각각 prefill 또는 decode 입력의 연산자 수집·집계 | 온라인 네 모드와 별도; NPU 트랙·시간축 미확인(H6b) |

본측정 구간 W=[첫 scheduled_arrival, 마지막 completed], 길이를 D초로 둔다.
warm-up은 별도 파일·ID로 제외한다. 성공은 loadgen의 `request_succeeded()`를 공유한다:
오류 없음, HTTP 2xx, 실제 출력 토큰 수=요청 출력 토큰 수.
`validate_raw()`는 파일 존재, 예정 수=고유 ID 수=성공 수=원자료 행 수,
각 요청의 실제 입력·출력 토큰 수도 추가 검사한다. 위반하면 `run_status=FAILED`다.

## 클라이언트 M01~M05

A=scheduled_arrival, Q=assigned, S=actual_send, F=first_output,
L=last_output, C=completed, N=실제 출력 토큰 수. 모두 MONOTONIC 초다.
출처는 본측정 `requests.jsonl`; 단위 s인 지연은 모든 온라인 모드에서 계산한다.
deep 결과는 분석 전용이며 일반 성능 비교의 기준으로 쓰지 않는다.

| 번호 / 출력 키 | 계산식 | 단위 | 가용성·주의 |
|---|---|---|---|
| 큐 / queue_wait_seconds | Q−A | s | 클라이언트 배정 대기; 서버 큐 M06과 다름 |
| client_send_delay_seconds | S−A | s | 배정 이후 송신 지연도 포함 |
| M01 ttft_seconds | F−A | s | 클라이언트 큐 포함; 옛 loadgen 정의 |
| M01b service_ttft_seconds | F−S | s | 신규 service 기준 TTFT |
| M02 tpot_seconds | (L−F)/(N−1) | s (토큰당 평균) | N=1이면 not_collected/not_applicable |
| M02 second_token_delay_seconds | 두 번째 토큰 시각−첫 토큰 시각 | s | N=1이면 not_applicable; 토큰 시각 불충분이면 미수집 |
| M02 token_itl_seconds | 인접 토큰 도착 시각의 차이 목록 | s | 토큰 ID·토큰 시각 수가 출력 수와 일치하고 시각도 서로 달라야 함 |
| 보조 chunk_itl_seconds | 내용 또는 ID가 있는 인접 청크의 시각 차이 | s | token ITL과 별개; 묶음 전송을 토큰 간격으로 만들지 않음 |
| M03 e2e_seconds | C−A | s | 큐 포함 완료 지연 |
| service_seconds | C−S | s | M14a 분모 |
| makespan_seconds | max(C)−min(A) | s | 실패 요청도 구간 경계 계산에는 포함 |
| M04 output_tokens_per_second | 성공 요청 출력 토큰 합 / D | tokens/s | 본측정 전체 처리량 |
| M04 completed_requests_per_second | 성공 요청 수 / D | requests/s | 완료 요청 처리량 |
| M05 tok_per_s_per_device | M04 출력 처리량 / 3 | tokens/s/device | 두 구성 모두 물리 장치 3대 기준; 개별 장치의 실측 속도와 다름 |

분포는 평균·p50·p90·p95·p99다. 정렬된 n개 값에서 p백분위는
`x[ceil(p*n/100)-1]`(nearest-rank)이다. token/chunk ITL은 간격 목록을 펼쳐 집계한다.
옛 계산과의 재현을 위해 TTFT·TPOT 요약에서 0을 제외하며,
큐 요약은 실패 요청도 포함하고 그 밖의 지연 요약은 성공 요청을 사용한다.
실제 fixture 비교 허용오차는 상대오차 1e-9다.

`client.summary`에는 요청·성공·실패·timeout·불완전 수, 입출력 토큰 합,
target_request_counts, token_itl_request_count 등도 있다.
`retry_count=0`은 현재 loadgen의 재시도 없음에 대응한다.
`per_device`는 B0 assigned_target 또는 PDD의 실제 decoder 역할별 부분집합을 재계산한다.
이 부분집합의 makespan은 전체 W와 다를 수 있다. PDD prefill은 모든 성공 요청을 포함한다.
PLAN의 선택적 SLO goodput은 현재 CLI·계산에 구현되어 있지 않다.

## 서버 단계 M06~M14

| 번호 / 출력 위치 | 계산식·출처 | 단위 | monitor | trace·deep / 현재 한계 |
|---|---|---|---|---|
| M06 server.role.queue | gen_ai.latency.time_in_queue 또는 request_queue_time_seconds 차분 평균 | s | 집계 measured | span 요청별·평균 measured |
| M07 server.role.prefill | gen_ai.latency.time_in_model_prefill 또는 request_prefill_time_seconds | s | 집계 measured | span 요청별·평균 measured |
| M08 transfer.role.kv_transfer_seconds | nixl_xfer_time_seconds의 Δsum/Δcount | s | 집계 measured | 동일 집계 + requests.ID.kv=(done−start)/1e9 |
| M09a requests.ID.h2d | 모든 복사 그룹 h2d_ns 합/1e9 | s | 미수집 | measured, measurement=host_call; DMA 완료 시간 미확인(H4) |
| M09b requests.ID.reorder | 모든 그룹 reorder_ns 합/1e9 | s | 미수집 | measured; permute·unsqueeze·contiguous 호출 구간 |
| M10 server.role.decode | gen_ai.latency.time_in_model_decode 또는 request_decode_time_seconds | s | 집계 measured | span 요청별·평균 measured |
| M10d transfer.role.device_seconds | 유효 step.device(μs)의 평균/1e6 | s | 미수집 | 전 step 값 필요; 실측은 device=null로 미수집(H4) |
| M11 operators | 연산 이름·카테고리별 호출 수, Σdur, Σdur/호출 수, Σdur/카테고리 전체 합 | count, ms, μs, ratio | 없음 | 현재 WP9 오프라인 출력; 온라인 deep만으로 확보되지 않음(H6b) |
| M12 other_seconds | 계획식: service_ttft−(P e2e+KV+H2D+reorder+첫 decode step) | s | 미수집 | 현재 전 모드 not_collected, reason=미구현(후속) |
| M14a request_phase_share | 배타적으로 배정한 단계 시간 / (C−S) | ratio | 첫 토큰/이후 두 구간 estimated | 상세 구간 measured; 요청별 비율의 산술평균 |
| M14b device_phase_active_ratio | 해당 장치·단계 구간의 W 내 합집합 길이 / D | ratio | 클라이언트 구간 estimated | span·이벤트 구간 measured |

M06·M07·M10의 Prometheus 이름에는 `vllm:` 접두사가 붙는다.
trace 지연 속성은 H0에서 확인했다. span은 경과시간이며 커널 busy 시간이 아니다.
M08·M09는 PDD 전용이다. B0 또는 해당하지 않는 역할에는 적용 불가를 표시한다.

M14a trace는 각 구간을 [S,C]로 잘라 다음 우선순위로 한 시점에 한 단계만 배정한다.
`decode > layout > kv_transfer > prefill > 서버 큐`; 큐 동률은 D 큐를 P 큐보다 먼저 둔다.
B0 단계는 queue/prefill/decode, PDD는 p_queue/p_prefill/kv_transfer/layout/d_queue/decode다.
layout은 H2D+reorder 전체 구간이다. 기타는 서비스 구간에서 모든 단계 합집합을 뺀 나머지다.
따라서 정상 양의 서비스 구간에서 기타≥0이고 비율 합=1이다. 필수 구간이 없으면 미수집이다.
M06~M10의 원래 겹치는 경과시간은 그대로 보존한다. M12와 M14a의 기타를 혼동하지 않는다.
M14b는 단계별 합집합이므로 서로 다른 단계 비율의 합이 1을 넘을 수 있다.

## 전송·메모리 T01~T07

GB/s는 십진 10^9 bytes/s다. KV 논리 크기의 MiB는 2^20 bytes다.

| 번호 / 키 | 계산식·출처 | 단위 | 가용성·status |
|---|---|---|---|
| T01 nixl_gbps | Δnixl_bytes_transferred_sum / Δnixl_xfer_time_seconds_sum / 1e9 | GB/s | PDD monitor 이상 measured; GPU prefill 표시는 두 decoder 합계의 비 |
| T02 bytes_per_transfer | 전송 단위가 확인되면 Δbytes_sum/Δcount | bytes (전송당) | 현재 not_collected: 전송 count 단위 미확인(H5) |
| T02 보조 bytes_per_request_est | kv_h2d.bytes_est=num_blocks×block_bytes | bytes | PDD trace 이상 estimated; KV 그룹·물리 블록 해석 미확인(H4) |
| T03 h2d_gbps_est | Σbytes_est / (Σh2d_ns/1e9) / 1e9 | GB/s | COPY_SYNC_CONFIRMED=False여서 미수집(H4); 확인 후에도 바이트는 추정 |
| T04 link_theoretical | 레인당 GB/s × link width | GB/s | monitor 이상; links.json 또는 NPU 입력값; 코드 status=measured, 이론 규격임 |
| T05 pcie_efficiency | GPU=T01/T04, NPU=T03/T04 | ratio | estimated; NPU는 T03 부재로 미수집(H4) |
| T06 decode_mem_bw_gbps_est | (step 수×가중치 파일 bytes + Σ문맥 길이 합×KV bytes/token) / Σstep 시간 / 1e9 | GB/s | NPU trace 이상 estimated; GPU 온라인 값은 WP9에서 산출 사유로 미수집 |
| T07 mem_bw_efficiency_est | T06 / 장치 이론 메모리 대역폭 | ratio | estimated; 이론치 미설정이면 not_collected |

T04 레인당 값은 Gen3=0.985, Gen4=1.969, Gen5=3.938 GB/s다.
GPU는 nvidia-smi의 최대 gen·width, NPU는 rbln-smi 링크 정보 또는
`NPU_PCIE_GEN`, `NPU_PCIE_WIDTH`를 사용한다. 실효 데이터 전송 속도가 아니다(미확인(H3)).
T01이 낮다는 이유만으로 PCIe 병목을 확정하지 않는다. 실제 UCX TLS/lane 선택은 미확인(H5)이다.

T06의 KV bytes/token=`층 수×2(K,V)×KV 헤드 수×head_dim×dtype bytes`.
서비스 model_path의 모델 config.json과 safetensors 파일 크기를 읽는다.
가중치 파일 크기 및 매 스텝 전체 가중치 접근 가정은 논리 근사다.
문맥은 step.context_len_sum을 우선 사용하고, 없으면
`avg_running_requests×평균(input_tokens+(output_tokens−1)/2)`로 근사하며 context_source를 남긴다.
device가 하나라도 없으면 모든 decode step 시간을 호스트 span 길이로 대체한다.
이 경우 T06·T07의 유효값은 estimated이며 reason은
`device 시간 없음: 호스트 관측 구간 기준 하한 추정`이다. device/span 혼합 분모는 쓰지 않는다.
이 하한 표기는 현재 근사 모델의 해석이며 실제 DRAM 트래픽의 수학적 하한을 실측 보증하지 않는다.
문맥·SDK device 의미는 미확인(H4)이다.
오프라인 WP9도 현재 GPU T06·T07을 `step별 KV 문맥/가중치 접근량 미확인(H6b)`으로 남긴다.

## 자원 R01~R12 및 에너지

| 번호 / 대상 | 출처·계산 | 단위 | 모드·status |
|---|---|---|---|
| R01 시스템 CPU | all_smi_cpu_utilization, 시스템 상태별 시간가중 평균·최대 | % | monitor 이상 estimated |
| R02 시스템 메모리 | all_smi_memory_used_bytes, 같은 귀속 | bytes | monitor 이상 estimated |
| R03 GPU 활용률 | all_smi_gpu_utilization, GPU 상태별 평균·최대 | % | monitor 이상 estimated |
| R04 GPU 메모리 | all_smi_gpu_memory_used_bytes | bytes | monitor 이상 estimated |
| R05 GPU 보드 전력 | all_smi_gpu_power_consumption_watts | W | monitor 이상 estimated |
| R06 NPU 활용률 | 같은 all-smi 지표를 NPU0·NPU1별 귀속 | % | monitor 이상 estimated |
| R07 NPU 메모리 | 같은 메모리 지표를 NPU별 귀속 | bytes | monitor 이상 estimated |
| R08 NPU 보드 전력 | 같은 전력 지표를 NPU별 귀속 | W | monitor 이상 estimated |
| R09a npu_step_window_ratio | step 호스트 span의 W 내 합집합 길이/D | ratio | trace 이상 estimated; 현재 코드에 유지, 장치 가동률로 쓰지 않음 |
| R09b npu_device_time_ratio | Σstep.device(μs)/1e6/D | ratio | 전 값 유효 시 코드상 measured; 실측 device 누락으로 미수집(H4) |
| R10 kv_cache_usage | vllm:kv_cache_usage_perc의 시간가중 평균 | ratio | monitor 이상 estimated |
| R10 avg_running_requests, waiting_requests | vllm:num_requests_running / num_requests_waiting 시간가중 평균 | requests | monitor 이상 estimated |
| 보조 energy_j | 본측정 내 인접 전력 표본 사다리꼴 적분 | J | monitor 이상 estimated |
| R11 energy_per_output_token_j_est | 장치 또는 세 장치 합계 에너지 / 성공 출력 토큰 합 | J/token | monitor 이상 estimated |
| R12 tok_per_s_per_w_est | 성공 출력 토큰 합 / 에너지 | tokens/s/W | monitor 이상 estimated; R11의 역수 |

R09b가 1을 넘으면 원값은 raw_value로 보존하고 value=null,
reason=device_time_missing_or_exceeds_window(H4)로 둔다. 1 이하라는 조건만으로
SDK 중첩·합산 의미까지 확인된 것은 아니다. 현재 실측으로 measured 승격을 인정하지 않는다.

자원 시간가중 평균은 유효 인접 샘플 사이의 왼쪽 값을 사용한다.
샘플 간격>1.5초 또는 어느 끝 값이라도 결측이면 연결하지 않으며, 외삽하지 않는다.
`coverage=유효하게 덮인 시간/해당 상태 구간 합집합 길이`다.
평균은 덮인 시간으로 나누고, 최대는 해당 상태에서 관측한 점도 포함한다.
장치 지표는 해당 장치 상태, CPU·메모리는 전체 장치의 합성 상태에 귀속한다.

에너지는 동일한 유효 구간에서 선형 보간한 전력을 사다리꼴 적분한다.
장치별 유효 전력 표본≥2, coverage≥0.8일 때만 값을 내며, 세 장치 모두 유효해야 합계를 낸다.
결측 시간의 에너지를 임의 보충하지 않는다. 장치별 R11·R12도 분모/분자는 workload 전체 성공 출력 토큰이다.
CPU·시스템 DRAM 전력은 합계에 포함하지 않는다. 같은 토큰·에너지를 쓰므로 R11×R12≈1이다.

## 장치 식별·메모리 종류와 D2

| 장치 | 계획에 명시된 메모리 | 실제 지표 대응 | 해석 |
|---|---|---|---|
| RTX PRO 6000 Blackwell Workstation Edition | 96GB GDDR7 | all-smi gpu_uuid ↔ nvidia-smi uuid | GPU 메모리를 GDDR6로 표기하지 않음 |
| RBLN-CA22 ×2 | 각 16GB GDDR6 | all_smi_gpu_info.pci_bus_id ↔ rbln-smi pci.bus_id | NPU 번호는 rbln-smi로 결정; gpu_index 사용 안 함 |

제품 종류·용량은 PLAN의 사양이며 실제 사용량은 R04/R07이다.
NPU_MEM_BW_GBPS(계획값 256), GPU_MEM_BW_GBPS(계획값 1792)는 이론 입력이다.
현재 NPU T07은 환경변수에서 읽으며, GPU 이론 입력을 두었다고 GPU T07이 자동 수집되지는 않는다.
H2 실측 라벨은 `gpu`, `gpu_uuid`, `gpu_index`이며 NPU info에 PCI bus가 있다.
GPU info에는 PCI 라벨이 없어 UUID로 연결한다. 센서 자체 갱신 주기·장시간 안정성은 미확인(H2)이다.

D2 합의는 분할 비율 실험을 S5의 입출력 길이비로 대체하는 것이다.
S5는 4096/256, 2176/2176, 256/4096으로 합계 길이를 4,352로 고정한다.
PDD 1P:2D라는 장치 구성은 유지된다. M14a·M14b는 그 결과로 관측한 비율이며
입출력 길이비나 장치 분할 비율 자체로 대체하지 않는다.

## 실측 한계와 결과표 해석

| 근거 | 확인된 사실 | 남은 해석 |
|---|---|---|
| H0·H4/H6/H7 메모 | H0·S8의 복사 호출 시간 증가, step.device 전부 null | copy_ 완료 동기성·SDK device 의미 미확인(H4); T03·R09b 미수집 |
| S1/S2 H5 | 두 decoder NIXL histogram count 차분이 배정 요청 수와 일치 | 일반 전송 단위임은 미확인(H5); T02 산출 보류 |
| WP7b | H0 매칭 성공, 본측정 샘플 1개로 INCOMPLETE | 요청 매칭 PASS와 계측 완전성 구분 |
| WP7b | S1 B0 coverage 약 0.713으로 에너지 미수집 | 수집 개수 검사 PASS가 에너지 coverage PASS를 뜻하지 않음 |
| M14a 수정 메모 | 단순 시간 합의 기타 −37.553436%를 배타 분할로 수정 | 음수를 자르지 않고 중첩의 원인을 제거 |
| H6_fix·WP9 코드 | 온라인 NPU 장치 trace 확보 실패, 오프라인 경로 구현 | 정상 PB·장치 트랙·시간축·상위 연산 미확인(H6b) |

results.csv의 KV 논리 MiB=`KV bytes/token×입력 토큰/2^20`이며 물리 전송량과 다르다.
comparison.csv는 검증된 같은 pair/comparison/workload의 최신 성공 결과에서
처리량 개선율 `(PDD/B0−1)×100`, makespan 감소율 `(1−PDD/B0)×100`을 낸다.
정합성 파일이 없으면 not_evaluated다. S7 outputs_identical은 완전한 request_index별
토큰 ID 대응에서만 판정하며, 성능 실행 PASS를 출력 정합성 PASS로 바꾸지 않는다.
