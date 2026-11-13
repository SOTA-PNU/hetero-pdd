# 기존 실험 원자료

원본은 로컬 `.notes/experiment_samples/`의 다음 네 workload다.
각 workload 디렉터리에 `requests.jsonl`, `summary.json`, `config.json`과
상위 run의 `service_config.json`을 함께 보관한다.

| 원본 run | 원본 workload 디렉터리 | 요청 수 |
| --- | --- | ---: |
| b0_qwen2p5_3b_bf16_b4_ctx4352 | E_short_short_c1 | 1 |
| p0_qwen2p5_3b_bf16_b1_ctx4352 | E_short_short_c1 | 1 |
| b0_qwen3_0p6b_b1_ctx1024 | workload_i512_o128_b18 | 18 |
| p0_qwen3_0p6b_b1_ctx1024 | workload_i512_o128_b18 | 18 |

절대 파일시스템 경로인 문자열 값 88개를 값 전체가 `<REDACTED_PATH>`가 되도록
치환했다. 경로에서 발견한 사용자명이 다른 문자열에도 나타나면
`<REDACTED_USER>`로 치환하는 규칙을 적용했다. HTTP URL, 상대 식별자, 시각,
수치, 토큰 ID, 생성된 텍스트와 그 밖의 값은 변경하지 않았다. 복사 시 원본과
치환본을 재귀 비교하여 이 규칙 밖의 값 변경이 없는 것을 확인했다.
시험은 이 디렉터리만 읽으며 로컬 원본의 존재에 의존하지 않는다.

`requests.jsonl`의 관측 원자료는 다음과 같다.

- 식별·배정: `request_id`, `request_index`, `workload`, `seed`, `assigned_slot`,
  `assigned_target`, `execution_path`.
- 절대 monotonic 시각: `scheduled_arrival_monotonic`, `assigned_monotonic`,
  `actual_send_monotonic`, `first_output_monotonic`, `last_output_monotonic`,
  `completed_monotonic`. Unix 시각으로 해석하지 않는다.
- 청크 관측: `chunk_timestamps`의 `t_rel_arrival_seconds`는 scheduled arrival
  기준 상대 수신 시각이다. 함께 저장된 `text_chars`, `token_ids`,
  `finish_reason`도 관측 자료다. 마지막 청크에 토큰과 `finish_reason=length`가
  함께 기록되어 있다. 완전한 SSE 원문은 보존되어 있지 않다.
- 입출력: 요청·실제 입력 및 출력 토큰 수, `streamed_token_ids`,
  `generated_text`, `http_status`, `response_headers`, `error`, `fallback`.
  실제 토큰 수는 usage 또는 옛 loadgen의 fallback으로 얻은 값이다.

옛 loadgen이 계산한 값은 `queue_wait_seconds`, `client_send_delay_seconds`,
`ttft_seconds`, `tp_ot_seconds`, `e2e_seconds`, `service_seconds`,
`second_token_delay_seconds`, `token_itl_seconds`, `chunk_itl_seconds`다.
`token_itl_available`과 `success`도 옛 loadgen의 판정 결과다.
계산된 지표와 `summary.json`은 **WP7a 시험의 기대값**으로 보존한다.
새 분석기의 원자료 입력으로 계산된 지표를 재사용하지 않는다.

이 자료의 PDD 응답 헤더에는 `x-decode-target`이 없다. 옛 proxy의 실제 출력이므로
추가하지 않았다. decoder 배정 시험용 헤더는 `synthetic/pdd_skew/`에만 있다.
새 loadgen의 `token_times_monotonic` 등의 필드로 이름을 바꾸는 작업은 WP1이다.

## PC4 all-smi 원문 발췌

`allsmi_metrics_sample.txt`는 2026-11-06에 all-smi 0.28.1을
`api --port <임의 포트> --interval 1`로 실행해 받은 실제 Prometheus 텍스트다.
GPU 1장과 RBLN-CA22 2장의 utilization·memory·power·info, CPU·메모리
지표를 발췌했다. 반복되는 HELP·TYPE 줄과 수치는 원문 그대로다.
호스트·사용자명은 `<REDACTED_HOST>`·`<REDACTED_USER>`, 장치 UUID와
시리얼 번호는 일관된 `<REDACTED_DEVICE_UUID_N>`·`<REDACTED_SERIAL_N>`으로
치환했다. 장치 인덱스와 PCI 버스 ID는 보존했다.

바인드는 `0.0.0.0`, `/-/ready`는 첫 수집 전 503, 이후 200이었다.
유휴 상태에서 29.33초 실행 후 종료했으며, 자식 프로세스를 포함해
0.814 CPU초(한 코어 기준 평균 2.77%)를 사용했다. 부하 중 수집과 장치
대응의 안정성은 H2에서 확인해야 한다. NVIDIA의 info에는 PCI 버스 ID가
없었고, NPU에는 `pci_bus_id`가 있었다. 이 자료는 합성 기대값이 아니다.
