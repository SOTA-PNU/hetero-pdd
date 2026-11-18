# llm-d EPP 정책

`cmd/epp`는 공개 plugin registry에 factory를 등록한 뒤 공식 `runner.NewRunner().Run`을 실행한다. Go module은 `github.com/llm-d/llm-d-router v1.0.0`과 Go 1.27.2를 고정한다. InferencePool 하나가 하나의 모델 revision과 tokenizer를 서비스한다. 배포 도구는 실제 모델명을 `route-contract.parameters.servedModel`에 넣는다. 프로젝트 플러그인은 Alpha로 등록되므로 이 바이너리는 `--allow-experimental-plugins`로 시작한다. [공식 runner와 확장 API](https://github.com/llm-d/llm-d-router/tree/v1.0.0/pkg/epp/framework/interface)

## Native profile 선택

실험 YAML의 `profile`은 `profiles/*.yaml` 중 하나 또는 사용자 `llm-d.ai/v1 EndpointPickerConfig` 파일을 가리킨다.

| YAML | 실행 규칙 |
|---|---|
| baseline | 공식 random-picker가 full 역할 엔진 하나를 균등 추첨한다. GPU-only는 full 역할 엔진 구성을 제한하거나 label-selector-filter의 matchLabels에 backend를 추가한다. |
| greedy | free 슬롯을 endpoint order 순서로 한 라운드에 하나씩 배치한다. 완료된 슬롯은 호환 후보를 가진 가장 오래된 waiter가 받는다. |
| round-robin | 인스턴스 간 한 번씩 순환한다. endpoint order와 ID로 순서를 정하며 후보가 변경되면 현재 cursor를 새 길이에 맞춘다. |
| least-queue | fresh-queue-filter와 공식 queue-scorer/max-score-picker가 waiting 수가 가장 적은 full 엔진을 고른다. 동률은 공식 picker가 처리한다. prefix-cache scorer는 설정에 없다. |
| fixed-pdd | prefill과 decode profile 결과의 호환 pair 중 합산 profile 점수가 가장 높은 조합을 선택한다. 동률이면 endpoint order 순이다. pair-handler.parameters의 producer와 consumer에 ID를 지정해 고정할 수 있다. 유효 pair가 없으면 요청이 실패한다. |
| selective | fresh metrics, 모델별 비용표와 요청 특성으로 가능한 모든 full 엔진과 P/D 조합을 비교한다. |
| tiktok | selective와 같은 후보를 비교하며 사용 가능한 host 슬롯과 byte credit이 있고 디코더가 작업 중이면 예상 전송 시간 중 숨길 수 있는 부분을 반영한다. |
| trace-allocation | 연구 trace의 고정 배정 규칙을 재생한다. x-pdd-request-index와 allocation으로 배정한 ID를 찾아 full 또는 PDD를 선택하고 해당 target 슬롯을 점유한다. |
| example-weighted | 독립적인 endpoint-label-scorer 두 인스턴스와 공식 max-score-picker를 조합한다. 원본 예제는 CUDA 가중치 2, RBLN 1이다. RBLN 가중치를 4로 바꾸면 선호 엔진이 바뀐다. |

greedy와 trace-allocation은 단일 EPP 프로세스의 슬롯 장부를 사용한다. 제공 배포는 EPP 한 개를 사용하며 여러 EPP replica 간에 greedy 장부를 동기화하는 기능은 없다. 스트리밍 중간 chunk는 슬롯을 유지한다. 공식 최종 ResponseBody hook이 정상 응답, backend 오류, ext-proc 중단에서 슬롯을 회수한다. 대기 중 취소는 FIFO 대기열에서 제거한다.

compiler가 계산한 양 끝의 동일한 model/KV group, 서로 다른 ID, 각 역할 label, consumer의 `pdd.sota.dev/peer-<producer-id>=true`를 모두 검사한다. Kubernetes에서 producer에 `pdd.sota.dev/engine-id-source=pod-name`을 지정하면 공식 discovery의 `EndpointMetadata.Name`과 Downward API의 Pod 이름이 engine ID가 된다. Pod 교체 시 새 이름으로 갱신된다. 파일 discovery는 기본적으로 정적 `pdd.sota.dev/engine-id`를 쓴다. PDD는 선택한 producer의 raw engine port, engine ID, KV port, TP, block size와 block 수를 PD sidecar에 전달한다. 새 decoder가 producer와 연결할 수 있는지는 배포 compiler가 해당 transport와 장치 조건으로 증명해야 한다.

## 비용 입력과 관측값

모델과 배포에 맞춰 `cost-route-handler.parameters`를 설정한다. 샘플 `costs`는 기능 검증용 합성값이다. ID별 row가 backend row보다 우선한다. `prefillMsPerToken`과 `decodeMsPerToken`은 서비스 시간의 선형 근사값이다. `remainingDecodeTokens`는 현재 배치의 예상 잔여 토큰 수다. `transferFixedMs`와 `transferMsPerToken`은 producer row에서 읽는다. 특정 링크의 측정값은 `links.<producer-id>.<consumer-id>.{transferFixedMs,transferMsPerToken}`로 덮어쓴다. `kvBytesPerToken`은 decoder의 토큰당 host KV 바이트다. 선택적 RFC3339 `validUntil`이 지나면 이 표를 쓰는 요청이 실패한다.

입력 길이 I는 공식 parser가 가진 토큰 ID 수를 우선한다. 텍스트만 있으면 UTF-8 바이트 수를 `textBytesPerToken`으로 나눠 올림한다. 구조화된 텍스트도 없으면 본문 바이트 수를 쓴다. 출력 길이 O는 `outputPriorTokens`와 명시된 출력 상한 중 작은 값이다. EOS 이후 엔진이 보낸 usage가 실제 관측값이다.

공식 metrics-data-source가 기본 1초 주기로 Prometheus를 읽는다. live-state-extractor는 그 scrape에서 본 waiting/running 합계, 최대 KV 사용률, rank 중 최소 host credit을 timestamp와 함께 저장한다. 누락되거나 유한하지 않은 series는 해당 scrape에서 결측 처리한다. `ttlMillis`보다 오래된 표본은 사용하지 않는다. Selective 후보는 queue, running, KV 사용률을 모두 요구하고 `maxKvUtilization`에 도달한 엔진을 제외한다. lookahead가 켜진 RBLN의 vLLM multiprocess gauge에 초기 표본이 없으면 해당 항목의 실제 scheduler gauge(hetero_pdd_waiting_requests, hetero_pdd_running_requests, hetero_pdd_kv_usage)를 읽는다. 양쪽 모두 없으면 결측 상태를 유지한다. Least-queue는 waiting을 요구한다. host credit만 누락되면 일반 full/PDD 비용 비교를 계속한다.

- 엔진 대기 W는 `floor((waiting + running) / slots) × remainingDecodeTokens × decodeMsPerToken`으로 추정한다. 관측 시점의 작업 수와 평균 잔여 비용을 사용하는 배치 근사다.
- Full의 첫 토큰 예상 시각은 `W + I × prefillMsPerToken + decodeMsPerToken`이다. 완료 시각은 첫 토큰 시각에 나머지 출력 토큰의 decode 시간을 더한다.
- PDD에서 P의 준비 시각 A는 `W_p + I × prefillMsPerToken_p`이고 전송 시간 T는 `transferFixedMs + I × transferMsPerToken`이다. 일반 PDD의 첫 decode 시작은 `max(A, W_d) + T`로 추정한다.
- TikTok은 decoder가 실행 중이고 `W_d > A`이며 EPP의 미완료 예약을 차감한 host credit이 충분할 때 `max(A + T, W_d)`를 쓴다. 필요 byte는 decoder block 단위로 올림한 토큰 수와 `kvBytesPerToken`의 곱이다. 실제 엔진의 슬롯·byte 예약과 host→device 동기화가 최종 admission을 결정한다.
- 기본은 예상 완료 시간이 가장 짧은 후보를 고른다. 선택적 요청 헤더 `x-pdd-ttft-ms`가 있으면 먼저 TTFT 목표 초과량을 최소화하고 완료 시간과 native profile 점수로 동률을 푼다. 선택 이유·실행 경로는 응답의 `x-hetero-{policy,reason,path,prefill,target,preload}`와 EPP 로그에 기록한다. `hetero_route_prediction`의 예측과 `hetero_route_finished`의 usage를 동일 `request_id`로 연결해 예측 오차를 계산할 수 있다.

## 연구 구현의 대응 관계

기존 연구 checkout `pdd_lookahead_20260926-202449`의 commit `f2c5f1ed4b0a840cbc9dffc845858643406dcc87`, `tools/run_trace.py:route_for`는 trace index별 장치 allocation을 사용한다. non-GPU 배정에서 `pdd_request_indices`가 생략되면 전체를 PDD로, 명시되면 해당 index만 PDD로 보낸다. `trace-allocation`은 이 규칙을 옮겼다. endpoint ID, `pddTargets`, 선택적 `pddRequestIndices`로 같은 배정을 표현한다. 예제의 index 0, 1, 2는 각각 gpu0 full, npu0 PDD, npu0 full이다. trace generator가 부여한 index를 그대로 보내야 한다. 실제 FIFO 도착 순서와 wire timing은 gateway로 요청이 도착한 시점에 따라 달라진다.

현재 `selective`는 온라인 요청을 위한 위의 비용 모델이고, `tiktok`은 사용자 정의의 decode 중 선행 host KV 수신 기회를 추가한다. 이전 trace의 static allocation 및 별도 `tail_aware_pdd` 실험(commit `1046b8b3cddcde4baa1ba822950e539fcd9ec509`)의 cohort 계획과는 각각 다른 정책이다. 이전 알고리즘의 성능 재현 여부는 동일 trace·자원·실측 비용표와 반복 실험을 통해 평가해야 한다.

## 확장과 검증

공개 Filter/Scorer/Picker/ProfileHandler 또는 request/data-layer 인터페이스를 구현하고 `plugins/register.go`에 factory를 등록한다. `plugins/label_scorer.go`와 `profiles/example-weighted.yaml`이 최소 예제다. Go 단위 테스트와 race 테스트에는 호환 행렬, stale/missing 지표, 비용 만료, 길이 추정, TikTok credit, greedy 슬롯 회수·resize, historical trace index 회귀가 포함된다.

```sh
# repository root
(cd llmd/epp && go test ./... && go test -race ./plugins)
python -m llmd.epp.tests.policy_process --bin-dir /path/to/built/bin --work /new/run-directory
```

2026-11-18에 Go 1.27.2에서 위 Go 테스트가 통과했다. process suite는 실제 Envoy·공식 EPP·PD sidecar를 실행한다. HTTP 엔진 fixture가 합성 metrics와 token/lease를 제공하며, 결과·요청별 경로와 실제 프로세스 로그를 실행 디렉터리에 남긴다. 실제 모델 token correctness와 장치 간 전송·overlap은 엔진 실행 결과에서 별도로 확인한다.
