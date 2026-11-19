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

실험 YAML에서 native EPP profile을 지정한다. `cost-route-handler.parameters.costs`는 endpoint ID별 row를 우선하고 backend row를 다음으로 사용한다. `tiktok.yaml`의 주석에 적용 가능한 모델 revision, 장치, 설정과 측정 조건이 있다. `selective.yaml`은 합성 비용 예제다. 선택적 RFC3339 `validUntil`이 지나면 요청이 실패한다.

입력 길이 I는 parser의 토큰 ID 수, 구조화된 텍스트의 UTF-8 byte 수, 본문 byte 수 순서로 구한다. byte 단위 입력에는 `textBytesPerToken`을 적용한다. EOS를 무시하고 별도의 stop 조건이 없는 고정 길이 요청에서는 명시한 출력 상한이 예상 길이 O가 된다. 일반 생성에서는 `outputPriorTokens`를 사용하고 클라이언트의 출력 상한이 작으면 그 값으로 제한한다.

| 항목 | 단위와 계산 |
|---|---|
| Prefill P | `prefillFixedMs + I × prefillMsPerToken` |
| 평균 decode 간격 D | `decodeMsPerToken + decodeMsPerContextToken × (I + (O-1)/2)` |
| Full TTFT | `W + fullFixedMs + P`. 기존 비용표에서 `fullFixedMs`를 생략하면 첫 토큰 비용으로 D를 더한다. |
| Host 수신 T | `links.<producer>.<consumer>`의 `transferFixedMs + I × transferMsPerToken + blocks × transferMsPerBlock`. 링크가 없으면 producer row의 기존 fixed/token 값을 쓴다. |
| Import H | 같은 링크의 `importFixedMs + blocks × importMsPerBlock` |
| PDD 준비 A | `W_p + P_p + dispatchFixedMs`. GPU 예상 점유는 `W_p + P_p`에서 끝난다. |
| 일반 PDD TTFT | `max(A, W_d) + T + H + D_d` |
| Lookahead PDD TTFT | `max(A + T, W_d) + H + D_d` |
| 완료 시각 | `TTFT + (O-1) × D`. 모든 시간은 현재 선택 시점 기준 ms다. |

입력에 필요한 KV block 수는 decoder의 block 크기로 올림한다. Host 수신 예약은 정렬된 토큰 수와 `kvBytesPerToken`의 곱을 사용한다. `preload`가 켜지고 수신 도착 시각에 이전 요청의 예상 decode 구간, decoder 대기, host 슬롯 및 byte 예산이 모두 있을 때만 선행 수신을 선택한다. 슬롯·byte 상한을 제공하는 배포에서는 최신 gauge와 EPP 미완료 예약을 함께 사용한다. 최종 할당과 안전한 import 시점은 엔진이 확인한다.

공식 metrics-data-source가 waiting/running 합계, 최대 KV 사용률, rank 중 최소 host headroom을 수집한다. RBLN scheduler가 state transition마다 갱신한 gauge가 우선하고 없는 항목은 vLLM 통계를 사용한다. `ttlMillis`보다 오래된 표본, queue/running/KV 결측, `maxKvUtilization`에 도달한 엔진은 제외한다. Host credit만 결측인 경우 일반 full/PDD는 계속 비교한다.

한 EPP 안에서 경로 선택과 단계별 예약이 같은 lock을 사용한다. 슬롯별 대기 W에는 이미 배정한 요청의 실제 길이와 예측 단계 시간이 반영된다. 들어온 stream data event 수로 잔여 decode 추정을 갱신한다. 여러 토큰이 하나의 event로 합쳐지면 근사값이다. 동일 scrape에 보일 수 있는 local 요청 수를 관측 queue/running 합계에서 빼고, 출처를 식별할 수 없는 초과 작업에만 `remainingDecodeTokens`를 적용한다. 값이 없으면 출력 prior를 쓴다. 정상 완료·오류·연결 종료에 대한 최종 ResponseBody hook이 예약을 해제한다. 이미 종료된 요청의 시각만 다음 유효 scrape와 대응시키기 위해 보관한다. 실제 request ID가 없는 외부 scrape와 다른 EPP replica의 작업량은 추정에 의존한다.

`objective: minimumCompletion`은 예상 완료 시간을 최소화한다. `workConserving`은 이 EPP에 미완료 요청이 있고 후보의 primary 슬롯 사용 가능 여부가 다르면 즉시 시작 가능한 primary를 우선한다. 두 경우 모두 선택적 `x-pdd-ttft-ms`의 목표 초과량이 우선하고, 동률은 native profile 점수로 푼다. `hetero_route_prediction`에 입력/출력 추정 근거, 단계별 ms, 예측한 host 수신 단축 시간이 기록된다. `request_id`로 클라이언트 결과와 `hetero_route_finished`를 연결한다. 실제 overlap 판정에는 worker 이벤트와 admission 기록을 사용한다.

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
