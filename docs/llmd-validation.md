# 2026-11-18 검증 기록

검증한 기준 브랜치의 출발점은 `5aa0bfdc035b762092d1119b80eca03c52e072ba`이다. 요청 ID, 출력 토큰, 블록 매핑, 실제 이미지 digest 및 이벤트 시각을 선별해 JSON으로 보존했다. 원본 로그는 이 호스트의 `~/sota/official-llmd-validation/` 아래에 있다. [요청별 증거](llmd-evidence.json) · [실행 명령](llmd.md)

## 결과

| 검증 | 실행한 조건 | 관측 결과 |
| --- | --- | --- |
| 설정 및 엔진 회귀 | 로컬 Python 테스트 10개, 기존·신규 RBLN 스케줄러와 connector 테스트 99개, Go plugin 단위·race 및 vet | 통과. 실행 취소·실패 시 예약 회수, 불일치 reference 거부, 장치 소유권 및 준비된 캐시 세대 재사용을 포함한다. |
| 공식 serving과 정책 | 실제 Envoy, ext-proc EPP 및 patched upstream PD sidecar + 프로토콜 시험 엔진 | 정책 시나리오 10개, 요청 28개. 한 실행 내 full/PDD 선택, 가중치 반전, trace 배정 재생, greedy 취소와 backend 오류, queue 지표 누락을 검사했다. |
| 2P+2D 및 오류 격리 | 공식 파일 discovery, 생산자 2개·소비자 2개인 HTTP 시험 환경 | 검사 17개. 두 번째 P/D 쌍으로의 재선택을 관측했다. 잘못된 producer/request·토큰·블록·address·port·transfer 방향과 누락 handle을 거부했다. 손상된 handle은 디코더에 도달하지 않았다. |
| Qwen2.5-0.5B 정확성 | 실제 RTX PRO 6000 → RBLN-CA22, FP16/HND, 독립 GPU full baseline | 257토큰 입력 2건의 출력 각 8토큰이 기준과 일치했다. 요청마다 KV 3블록을 import했고 NIXL 합계는 9,437,184바이트였다. |
| 실제 1P+2D | 같은 Qwen2 모델, GPU 1개 및 RBLN 카드 2개, 공식 file-discovery 변경 | npu0과 npu1에 각각 2건이 도착했다. 네 요청 모두 해당 카드의 worker import 및 GPU 기준과의 출력 일치를 확인했다. |
| 컴파일 캐시 | Qwen2 257토큰 실험 종료 후 새 run으로 동일 구성 재시작 | 양쪽 엔진에서 같은 cache key와 게시된 세대를 재사용했다. RBLN 최초 실행에서 compiler compile 배너 2개, 재시작에서 0개를 기록했다. 재시작 후에도 3블록 import와 출력 일치를 확인했다. |
| Qwen3-0.6B 선행 수신 | 실제 RBLN 실행 한도 1, prepare depth 1, host byte budget 128 MiB | A의 request-scoped decode step 중 B의 native READ 진행과 완료를 관측했다. 수신량 14,680,064바이트, 해당 step과 수신 구간의 교집합 7.566119 ms, Host 보관 8.45635749초, H2D 17.834925 ms였다. |
| 선행 수신 결과 및 취소 | 새 GPU full 배포에 동일 prompt와 생성 옵션 재전송. 별도 요청에서 실제 클라이언트 연결 해제 | A의 896개와 B의 32개 출력 token ID가 각각 전부 일치했다. 취소 시 할당 블록을 격리·반납했고 다음 PDD 요청이 성공했다. |
| Kubernetes 제어 계층 | Kubeconform, 실제 Kubernetes 1.34.1 API 서버·etcd·Pod informer, chart TLS Envoy/EPP, 모의 엔진 | 2P+2D 리소스 15개가 strict schema 및 API 검사를 통과했다. Ready Pod 교체를 발견했고, 교체된 생산자 A의 lease는 HTTP 502로 거부한 뒤 현재 B의 lease는 HTTP 200으로 처리했다. 시작 timeout 후 배포가 만든 리소스만 정리하고 외부 ConfigMap은 보존했다. |

Qwen2 실행의 가중치 revision은 `060db6499f32faf8b98477b0a26969ef7d8b9987`이고 architecture는 `Qwen2ForCausalLM`이다. Qwen3는 `c1899de289a04d12100db370d81485cdf75e47ca`, `Qwen3ForCausalLM`이다. 독립 replay에서 실제 Qwen3 가중치 SHA-256도 대조했다. 각 비교는 해당 prompt, seed, dtype, tokenizer 및 출력 길이에 적용된다.

## 원본 결과 위치

아래는 검증 디렉터리 기준 상대 경로다.

- `incarnation-policies/policy-results.json`, `incarnation-protocol/protocol-results.json`, `incarnation-go-race.log`, `engine-tests/pytest.log`, `launcher-unit.log`
- `q2-257-gpu/verification.json`, `q2-257-fixed/verification.json`, `q2-257-warm/verification.json`. 각 디렉터리의 `run.json`에 실제 설정, 런타임과 장치가 있다.
- `q2-pool-1p2d/first-decoder.json`과 `second-decoder.json`. 두 검사 사이 첫 디코더를 native file inventory에서 제외해 나머지 consumer까지 검증하고 원래 inventory를 복구했다.
- `q3-tiktok-v2/overlap-verification.json`, `cancel-verification.json`, `overlap-gpu-reference.json`, `run-binaries.json`, `npu0-admission.jsonl`, `events/npu0/events_0.jsonl`. 수정 전 cold-start 실험과 forward dispatch 구간만으로 판정했던 초기 실패 기록도 보존했다. 최종 probe는 A의 모델 호출과 sampler를 포함하는 decode step 안에서 B의 READ 진행/종료 시각을 대조한다.
- `k8s-render/kubeconform.log`, `k8s-render/kubectl-server-validation.log`, `k8s-incarnation/result.json`, `cluster-lifecycle/check.log`, `serving-build.json`

최종 Linux amd64 EPP 이미지 digest는 `sha256:b507105dc62b22f91ab61b810becc52cb162b5954346e2c710dc9d09518038a5`, PD sidecar는 `sha256:af3dcdb3ab6018ac888b670d9d3bd2c0ed077c86d908974a746ff2620020b7e3`이다. 이미지 archive 안의 실행 파일과 프로세스 통합 시험에 사용한 바이너리의 SHA-256이 일치했다.

## 지원 범위와 남은 검증

- 이 호스트의 GPU는 1개다. 실물 2P+2D, GPU→GPU TP2, 다중 노드 NIXL, BF16, 다른 RBLN SKU 및 Qwen3-1.7B의 새 serving 경로는 미검증이다.
- Kubernetes 검증은 실제 API·RBAC·discovery·Proxy 및 배포 실패 정리까지 수행했다. GPU/NPU device plugin과 kubelet이 있는 클러스터에서 엔진 컨테이너 기동, imageID 수집과 장치 통합은 미실행이다. 엔진 이미지는 기존 Dockerfile과 wheel로 사이트에서 빌드해야 한다.
- Host overlap은 실물에서 관측했다. device kernel 수준의 동시 실행, 지연·처리량 개선, 연구 논문의 성능 재현은 측정하지 않았다. 전송 timeout 및 엔진 소실의 위험 경로는 모의 상태·회귀 테스트 범위다. 실제 클라이언트 취소는 장비에서 검증했다.
- RBLN producer, 이종 TP shard 재분배, 노드 간 CUDA→RBLN host import는 구현 범위 밖이다. 배포 검증기가 이 조합을 거부한다. 단일 EPP의 greedy 장부를 여러 EPP replica가 공유하는 기능도 제공하지 않는다.
- CI trigger·job·artifact 연결, profiler의 비용표 생산·부하 분석·리포트 게시가 담당자 후속 작업이다. 기존 CI 및 profiler 소스는 변경하지 않았다.
