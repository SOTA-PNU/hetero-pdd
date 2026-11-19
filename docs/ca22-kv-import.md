# CA22의 KV import와 로컬 prefill

Qwen3-0.6B의 FP16 KV를 RTX PRO 6000 Blackwell에서 CA22로 넘기는 host staging 경로를 측정했다. 출발 브랜치 `exp/tiktok-hardware-20261120`의 실제 로컬·원격 HEAD는 모두 `edecb026d403b70dd98242be4686f0e1da23d78d`였다. 이전 검토 이후 해당 브랜치에 추가 변경은 없었다. 주 평가지표는 실제 서비스의 출력 tokens/s다. import는 전송 후 NPU 캐시에 쓰는 구간이며, 앞선 네트워크 READ는 별도 지표다.

**확인한 병목은 RBLN의 논리 FP16을 장치의 물리 FP16 형식으로 바꾸며 복사하는 경로였다.** 실제 링크에서 공식 도구는 56 MiB H2D를 57.48 GB/s로 보냈지만 애플리케이션의 단일 FP16 복사는 0.67 GB/s였다. 호출 수를 줄이는 기존 coalescing과 입력값 변환 비용을 줄이는 새 구현은 서로 다른 비용을 낮췄다. 실제 KV의 절반가량에서 이 변환 최적화가 적용될 수 있는 비트 패턴이 관측됐다(첫 view 4,096개 원소 중 2,078개).

- 독립 보정에서 128 / 256 / 512토큰 기본 import **22.62 / 44.46 / 86.73 ms**가 최종 **17.81 / 32.86 / 61.45 ms**로 줄었다. 세 크기 모두 NPU 로컬 prefill이 더 빨랐다.
- 고정 PDD 512/32에서 기본 대비 평균 TTFT는 **269.87→237.66 ms**, 처리량은 **52.57→54.94 tokens/s(+4.51%)**였다. 기존 coalescing과 새 구현의 짝 비교는 유효한 2회에서 +1.32%였고 두 반복의 방향이 달랐다. 한 블록에서 기본 대비 처리량 변화는 −0.04%였다.
- 동일한 최종 구현에서 고정 PDD lookahead는 **90.68→94.14 tokens/s(+3.82%)**, 실제 겹침은 17건이었다. 자동 정책은 **710.69 tokens/s**를 내며 192건을 GPU full 165건, NPU full 27건으로 처리했다. 자동 정책에서 PDD는 선택되지 않았다.
- 세 최종 시험의 전체 측정 출력은 독립 GPU 대비 **1,055/1,056개 요청, 107,517/107,520개 토큰 위치**에서 정확했다. 불일치 1건은 기존 coalescing 대조에서 발생했다. 정규화를 활성화한 최종 PDD 요청은 144/144개가 일치했다.

## 장비와 측정 경계

- 평가 대상은 GPU 0 `GPU-3bfcb8e6-d98d-00e9-6a2a-3b04c61792a3`와 NPU 0 `bc133fcd-240a-473d-b194-8db986cdeb53`이다. NPU의 PCI 주소는 `0000:34:00.0`, 실협상 링크는 32 GT/s ×16, NUMA 노드는 0이다. GPU PCI 주소는 `0000:70:00.0`이며 GPU 링크는 유휴 시 2.5 GT/s ×16, 서비스 중 32 GT/s ×16까지 확인했다. 두 장치의 NUMA 노드는 0이다. CPU는 Xeon w7-3455, 호스트 허용 CPU는 0–47이다. CA22 KMD·firmware는 3.2.2이고 GPU 드라이버는 580.105.08이다. 제조사가 표기한 **호스트 인터페이스 사양은 64 GB/s**다. 아래 GB/s는 단방향 실측이고 MiB는 2²⁰ 바이트다. [CA22 제품 사양](https://rebellions.ai/rebellions-product/rbln-ca22/)
- 모델 revision은 `c1899de289a04d12100db370d81485cdf75e47ca`다. 양쪽 모두 FP16, HND, TP1이며, 블록 크기 128토큰, 캐시 32블록, context 1,024, prefill chunk 128, 동시 활성 시퀀스 1로 고정했다. 28개 레이어의 K/V를 합치면 블록 하나는 14,680,064바이트다. NPU에는 host staging slot 1개와 예산 128 MiB가 있다. NPU Python 환경은 torch 2.11.0+cpu, torch-rbln 0.3.0, rebel-compiler 0.11.1.post2.dev2+g2995098f.prod, vLLM 0.22.0+cpu, NIXL 1.4.1이다. GPU·NPU 각각의 패키지와 PCI/NUMA 원본은 압축 파일의 `environment/`에 있다.
- 기존 host buffer는 레이어마다 `rebel.kv_cache.aligned_tensor`로 만든다. 내부 NumPy 할당에서 4 KiB 경계를 맞추고 페이지를 미리 접근한 뒤 같은 저장소를 torch tensor로 본다. 진단 버퍼에서 주소 나머지는 0이었고 Linux 매핑의 `Locked`는 0 kB, `is_pinned("rbln")`는 false였다. 엔진은 이 host 저장소를 NIXL에 별도로 등록해 GPU→host READ에 쓴다. NIXL 등록만으로 해당 포인터에 대한 RBLN H2D 직접 DMA 등록이나 영구적인 page lock을 입증할 수 없다. 진단 버퍼에는 NIXL 등록이 없으며, 등록된 실제 엔진의 시간도 별도로 측정했다.
- 설치된 `torch_rbln/device/device.py`의 `torch.rbln.synchronize(device)`는 해당 장치의 미완료 전송을 기다린다. SDK 헤더 `rebel/runtime/api/rbln_runtime_api.h`는 비동기 handle이 0이면 동기 완료이며, 비동기 목적지를 소비하기 전에 transfer wait 또는 device synchronize가 필요하다고 명시한다. 진단은 batch 전 idle fence, 동기 `copy_` batch, batch 후 fence 순서다. CPU의 호출 반환과 완료 확인 시각을 각각 기록했다. 빈 fence 중앙값은 0.00177 ms, P95는 0.00229 ms다. 엔진의 실제 import는 worker 시작부터 반환까지의 host 구간이다. 엔진에 새로운 장치 동기화를 삽입하지 않았다.
- 모든 할당·등록·초기화·컴파일과 정확성용 readback은 진단 타이머 밖에서 수행했다. 장치 메모리, 런타임의 실제 H2D 횟수·바이트, 컴파일된 장치 연산을 거친 내용과 미선택 블록을 함께 확인했다. 2번째 CA22의 보조 경로 분해 측정은 장치 0의 성능 표와 구분했다.

## 바이트 수, 호출 수, tensor view 분리

공식 `rblnBandwidthLatencyTest` 3.2.2의 H2D 시험 `--tc=3`에서 14 MiB는 평균 **24.79 GB/s**, 56 MiB는 **57.48 GB/s**였다. 사전 독립 실행에서는 각각 40.94와 46.72 GB/s였다. 애플리케이션의 단일 연속 FP16 복사는 같은 장치에서 각각 **1.03, 0.67 GB/s**였다. 공식 도구의 결과는 실행마다 달랐지만 두 경로의 차이는 유지됐다. 두 값 모두 D2H를 제외했다. [공식 도구 설명](https://docs.rbln.ai/latest/software/system_management/management_tools/rbln_bandwidth_latency_test.html)

호출 수와 바이트 수를 분리하는 이 표에서는 CPU intra-op과 inter-op을 각각 1로 고정했다. 실제 워커의 12스레드 조건은 다음 절에서 별도로 측정했다. 각 셀은 5회 warmup 뒤 30회 측정했고 순서를 바꿔가며 실행했다. 표의 host는 뷰 선택과 호출을 포함한 batch 반환까지의 중앙값이다. 완료는 batch당 fence 하나를 포함한다. 분산은 완료시간의 표본분산이다. 각 샘플의 실제 바이트·호출 수, 내부 호출시간 합계·reorder·fence, 블록 ID, source/destination shape·stride·offset은 `micro/factorial-final/{samples.jsonl,plans.json}`에 있다.

| CA22 0번 조건 | MiB | 호출 | 내부 copy 호출 합 중앙 ms | Host 전체 중앙 ms | 완료 중앙 ms | 완료 P95 ms | 분산 ms² |
|---|---:|---:|---:|---:|---:|---:|---:|
| 단일 연속 버퍼 | 14 | 1 | 14.197 | 14.227 | 14.239 | 17.180 | 3.793 |
| 단일 연속 버퍼 | 56 | 1 | 87.205 | 87.222 | 87.238 | 91.630 | 68.107 |
| 연속 버퍼를 동일 크기로 분할 | 56 | 56 | 80.717 | 81.134 | 81.143 | 85.090 | 39.517 |
| A1 · 레이어 K/V별 1블록 | 56 | 224 | 87.164 | 88.557 | 88.576 | 93.598 | 35.354 |
| A2 · 레이어 K/V별 2블록 | 56 | 112 | 81.770 | 82.801 | 82.812 | 88.760 | 39.787 |
| A3 · 레이어 K/V별 4블록 | 56 | 56 | 77.870 | 78.897 | 78.913 | 83.729 | 25.317 |
| B1 · 레이어 K/V별 1블록 | 14 | 56 | 21.898 | 22.451 | 22.459 | 24.393 | 2.723 |
| B2 · 레이어 K/V별 2블록 | 28 | 56 | 41.005 | 41.698 | 41.707 | 44.676 | 10.151 |
| A3와 같은 28개 개별 allocation의 평탄화된 view | 56 | 56 | 77.099 | 78.320 | 78.330 | 81.820 | 36.075 |
| 파편화된 allocator 맵 재생 · 4개 구간 | 56 | 224 | 87.461 | 88.830 | 88.846 | 94.152 | 22.999 |

A1–A3는 동일한 오름차순 물리 블록에 같은 56 MiB를 썼다. 168개 호출을 줄이면 9.66 ms 절약됐다. 호출을 56개로 고정하고 14→56 MiB로 늘리면 56.45 ms 늘었다. 동일 레이어 allocation과 동일 호출로 view만 평탄화한 차이는 0.58 ms로 셀 내 변동보다 작았다. 5개 KV 셀 중앙값에 맞춘 식은 시간(ms) ≈ 0.570 + 0.05628 × 호출 수 + 1.34883 × MiB였다. A1/A2/A3/B1/B2 순서의 잔차는 −0.135, +0.405, −0.343, −0.146, +0.219 ms다. 이 계수에는 runtime dispatch와 형식 처리 및 실제 전송이 들어 있다. 단일 대형 버퍼까지 일반화하면 안 된다.

### 수치 형식 변환과 호스트 스레드

설치한 런타임의 `rbln_get_memory_info`에서 컴파일된 KV 형상의 logical dtype은 IEEE Float16(7), physical dtype은 CustomFloat16/DLFloat16(13)이었고 물리·논리 shape은 같았다. 장치 profiler에서 `rbln_memcpy_h2v`가 호출시간 대부분을 차지했다. 여기에는 수치 형식 변환, 가상 메모리 처리와 DMA가 모두 포함된다. 복사 호출 합계와 outer import의 차이, 빈 fence, view 비교로 보면 Python 인덱싱과 fence가 수십 ms의 주된 원인일 가능성은 작다. H2V 내부 각 하위 단계의 정확한 서비스 시간 비율은 측정되지 않았다. 동시성 1의 ordinary PDD와 유휴 장치에서도 문제가 재현되어 독립 버퍼로 decode 간섭을 유발하는 추가 실험은 실시하지 않았다. 동시에 decode하는 다른 부하에서의 CPU·메모리 경합 크기는 열려 있다.

별도 CA22 1번에서 1 MiB FP16 view의 논리 복사는 기본 host에서 0.982 ms, torch-pinned에서 0.975 ms였다. 논리·물리 형식이 같은 INT16 대조는 각각 0.280, 0.078 ms였다. 레이어별 THP host 할당과 외부 Python 스레드에 나눠 맡긴 복사도 지속적인 이득이 없었다. 미리 물리 형식으로 변환한 뒤 typed copy를 쓰면 반환 시점에는 H2D가 0바이트였다. 후속 장치 준비 확인에서 168 MiB가 전송되어 준비 완료까지 총 169–184 ms가 걸렸다. 원래 논리 복사는 정확히 요청한 바이트를 전송했고 같은 확인에서 추가 전송은 0바이트였다. 이 비교에서 예측 가능한 소비 시점과 요청별 캐시 범위를 지킨 경로는 동기 논리 복사였다. SDK·형식·카드 1번 실험의 재현 소스는 원본의 `runtime/`에 있다.

실제 CA22 0번에 TP1 worker의 CPU 설정을 적용하고 28개 개별 장치 allocation으로 재측정했다. 12스레드 상한과 CPU 0–23은 준비가 끝난 서비스 엔진에서 관측했다. 각 셀은 5회 예열 후 30회이며, 표 값은 batch 완료 중앙값 / P95(ms)다. 14 MiB와 56 MiB 모두 호출 56회다.

| 호출 스레드 상한 | 상속(12) | 1 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| 14 MiB | 10.84 / 12.75 | 18.87 / 21.55 | 11.11 / 13.45 | 10.32 / 11.26 | 10.29 / 12.62 | 10.39 / 12.00 |
| 56 MiB | 20.84 / 22.71 | 60.73 / 70.87 | 29.57 / 31.97 | 22.07 / 23.73 | 19.95 / 22.25 | 19.50 / 20.79 |

같은 회차에서 8→12의 56 MiB 단축 중앙값은 2.08 ms(95% bootstrap 1.81–2.58)였다. 12와 16의 차이 구간은 −0.39~+0.94 ms로 0을 포함했다. 성능을 구분하기 어려우면 CPU를 적게 쓰기로 사전에 정해 12를 선택했다. 4개로 파편화된 실제 물리 ID `[2,20,29,31]`에서는 224회가 필요했고 상속 12에서 40.49 ms, 8에서 41.78 ms였다. 새 설정 `engine.kv_import_threads`는 가져오는 호출의 OpenMP 상한만 한 batch 동안 바꾸고 원래 값을 복구한다. 기존 서비스도 12를 상속하므로 1→12의 진단상 이득은 배포 개선량으로 셀 수 없다. 운영에서는 원래 per-block 복사, 기존 coalescing, coalescing+12를 독립적으로 비교했다.

각 실제 KV 조건은 234,881,024개 전체 원소에서 요청 블록·미선택 블록·가드의 일치를 확인했다. 하드웨어의 H2D 카운터는 각 경우의 바이트·호출과 정확히 같았고 장치 준비를 확인할 때 추가 H2D는 0이었다. 별도의 56 MiB **단일** 버퍼를 연달아 갱신한 카드 0번 보조 실험에서는 8스레드에서 이전 값이 한 차례 관측됐다. 최초 불일치는 원소 2048이었다. 원인 분리 표의 기본 단일 전송과 14개 실제 KV 조건은 각각 검증을 통과했다. 카드 1번에서 상속→상속·상속→8·8→상속을 별도 할당에 재실행한 6회는 모두 정확했다. 카드 0번의 그 실패 원인은 미확정이며 원본의 `micro/failed-optional-flat/` 및 `micro/flat-replay/`에 기록했다.

## 실제 FP16 값이 만드는 추가 비용

앞선 진단의 가드 패턴에는 정확히 표현되는 작은 정수가 많았다. 이 패턴과 실제 모델 KV 분포의 차이를 분리하기 위해 동일 크기에 고정 seed의 정규분포에서 뽑은 소수 FP16을 넣었다. 설치된 SDK 변환기는 일반 유한값의 mantissa 최하위 비트에 따라 분기했다. CPU 한 코어에서 256 KiB를 변환하는 중앙값은 정수 패턴 171 µs, 소수 패턴 690 µs였다. 1 MiB에서는 660 µs와 2,773 µs였다. PCIe와 Python의 실행을 제거한 실험으로, 입력값에 따라 변하는 변환 비용을 확인했다.

이 SDK가 물리적으로 같은 CustomFloat16 비트를 만드는 IEEE FP16 입력을 골라 일반 동기 `copy_`에 전달했다. 정규화 대상은 일부 유한 정규수이며 ±0, subnormal, NaN, 무한대, 양·음의 최대 유한값은 그대로 둔다. 전체 65,536가지 원소 중 30,718개의 입력이 바뀌었다. 원본/정규화 입력을 설치한 SDK의 독립 스칼라 변환과 일괄 변환에 각각 대입해 물리 비트가 **65,536/65,536 일치**함을 확인했다.

아래 비교는 원래 물리 CA22 0번에서 28개 독립 레이어, 같은 주소 offset, 56회 호출, worker와 같은 CPU 설정을 사용했다. 입력 종류마다 새로 할당한 동형의 장치 캐시를 사용했다. 표 값은 완료 중앙 / P95 / 분산(ms, ms, ms²)이다. 정규화 모드는 재사용 scratch로 준비하는 시간을 포함하며, 목적지의 논리형을 해석하는 일반 동기 H2V API로 복사했다. 세 모드, 두 크기의 실행 위치를 교대했고 셀별로 5회 예열·30회 측정했다.

| 입력·처리 방식 | 14 MiB | 56 MiB |
|---|---:|---:|
| 작은 정수, 기존 복사 | 10.09 / 11.01 / 0.37 | 18.83 / 21.12 / 1.50 |
| 소수 FP16, 기존 복사 | 19.79 / 22.34 / 1.29 | 30.85 / 33.96 / 2.00 |
| 같은 소수 FP16, 정규화 후 복사 | 11.79 / 12.57 / 0.26 | 25.68 / 28.24 / 1.68 |
| 위 완료시간에 포함된 CPU 준비 중앙값 | 1.59 | 6.33 |

소수 입력의 전후를 회차별로 짝지으면 절약한 시간은 14 MiB에서 8.13 ms(95% bootstrap 7.89–9.05), 56 MiB에서 5.47 ms(4.69–5.93)였다. 각각 30회 모두 빨랐다. 원본과 최적화의 실제 장치 연산 결과, 직접 읽기, 다시 연산한 결과가 비트 단위로 일치했다. 전체 비트 패턴을 넣은 검사에서는 CPU의 IEEE 부호 반전식을 장치 산술의 기준으로 두자 +0의 결과 부호가 112곳에서 달랐다. 기존·최적화 장치 양쪽 모두 같은 +0을 반환했다. 원래 전송값을 직접 읽을 때는 signed zero까지 SDK 기준과 일치했다. 미선택 블록과 가드, 실제 H2D 바이트·호출 수도 모두 일치했고 준비를 확인한 뒤 추가 H2D는 없었다. 원본은 `runtime/value-cpu.json`, `runtime/value-device/{checks.json,samples.jsonl,summary.json}` 및 최초 부호 검사 전체를 보존한 `runtime/value-device-ieee-neg-zero/`에 있다.

실제 Qwen의 첫 적격 KV view에서도 처음 4,096개 중 2,078개(50.73%)가 같은 물리값을 갖는 다른 논리 입력으로 정규화됐다. 값 자체는 기록하지 않았다. 첫 서비스 구현은 1 MiB thread-local scratch를 쓰고 각 view의 물리 형상·상주 상태·실제 전송을 확인했다. 모든 PDD 요청의 100% 바이트에 적용됐고 18/18개 전체 출력이 GPU와 일치했다. 그러나 독립 calibration의 import는 128 / 256 / 512 입력에서 22.92 / 41.50 / 74.37 ms였다. 맞춘 coalescing 대조는 22.46 / 43.48 / 62.60 ms였다. 56 MiB에서는 준비에만 11.35 ms가 들었다. 이 실패를 `cal-canonical/`에 보존했다.

1블록 및 4블록의 단일 K/V view를 분리해 첫 서비스 워커와 진단 코드를 같은 카드에서 비교했다. 256 KiB에서는 워커 전체 317.84 µs, 같은 준비 함수와 직접 논리 복사 212.61 µs였다. 1 MiB에서는 610.34 µs와 465.76 µs였다. SDK 형상·범위 조회에 각각 50.73 µs와 63.13 µs가 들었다. 4 KiB에 정렬한 scratch의 동일 비교는 215.53 / 461.34 µs로 주소 정렬만 바꾼 효과는 작았다. CPU에 맞춘 벡터 명령으로 준비하는 시간은 35→21 µs와 135→87 µs였다. 장치 byte/call과 전후 가드 및 읽기 결과는 같은 조건에서 확인했다. 직접 분해 원본은 `runtime/worker-overhead/records.json`에 있다.


## 최종 구현과 독립 비용 보정

CPU에 맞는 AVX-512·AVX2 또는 일반 커널을 실행 시 선택한다. SDK 버전은 실제 로드된 라이브러리의 build ID로 확인하며, 변환표 전체를 검사한다. 논리 FP16/물리 CustomFloat16, 상주한 단일 장치 저장소, 연속한 동일 형상과 범위를 확인한다. 동기 호출 안에서 같은 레이어의 불변인 형상·범위 증거를 재사용하고, 복사마다 실제 전송 바이트와 완료 상태를 점검한다. 다음 요청에서 다시 검사하므로 그래프가 재바인딩될 때 오래된 형상 정보를 재사용하지 않는다. 원본 host KV는 수정하지 않는다. 현재 최적화는 한 view당 최대 1 MiB를 지원하며, 스레드당 최대 1 MiB scratch가 추가된다. 지원 범위를 벗어난 view, 재정렬 경로, 지원되지 않는 SDK에는 원래 복사를 적용한다. 시험 runner는 최적화 비교군의 모든 요청에서 요청한 바이트·호출 전부가 선택한 구현으로 처리됐는지 별도로 검증한다. 최종 독립 보정에서 1,680개 view, 588 MiB가 모두 최적화됐고 fallback은 0회였다.

보정 입력은 같은 prompt·seed로 GPU full, NPU full, ordinary PDD를 각각 격리 실행했다. 아래는 전체 출력이 GPU와 같은 요청의 중앙값이다. 각 길이·경로별 6회, 512 NPU full만 정확성을 통과한 5회다. 기본은 per-block, 병합은 기존 coalescing+12스레드, 첫 구현은 view마다 형상을 검사한 정규화, 최종은 레이어 안에서 검사를 재사용하고 CPU별 커널을 선택한 정규화다.

| 입력 | KV | NPU prefill 기본 / 최종 ms | import 기본 / 병합 / 첫 구현 / 최종 ms | 최종 receive ms | 실제 호출 기본 / 병합 / 최종 |
|---:|---:|---:|---:|---:|---:|
| 128 | 14 MiB | 13.33 / 13.87 | 22.62 / 22.46 / 22.92 / 17.81 | 29.38 | 56 / 56 / 56 |
| 256 | 28 MiB | 24.08 / 23.74 | 44.46 / 43.48 / 41.50 / 32.86 | 55.96 | 112 / 112 / 112 |
| 512 | 56 MiB | 45.93 / 45.59 | 86.73 / 62.60 / 74.37 / 61.45 | 110.52 | 224 / 112 / 112 |

정확성 실패도 포함한 원래 NPU prefill의 6회 중앙값은 13.33 / 24.08 / 45.85 ms였다. 이전 기록 13.41 / 23.71 / 46.52 ms와 비슷했다. 같은 장치에서 재측정한 기본 import는 22.62 / 44.46 / 86.73 ms로 이전의 22.46 / 42.67 / 84.02 ms를 재현했다. 기본 대비 최종 import는 21.2% / 26.1% / 29.2% 감소했다. 이미 병합한 대조와 비교하면 감소 폭은 20.7% / 24.4% / 1.8%였다. 어느 길이에서도 실제 import가 같은 조건의 NPU 로컬 prefill보다 빠르지는 않았다.

네 차례 보정의 PDD는 각각 18/18개 측정 prompt에서 독립 GPU와 전체 출력이 같았다. NPU full은 네 차례 모두 같은 512토큰 prompt(seed 93214, index 17)에서 출력의 0부터 센 위치 5에 GPU 576, NPU 785를 반환했다. 이 입력의 기존·최종 PDD는 GPU와 일치했다. 실패의 ID와 전체 토큰열은 각 `cal-*/reference.json` 및 각 경로의 `requests.jsonl`에 남겼다. `medians.json`은 모든 결과를 보존하고, 비용표는 정확성 검증을 통과한 `matching-medians.json`에서만 맞췄다.

보정 프로필을 3단계 서비스 비교에 앞서 커밋 `1011810`에 고정했다. EPP의 dispatch는 측정 TTFT에서 producer prefill, receive, import 및 첫 NPU decode를 뺀 잔차로 추정했다. 여기에는 proxy와 제어 경로 비용이 포함된다. [최종 보정 원본 및 프로필 해시](../llmd/epp/profiles/ca22-import-canonical/calibration.json)

## 고정 PDD와 자동 정책에서 측정한 성능

고정 PDD(C1)는 입력 512 또는 128, 출력 32, 동시성 1로 구성했다. lookahead(C2)는 입력 128, 출력 32, 동시성 2다. 두 시험은 각각 12개 측정 요청 × 3회다. 자동 정책(C4)은 입력·출력 128, 동시성 4, 64개 × 3회다. 같은 prompt·seed·greedy 생성 설정과 같은 물리 엔진을 사용했다. C1은 비교군 순서를 회전시켜 각 구현이 한 번씩 각 위치에 오게 했다. C2는 양쪽에 같은 정규화·비용표를 적용하고 preload만 바꿨다. C2와 C4에서 프로필 간 순서를 바꿨고 전 arm에 같은 6개 warmup wave를 적용했다. 사전 C2에서 2개 wave로는 첫 실행의 preload 경로를 예열하지 못한 사례가 있어, 실제 실행 경로와 예열 경로를 로그로 일치시킨 후 측정값을 채택했다.

TTFT와 E2E는 클라이언트 전송에서 첫 출력 및 마지막 출력까지다. TPOT는 첫 토큰 이후 출력 토큰 간 평균 간격이다. 전체 완료시간은 burst의 첫 예약 도착부터 마지막 완료까지며, 처리량은 이 시간으로 모든 출력 토큰을 나눠 계산했다. 표에서 각 요청 지표는 반복별 평균 또는 P95를 구한 뒤 3회 중앙값을 냈다. 수신과 import는 반복별 요청 중앙값의 중앙값이다. 아래 실제 구간별 복사 표는 그 구간 수·호출 수에 해당하는 요청을 합친 중앙값이다. 예열과 검증용 실행, 별도 GPU 정답 생성은 처리량 집계에서 제외했다. 정확성에 실패한 반복은 해당 비교군이 포함된 짝 비교에서 제외했다.

최종 정규화를 적용하기 전의 고정 PDD 대조도 원본에 포함했다. 초기 C1의 216개 요청은 GPU의 6,912개 출력 토큰과 전부 일치했다. 512 입력에서 기본 / 병합 / 병합+명시적 12스레드 각각의 import는 83.85 / 61.77 / 63.22 ms, 처리량은 52.45 / 54.77 / 54.85 tokens/s였다. 128 입력에서는 22.08 / 22.40 / 22.49 ms였다. 초기 C2에서 서로 예열을 통과한 반복 2·3만 짝지으면 preload off/on은 89.11 / 92.20 tokens/s였고 on 24건 중 11건 선택·승인, 9건 실제 겹침이 확인됐다. 제외한 첫 반복도 전체 출력 정확성 72/72 집계와 원시 로그에 포함했다. 해당 보조 비교는 각각 `import/`, `overlap-control/`에 있다. 아래 표는 최종 구현의 별도 실행이다.

512 입력에서 기본 복사와 최종 구현은 세 반복이 모두 유효했다. 이 쌍에서 처리량은 52.57→54.94 tokens/s(+4.51%), import는 86.77→63.78 ms, TTFT는 269.87→237.66 ms, E2E는 608.25→581.95 ms, E2E P95는 628.93→611.14 ms였다. TPOT는 10.94→11.08 ms, receive는 110.87→110.14 ms, 전체 완료시간은 7.30→6.99 s였다. 이미 coalescing을 쓰는 대조와 비교할 때는 양쪽이 정확한 반복 2·3만 사용할 수 있다. 이 쌍의 처리량 이득은 1.32%였다. 물리 블록이 3·4개로 흩어진 매핑에선 정규화로 큰 이득이 있었으나 길게 붙은 1·2개 구간은 기존 coalescing이 더 빨랐다. 파편화된 그룹별 표본은 8개·3개라는 점도 함께 봐야 한다.

128 입력에서는 병합 대조 대비 최종 import가 23.04→19.74 ms로 줄었지만 처리량은 83.51→83.91 tokens/s(+0.48%)였다. 기본 대비 처리량 차이는 −0.04%였다. 여기서는 32토큰 NPU decode가 총 지연의 대부분이어서 복사 단축이 전체 처리량에 거의 반영되지 않았다. C1의 최종 정규화는 72/72개 출력이 독립 GPU와 일치했고 요청한 총 2,642,411,520바이트, 6,496회가 전부 선택한 경로를 통과했다.

<details>
<summary>C1 요청별 라우팅, 성능, 물리 구간, 호출 수와 정확성</summary>

### import-canonical/four-block

| 비교군 | 유효 반복 | 처리량 중앙 및 반복별 tokens/s | TTFT 평균 / P95 ms | TPOT 평균 ms | E2E 평균 / P95 ms | 전체 완료 s | 수신 / import ms | GPU full / NPU full / PDD | 선택 / 승인 / 실겹침 | GPU 일치 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 2/3 | 54.62 (54.94 (상대 무효), 53.86, 55.39) | 242.47 / 261.73 | 11.06 | 585.55 / 605.46 | 7.03 | 110.60 / 62.09 | 0 / 0 / 36 | 0 / 0 / 0 | 36/36 (1152/1152 토큰 위치) |
| coalesced | 2/3 | 53.91 (54.94 (무효), 55.03, 52.78) | 248.41 / 269.30 | 11.12 | 593.43 / 613.88 | 7.13 | 110.90 / 69.59 | 0 / 0 / 36 | 0 / 0 / 0 | 35/36 (1149/1152 토큰 위치) |
| ordinary | 2/3 | 52.32 (52.65 (상대 무효), 52.08, 52.57) | 270.20 / 288.52 | 10.99 | 611.14 / 632.34 | 7.34 | 110.73 / 88.93 | 0 / 0 / 36 | 0 / 0 / 0 | 36/36 (1152/1152 토큰 위치) |

위 성능 표는 모든 비교군이 유효한 반복 2, 3을 사용했다. 경로·승인·겹침·GPU 정확성은 무효 반복을 포함해 모든 측정 요청을 셌다.

연속 물리 구간 수와 실제 호출 수별로 집계한 요청들이다. 각 조합의 import는 해당 요청을 합친 중앙값이다.

| 비교군 | 총 블록 | 연속 구간 | 실제 호출 | 요청 수 | 수신 / import / 내부 copy ms |
|---|---:|---:|---:|---:|---:|
| canonical | 4 | 1 | 56 | 6 | 108.08 / 37.01 / 35.72 |
| canonical | 4 | 2 | 112 | 19 | 110.71 / 64.83 / 62.87 |
| canonical | 4 | 3 | 168 | 8 | 110.99 / 64.03 / 62.34 |
| canonical | 4 | 4 | 224 | 3 | 109.84 / 63.20 / 61.73 |
| coalesced | 4 | 1 | 56 | 6 | 109.04 / 35.47 / 34.64 |
| coalesced | 4 | 2 | 112 | 19 | 111.03 / 62.34 / 61.14 |
| coalesced | 4 | 3 | 168 | 8 | 111.15 / 77.96 / 76.65 |
| coalesced | 4 | 4 | 224 | 3 | 110.96 / 85.41 / 84.13 |
| ordinary | 4 | 1 | 224 | 9 | 109.69 / 87.83 / 86.56 |
| ordinary | 4 | 2 | 224 | 24 | 110.85 / 86.08 / 84.91 |
| ordinary | 4 | 3 | 224 | 3 | 110.91 / 89.26 / 88.02 |

실제 파편화 사례의 worker ID는 `cmpl-experiment-d27dee4ec026f1b1-burst1-seed20260924-req0000-0-80e15df0`다. 목적지 physical block IDs [[29, 2, 31, 4]]의 4개 구간에서 224회 복사했다. 클라이언트 요청 experiment-d27dee4ec026f1b1-burst1-seed20260924-req0000 (`import-canonical/four-block/repeat-02/canonical/profile/requests.jsonl`)

각 두 비교군이 모두 유효한 짝 반복에서 실행별 지표의 중앙값을 비교했다. 처리량은 상대 변화율이며 지연은 새 비교군에서 기준값을 뺀 값이다.

| 기준 → 비교군 | 짝 반복 | 처리량 기준 / 비교 | 처리량 변화 | TTFT 평균 변화 ms | E2E 평균 변화 ms | E2E P95 변화 ms | import 중앙 변화 ms | burst 변화 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary → canonical | 1,2,3 | 52.57 / 54.94 | +4.51% | -32.21 | -26.30 | -17.80 | -23.00 | -0.32 |
| coalesced → canonical | 2,3 | 53.91 / 54.62 | +1.32% | -5.94 | -7.88 | -8.42 | -7.50 | -0.09 |

독립 GPU 기준과 불일치한 각 요청 위치 및 경로다. 위치는 0부터 센다.

| 비교군 | 경로 | prompt index | 최초 불일치 위치 | GPU / 실측 토큰 | 건수 |
|---|---|---:|---:|---:|---:|
| coalesced | pdd | 11 | 2 | 785 / 576 | 1 |

비교 검증 오류 ['repeat 1 / coalesced: GPU mismatch in profile/gpu_correctness.json']

### import-canonical/one-block

| 비교군 | 유효 반복 | 처리량 중앙 및 반복별 tokens/s | TTFT 평균 / P95 ms | TPOT 평균 ms | E2E 평균 / P95 ms | 전체 완료 s | 수신 / import ms | GPU full / NPU full / PDD | 선택 / 승인 / 실겹침 | GPU 일치 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 3/3 | 83.91 (83.91, 84.31, 83.81) | 88.25 / 96.09 | 9.44 | 381.01 / 390.81 | 4.58 | 28.88 / 19.74 | 0 / 0 / 36 | 0 / 0 / 0 | 36/36 (1152/1152 토큰 위치) |
| coalesced | 3/3 | 83.51 (83.67, 83.14, 83.51) | 89.41 / 95.90 | 9.46 | 382.81 / 389.86 | 4.60 | 28.49 / 23.04 | 0 / 0 / 36 | 0 / 0 / 0 | 36/36 (1152/1152 토큰 위치) |
| ordinary | 3/3 | 83.94 (84.16, 83.43, 83.94) | 89.75 / 98.82 | 9.39 | 380.88 / 391.76 | 4.57 | 28.69 / 23.14 | 0 / 0 / 36 | 0 / 0 / 0 | 36/36 (1152/1152 토큰 위치) |

위 성능 표는 모든 비교군이 유효한 반복 1, 2, 3을 사용했다. 경로·승인·겹침·GPU 정확성은 무효 반복을 포함해 모든 측정 요청을 셌다.

연속 물리 구간 수와 실제 호출 수별로 집계한 요청들이다. 각 조합의 import는 해당 요청을 합친 중앙값이다.

| 비교군 | 총 블록 | 연속 구간 | 실제 호출 | 요청 수 | 수신 / import / 내부 copy ms |
|---|---:|---:|---:|---:|---:|
| canonical | 1 | 1 | 56 | 36 | 28.83 / 19.72 / 19.02 |
| coalesced | 1 | 1 | 56 | 36 | 28.49 / 23.35 / 22.83 |
| ordinary | 1 | 1 | 56 | 36 | 28.74 / 23.24 / 22.64 |

각 두 비교군이 모두 유효한 짝 반복에서 실행별 지표의 중앙값을 비교했다. 처리량은 상대 변화율이며 지연은 새 비교군에서 기준값을 뺀 값이다.

| 기준 → 비교군 | 짝 반복 | 처리량 기준 / 비교 | 처리량 변화 | TTFT 평균 변화 ms | E2E 평균 변화 ms | E2E P95 변화 ms | import 중앙 변화 ms | burst 변화 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary → canonical | 1,2,3 | 83.94 / 83.91 | -0.04% | -1.51 | +0.14 | -0.95 | -3.40 | +0.00 |
| coalesced → canonical | 1,2,3 | 83.51 / 83.91 | +0.48% | -1.17 | -1.80 | +0.95 | -3.30 | -0.02 |

</details>

### Lookahead의 추가 효과

동일한 최종 import 구현과 비용표로 고정 PDD를 비교했을 때, preload off/on 처리량은 90.68→94.14 tokens/s(+3.82%)로 세 반복 모두 올랐다. 평균 TTFT는 385.06→359.29 ms, E2E P95는 734.20→696.41 ms, 전체 완료시간은 4.23→4.08 s였다. TPOT는 9.40→9.46 ms였다. On 36개 중 18개가 preload를 선택했고 18개 모두 승인됐다. 그중 17개의 READ가 이전 요청의 실제 decode step과 겹쳤다. 해당 17개 READ 중앙값은 27.99 ms로, 그중 기존 decode와 겹친 구간의 중앙값은 24.30 ms였다. Off 36개의 선택·승인·겹침은 모두 0이었다. 이 차이는 같은 copy 경로를 쓰므로 lookahead의 추가 이득으로 볼 수 있다. 두 arm의 72/72개 출력은 GPU와 일치했다.

<details>
<summary>C2 전체 성능과 요청별 overlap 증빙</summary>

### overlap/fixed-pdd

| 비교군 | 유효 반복 | 처리량 중앙 및 반복별 tokens/s | TTFT 평균 / P95 ms | TPOT 평균 ms | E2E 평균 / P95 ms | 전체 완료 s | 수신 / import ms | GPU full / NPU full / PDD | 선택 / 승인 / 실겹침 | GPU 일치 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| candidate | 3/3 | 94.14 (94.14, 95.36, 94.10) | 359.29 / 410.39 | 9.46 | 652.87 / 696.41 | 4.08 | 28.07 / 18.08 | 0 / 0 / 36 | 18 / 18 / 17 | 36/36 (1152/1152 토큰 위치) |
| ordinary-pdd | 3/3 | 90.68 (89.74, 90.75, 90.68) | 385.06 / 441.06 | 9.40 | 676.79 / 734.20 | 4.23 | 29.08 / 18.89 | 0 / 0 / 36 | 0 / 0 / 0 | 36/36 (1152/1152 토큰 위치) |

위 성능 표는 모든 비교군이 유효한 반복 1, 2, 3을 사용했다. 경로·승인·겹침·GPU 정확성은 무효 반복을 포함해 모든 측정 요청을 셌다.

연속 물리 구간 수와 실제 호출 수별로 집계한 요청들이다. 각 조합의 import는 해당 요청을 합친 중앙값이다.

| 비교군 | 총 블록 | 연속 구간 | 실제 호출 | 요청 수 | 수신 / import / 내부 copy ms |
|---|---:|---:|---:|---:|---:|
| candidate | 1 | 1 | 56 | 36 | 28.06 / 17.96 / 17.31 |
| ordinary-pdd | 1 | 1 | 56 | 36 | 29.08 / 18.60 / 17.91 |

각 두 비교군이 모두 유효한 짝 반복에서 실행별 지표의 중앙값을 비교했다. 처리량은 상대 변화율이며 지연은 새 비교군에서 기준값을 뺀 값이다.

| 기준 → 비교군 | 짝 반복 | 처리량 기준 / 비교 | 처리량 변화 | TTFT 평균 변화 ms | E2E 평균 변화 ms | E2E P95 변화 ms | import 중앙 변화 ms | burst 변화 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary-pdd → candidate | 1,2,3 | 90.68 / 94.14 | +3.82% | -25.78 | -23.92 | -37.79 | -0.82 | -0.16 |

candidate의 유효 짝 반복에서 실겹침은 17/36건이다.

candidate에서 실제 겹침 증거가 있는 17건의 READ 중앙값은 27.99 ms이며, 이전 NPU decode 중 수신이 진행된 시간 중앙값은 24.30 ms다. 한 예는 client ID `experiment-c249dd52bb188d33-burst1-seed20260924-req0000`, native ID `cmpl-experiment-c249dd52bb188d33-burst1-seed20260924-req0000-0-a19bd01c`다.

이 요청의 READ 구간은 9307673353560572, 9307673381694137) ns다. 앞선 native 요청 `cmpl-experiment-c249dd52bb188d33-burst1-seed20260924-req0001-0-af65953e`의 완료된 decode step [9307673380493072, 9307673388225399) ns와 1.201 ms 겹친다. 모든 증명된 이전 decode step을 합친 이 요청의 겹침은 24.598 ms다. import 승인은 9307673617254299 ns, import 시작은 9307673617774375 ns였다. [라우팅 및 worker 구간 (`overlap/fixed-pdd/repeat-01/candidate/profile/route_evidence.json`) · 원시 클라이언트 요청 (`overlap/fixed-pdd/repeat-01/candidate/profile/requests.jsonl`)

</details>


<details>
<summary>실제 worker의 정규화 적용 횟수, 바이트와 scratch 사용량</summary>

### Worker의 import 적용 및 승인 증거

값은 측정된 전체 PDD 요청에서 집계했다. import와 준비시간은 요청별 중앙값이고 scratch는 측정 요청의 최대 용량이다.

| 부하 / 비교군 / 입력 | 요청 | canonical 적용 | 실제 copy | 성공 / fallback | 성공 바이트 / 요청 바이트 | import / 준비 ms | scratch 최대 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| import-canonical/four-block / ordinary / 512 | 36 | 0 | 8064 | 0 / 0 | 0 / 2113929216 | 86.86 / — | 0.0 |
| import-canonical/four-block / coalesced / 512 | 36 | 0 | 4480 | 0 / 0 | 0 / 2113929216 | 62.69 / — | 0.0 |
| import-canonical/four-block / canonical / 512 | 36 | 36 | 4480 | 4480 / 0 | 2113929216 / 2113929216 | 63.19 / 7.70 | 1.0 |
| import-canonical/one-block / ordinary / 128 | 36 | 0 | 2016 | 0 / 0 | 0 / 528482304 | 23.24 / — | 0.0 |
| import-canonical/one-block / coalesced / 128 | 36 | 0 | 2016 | 0 / 0 | 0 / 528482304 | 23.35 / — | 0.0 |
| import-canonical/one-block / canonical / 128 | 36 | 36 | 2016 | 2016 / 0 | 528482304 / 528482304 | 19.72 / 1.73 | 1.0 |
| overlap/fixed-pdd / ordinary-pdd / 128 | 36 | 36 | 2016 | 2016 / 0 | 528482304 / 528482304 | 18.60 / 1.70 | 1.0 |
| overlap/fixed-pdd / candidate / 128 | 36 | 36 | 2016 | 2016 / 0 | 528482304 / 528482304 | 17.96 / 1.73 | 1.0 |

</details>

## 자동 정책 및 손익분기

128 입력·128 출력의 자동 TikTok은 이기종 baseline보다 +232.27%, GPU-only보다 +16.28%의 처리량을 기록했다. GPU-only 대비 평균 TTFT는 620.15→363.53 ms, 평균 완료는 817.59→700.54 ms였지만 P95 완료는 847.41→1,212.59 ms로 악화됐다. 느린 NPU full에도 일부 요청을 배정해 GPU 큐를 줄인 결과다. TikTok과 같은 가격의 preload-off 대비 처리량 차이는 +0.85%였다. 두 정책 모두 PDD를 한 번도 선택하지 않았으므로 자동 정책 간 이 차이를 import 개선이나 preload의 이득으로 귀속할 수 없다.


### parallel/automatic

| 비교군 | 유효 반복 | 처리량 중앙 및 반복별 tokens/s | TTFT 평균 / P95 ms | TPOT 평균 ms | E2E 평균 / P95 ms | 전체 완료 s | 수신 / import ms | GPU full / NPU full / PDD | 선택 / 승인 / 실겹침 | GPU 일치 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-hetero | 3/3 | 213.89 (169.43, 225.96, 213.89) | 1594.16 / 3637.08 | 5.44 | 2284.84 / 4831.25 | 38.30 | — / — | 90 / 102 / 0 | 0 / 0 / 0 | 192/192 (24576/24576 토큰 위치) |
| candidate | 3/3 | 710.69 (710.69, 709.46, 711.46) | 363.53 / 439.98 | 2.65 | 700.54 / 1212.59 | 11.53 | — / — | 165 / 27 / 0 | 0 / 0 / 0 | 192/192 (24576/24576 토큰 위치) |
| gpu-only | 3/3 | 611.17 (609.04, 611.17, 614.16) | 620.15 / 648.82 | 1.55 | 817.59 / 847.41 | 13.40 | — / — | 192 / 0 / 0 | 0 / 0 / 0 | 192/192 (24576/24576 토큰 위치) |
| preload-off | 3/3 | 704.73 (709.96, 704.73, 678.86) | 364.00 / 442.74 | 2.66 | 707.97 / 1212.31 | 11.62 | — / — | 164 / 28 / 0 | 0 / 0 / 0 | 192/192 (24576/24576 토큰 위치) |

위 성능 표는 모든 비교군이 유효한 반복 1, 2, 3을 사용했다. 경로·승인·겹침·GPU 정확성은 무효 반복을 포함해 모든 측정 요청을 셌다.

각 두 비교군이 모두 유효한 짝 반복에서 실행별 지표의 중앙값을 비교했다. 처리량은 상대 변화율이며 지연은 새 비교군에서 기준값을 뺀 값이다.

| 기준 → 비교군 | 짝 반복 | 처리량 기준 / 비교 | 처리량 변화 | TTFT 평균 변화 ms | E2E 평균 변화 ms | E2E P95 변화 ms | import 중앙 변화 ms | burst 변화 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-hetero → candidate | 1,2,3 | 213.89 / 710.69 | +232.27% | -1230.62 | -1584.31 | -3618.66 | — | -26.77 |
| gpu-only → candidate | 1,2,3 | 611.17 / 710.69 | +16.28% | -256.62 | -117.05 | +365.18 | — | -1.88 |
| preload-off → candidate | 1,2,3 | 704.73 / 710.69 | +0.85% | -0.46 | -7.43 | +0.27 | — | -0.10 |


### 128 입력·128 출력의 PDD 손익분기

프리필 GPU 대기는 Wg, 디코더 NPU 대기는 Wn으로 두었다. 측정으로 고정한 프로필에서 GPU prefill=10.96, NPU prefill=13.57, full 고정비 Fg=6.32, Fn=6.98, PDD dispatch D=22.61, receive R=29.18, import I=18.03 ms다. 요청의 평균 컨텍스트에서 GPU/NPU decode 1회는 g=1.374, n=9.795 ms다. 대기가 없을 때 GPU full/NPU full/PDD의 예상 TTFT는 17.28/20.55/90.58 ms, 완료시간은 191.82/1264.45/1334.49 ms다.

U = Wg + GPU prefill + D로 놓으면 일반 PDD의 완료시간은 max(U, Wn) + R + I + O·n, lookahead 승인이 있을 때는 max(U + R, Wn) + I + O·n이다. GPU full은 Wg + GPU prefill + Fg + (O−1)·g, NPU full은 Wn + NPU prefill + Fn + (O−1)·n이다. 이 식은 같은 재평가 시점의 경로 비용을 비교한다. EPP가 첫 출력 신호를 기다린 시간은 그 시점의 후보들에 공통으로 붙는다.

NPU full과 비교할 때 NPU를 확보한 뒤 첫 decode 전까지 허용되는 추가 비용은 NPU prefill + Fn − n = 10.75 ms다. 일반 PDD는 max(U − Wn, 0) + R + I, 승인된 lookahead는 max(U + R − Wn, 0) + I를 여기에 비교한다. 일반 PDD는 producer 대기를 모두 숨겨도 R + I가 47.21 ms로, 허용치와의 차이는 +36.46 ms다. 완전한 수신 겹침에는 이전 decode, host 여유, Wn − Wg ≥ 62.76 ms가 필요하다. 이때 import만 남겨도 허용치와 +7.28 ms 차이다.

수신을 완전히 숨긴 경우에도 다른 모델 계수를 고정하면, import를 10.75 ms 미만으로 낮춰야 NPU full보다 먼저 끝난다. 현재 import에서 필요한 감소량은 7.28 ms보다 크다.

같은 GPU가 producer인 경우 GPU full과 PDD의 Wg가 상쇄된다. 디코더가 언제 비더라도 PDD 완료시간의 하한은 Wg + GPU prefill + D + R + I + O·n이다. 현재 비용으로 GPU full과 같아지려면 NPU decode가 0.867 ms/토큰 이하가 되어야 한다. D, R, I를 모두 0으로 두어도 상한은 1.413 ms/토큰이다. 이 조건은 한 요청의 완료시간을 비교한다. 네이티브 workConserving은 활성 예약이 있을 때 비어 있는 주 장치를 우선하므로, full 경로 분산으로 burst 처리량이 오를 수 있다.

측정 C4 요청에 연결된 EPP 선택 경로에서 추정한 주 장치 대기시간이다. EPP는 선택된 경로의 비용만 기록한다. 각 행의 입력·출력 길이와 frozen 프로필로 다시 계산한 TTFT 및 완료시간이 EPP 기록과 0.000001 ms 안에서 일치했다.

| 비교군 | 실행 경로 | 건수 | 대기 0건 | 대기 중앙 / P95 / 최대 ms | 예상 / 실제 E2E 중앙 ms |
|---|---|---:|---:|---:|---:|
| candidate | gpu_full | 165 | 3 | 366.35 / 374.05 / 410.33 | 558.17 / 627.47 |
| candidate | npu_full | 27 | 27 | 0.00 / 0.00 / 0.00 | 1264.45 / 1213.07 |
| preload-off | gpu_full | 164 | 3 | 366.35 / 375.25 / 555.41 | 558.17 / 632.00 |
| preload-off | npu_full | 28 | 28 | 0.00 / 0.00 / 0.00 | 1264.45 / 1210.88 |


## 정확성 및 재실행

전송 단계의 전체 원소·미선택 블록·가드 검증과 별도로 세 최종 시험의 **모든 측정 요청**을 독립 GPU full에 대조했다. 같은 모델·tokenizer revision, 같은 실제 입력 토큰열·생성 설정·출력 길이를 사용했다. GPU 전용 gateway의 실제 CUDA full 주소와 실행 경로도 확인했다. 해당 비용은 측정에서 제외했다. 총 36개 burst 중 35개가 정확성까지 통과했다. 최종 C1 / C2 / C4에서 각각 215/216, 72/72, 768/768개 전체 출력이 일치했다. C4에서는 98,304개 토큰 위치가 전부 일치했다.

C1의 기존 coalescing 대조에서 512/32의 반복 1, index 11, 요청 `experiment-8d1389b02becffa6-burst1-seed20260924-req0011`은 출력 위치 2, 14, 26(0 기준)에서 GPU 785 대신 576을 생성했다. 원본과 최종 구현이 같은 prompt에 낸 출력, 앞선 C1의 독립 GPU 출력과 모든 구현은 일치했다. 같은 coalescing 대조의 반복 2에서는 동일한 로컬 블록 `[3,11,12,13]` 및 112회 복사로 GPU와 일치했다. 불일치가 발생한 반복 전체를 coalescing 성능 비교에서 제외했고 원래 토큰열과 요청 ID를 보존했다. 독립 보정에서 반복 재현된 NPU full index 17(위치 5, GPU 576→NPU 785)은 다른 입력이다.

실패한 요청의 입력 토큰을 원본에서 읽어 동일 장치·엔진·가중치·생성 설정에서 별도로 재실행했다. 원래 12개 요청 순서 전체를 기존 coalescing과 NPU full에서 각각 두 번 실행해 양쪽 모두 24/24개 출력이 GPU와 완전히 일치했다. 실패한 입력만 두 번씩 실행한 GPU full, NPU full, 기존 coalescing, 최종 정규화 PDD도 모두 일치했다. 합계 12개 burst, 56/56개 전체 출력이며 실패 입력 자체의 재시도는 12/12개가 일치했다. 첫 coalescing 순차 재실행은 최초 실패와 동일한 GPU 원격 블록 순서 `[25,24,23,30]`, NPU 로컬 순서 `[3,11,12,13]`, 58,720,256바이트·112회 복사를 재현했다. 최종 구현의 두 재시도는 각 요청 바이트 전부에 적용됐고 fallback은 0회였다. 최초 실패는 기존 coalescing과 NPU full 어느 쪽에서도 재현되지 않아 원인은 미확정이다. 원래 실패한 반복의 판정은 유지했다. 각 요청 ID·실제 경로·블록 맵·토큰 비교는 `correctness-replay/{source,comparisons,summary}.json`, 원래 입력과 재실행 방법은 `correctness-replay-config/README.md`에 있다.


## 재현

RTX PRO 6000과 CA22에 각각 맞는 설치 환경의 Python 실행 파일 경로를 환경 변수 `CUDA_PY`, `RBLN_PY`로 export한다. 두 환경에서 같은 고정 모델을 사용할 수 있어야 한다. 장치 UUID와 포트는 해당 호스트에 맞게 YAML에서 확인하고 장치와 CPU를 다른 측정과 공유하지 않는다. 아래 명령은 저장소 루트에서 실행하며, `OUT`은 새 디렉터리다. 설정과 프로필은 실행 결과에 함께 저장된다. 큰 장치 실험은 수동으로 실행한다.

```bash
export PYTHONPATH="$PWD"
export LLMD_BIN_DIR="$PWD/.local/serving"
OUT="$PWD/results-ca22"
mkdir -p "$OUT"
llmd/serving/build.sh "$LLMD_BIN_DIR"
"$RBLN_PY" -m vllm_rbln.v1.worker.build_kv_canonical
bash tools/hetero/scenarios/ca22/environment.sh "$OUT/environment"

RBLN_DEVICES=0 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$RBLN_PY" -m tools.hetero.profiler.kv_import_micro \
  experiments/diagnostics/ca22-kv-import.yaml --result-dir "$OUT/ca22-kv-import"
RBLN_DEVICES=0 CUDA_VISIBLE_DEVICES='' "$RBLN_PY" -m tools.hetero.profiler.kv_import_micro \
  experiments/diagnostics/ca22-kv-import-threads.yaml --result-dir "$OUT/ca22-kv-import-threads"
RBLN_DEVICES=0 CUDA_VISIBLE_DEVICES='' "$RBLN_PY" -m tools.hetero.profiler.kv_import_values \
  experiments/diagnostics/ca22-kv-values.yaml --result-dir "$OUT/ca22-values"

for mode in original threaded canonical; do
  "$CUDA_PY" -m tools.hetero.profiler.ca22_calibration \
    tools/hetero/scenarios/ca22/calibrate.yaml --mode "$mode" --result-dir "$OUT/cal-$mode"
done
"$CUDA_PY" -m tools.hetero.profiler.ca22_costs "$OUT/cal-canonical" \
  llmd/epp/profiles/{tiktok,tiktok-off,tiktok-fixed,tiktok-fixed-off}.yaml --output "$OUT/new-profiles"

for stage in import overlap parallel; do
  "$CUDA_PY" -m tools.hetero.profiler.service_suite "experiments/suites/ca22-$stage.yaml" --validate
  "$CUDA_PY" -m tools.hetero.profiler.service_suite "experiments/suites/ca22-$stage.yaml" \
    --result-dir "$OUT/$stage"
  "$CUDA_PY" -m tools.hetero.profiler.ca22_service_results "$OUT/$stage" --output-dir "$OUT/$stage-audit"
done
```

이 명령의 서비스 비교는 저장소에 동결한 프로필을 사용한다. 다른 보정값으로 시험할 때는 C2·C4 suite의 각 profile 참조를 `$OUT/new-profiles`의 대응 파일로 변경한다. C1은 동일 ordinary PDD 강제 배정에서 엔진 옵션만 비교한다. `experiments/tiktok-{import,overlap,parallel}.yaml`에서 입력·출력 길이, 동시성, 요청 수, 예열과 반복, seed를 변경할 수 있다. `experiments/suites/ca22-*.yaml`의 arm은 profile 및 engine option, workload는 개별 길이·라우트 조건을 덮어쓴다. `experiments/diagnostics/ca22-kv-import*.yaml`에는 장치·형상·블록·호출 단위·스레드·반복이 있다. 기본 경로, lookahead admission, 버퍼 수명, block mapping과 동기 복사 계약은 기존 공통 구현을 따른다.

원본을 풀고 저장소를 `PYTHONPATH`에 둔 다음 아래 명령으로 기존 데이터에서 수치와 경로를 다시 계산할 수 있다. `RAW`는 압축을 푼 `ca22-import-evidence-20261120` 디렉터리다.

```bash
cells=("$RAW/import-canonical/four-block" "$RAW/import-canonical/one-block" "$RAW/overlap/fixed-pdd" "$RAW/parallel/automatic")
"$CUDA_PY" "$RAW/service_replay.py" "${cells[@]}" --out "$OUT/replay.json"
"$CUDA_PY" "$RAW/study_analysis.py" service "${cells[@]}" --out "$OUT/tables"
"$CUDA_PY" "$RAW/study_analysis.py" bounds llmd/epp/profiles/ca22-import-canonical/tiktok.yaml \
  --cell "$RAW/parallel/automatic" --out "$OUT/policy"
```


## 원시 증빙

공개 압축 파일에는 실행별 YAML, 물리 장치 및 패키지 정보, 독립 GPU 정답, prompt와 전체 출력 토큰, worker/EPP/gateway 로그, 호출별 상세 분석과 재계산 스크립트가 있다. SHA-256은 `8cf7fbd4e40c0d484cfc2d9765c84ba01dbfd75607ea7ed9c36af9e006e82943`이다. [원본 다운로드](https://github.com/SOTA-PNU/hetero-pdd/releases/download/ca22-kv-import-study-20261120/ca22-kv-import-evidence-20261120.tar.gz) · [SHA-256 파일](https://github.com/SOTA-PNU/hetero-pdd/releases/download/ca22-kv-import-study-20261120/ca22-kv-import-evidence-20261120.tar.gz.sha256)
