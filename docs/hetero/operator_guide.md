# 운영 안내서

문서 갱신: 2026-11-12. 아래 시험·자료 확인일은 2026-11-10이다.

기준: 2026-11-10. 운영 조치는 아래 코드·실측 사례에 근거하며 자동 적용하지 않는다.
명령 전 운영 환경 파일을 source하고 실제 경로 값은 저장소 밖에서 관리한다.
관련 문서: [설치](install_guide.md), [사용](user_guide.md).

## 측정 전 확인

| 확인 | 절차·근거 |
|---|---|
| 장치 점유 | nvidia-smi·rbln-smi로 메모리·사용 프로세스 확인, 다른 작업 담당자와 사용 시간 조정; orchestrate.verify_idle의 자동 검사도 통과해야 함 |
| 저장소 | 실측 전용 worktree에서 git status --porcelain이 비었는지 확인; 커밋과 태그를 run 기록에 보존 |
| 환경 | CUDA_PY·RBLN_PY와 모델, 고정 도구 버전 확인; 사용자 환경 파일의 내용을 로그로 출력하지 않음 |
| 결과 위치 | 실행마다 저장소 밖 새 빈 RUN_DIR. run.py가 service/를 만들기 전에 사용자가 미리 만들지 않음 |
| 포트 | 다른 서비스 점유 확인. 현재 all-smi 기본 PDD_ALLSMI_PORT=19090; 필요하면 환경 파일에서 변경 |
| 종료 | run.json의 *_stop 단계 및 processes.json·장치 프로세스 확인; 관련 없는 프로세스에 일괄 kill하지 않음 |

실측 worktree는 `git worktree add --detach "$MEASURE_TREE" "$MEASURE_COMMIT"` 형태로 만든다.
커밋 전에 작업 중인 디렉터리를 그대로 측정 경로로 쓰지 않는다.
문서 수정도 추적 변경이므로 clean 검사에 걸릴 수 있다. 원인을 숨기려고 검사 코드를 끄지 않는다.

## S7 및 KV 레이아웃 지원 조건

GPU producer가 `VLLM_KV_CACHE_LAYOUT=HND`이면 `VLLM_RBLN_PDD_LAYOUT_REORDER=0`을 쓴다.
reorder 1은 producer가 실제로 NHD 배치일 때만 쓴다.
producer의 commands.sh 설정뿐 아니라 prefill 로그의 실제 KV cache layout도 확인한다.
S7의 HND+reorder 1은 **설정 호환성 문제 재현용**이며 정상 성능 최적화 조건이 아니다.
NHD 조건의 출력 정합성은 사용 전 별도 검증하며, 향후 핸드셰이크 상대 layout 기반 자동 선택을 제안한다.
근거: sweep4 S7 prefill의 HND·FLASH_ATTN 로그와 출력 토큰 대조.

## 캐시 관리

직접 실행은 --cache-dir, sweep은 --cache-root로 외부 캐시를 지정한다.
sweep 키는 모델 키·구성·서비스 인자·layout-reorder·torch profiler 사용 여부로 만든다.
deep과 비deep은 구분된다. H7에서 같은 조건 반복의 기동 단축을 확인했지만,
SDK·모델 스냅샷·코드 변경 전반의 재사용 안전성은 미확인(H7)이다.
모델 키가 같아도 실제 가중치나 SDK가 바뀌면 운영자가 새 캐시 루트를 선택한다.
실행 중 캐시를 삭제하지 않으며, 재현에 쓴 캐시 조건과 새로 컴파일했는지를 기록한다.

## all-smi 접근·디스크·권한

PC4에서 all-smi는 0.0.0.0에 바인드했다. 수집 URL이 loopback이라고 외부 접근까지 차단된 것은 아니다.
운영 방화벽에서 해당 포트의 외부 수신을 차단하고 로컬 수집만 허용한다.
방화벽 정책·인터페이스별 실제 차단은 운영자가 검증해야 하며 이번 문서 작업에서 변경하지 않았다.
드라이버·서버 포트도 외부 공개를 전제로 설계하지 않았다.

결과·환경 파일은 운영 계정이 소유하게 하고 신규 생성 전 umask 077 등 조직 권한 정책을 적용한다.
공유 시에는 원본의 사용자·호스트·모델 경로를 점검하고 민감한 값을 치환한 뒤 공유한다.
trace는 매우 클 수 있으므로 시작 전 df와 결과 폴더 du로 여유 공간을 확인한다.
H6b NPU decode는 다수 PB를 만들었고 무결성 문제도 있었다. 공간 부족을 성공으로 처리하지 않는다.
보존 기간은 조직 정책으로 정한다(본 프로젝트가 임의의 삭제 기간을 강제하지 않음).
summary·CSV·meta·진단 기록을 보존한 뒤 승인된 원본만 삭제하며, 배포에는 원본 trace를 넣지 않는다.

## 자주 발생한 오류

| 증상 | 확인·대처 | 실측 근거 |
|---|---|---|
| ON 반복에서 Address already in use | SO_REUSEADDR bind 확인과 최대 30초 대기 적용됨. 실제 다른 listener면 점유자·포트를 확인. 종료는 SIGTERM 후 wait, 초과 시 kill/wait | per001_r1b ABORTED; monitors.free_port·stop |
| dirty 작업 트리로 기동 거부 | git diff·untracked 확인, 검토 후 커밋하거나 깨끗한 worktree 사용. 원래 작업을 강제로 삭제하지 않음 | per001_r2, orchestrate.verify_environment |
| B0 NPU batch 8 초기화 실패 | vLLM의 batch≥8 Triton 샘플러가 RBLN 미구현 num_compute_units 호출. 검증된 1/2/4 조건 사용. PDD decode batch 8 성공과 구분 | sweep1 s3a_b8_b0 npu0.log, S3a 메모 |
| 온라인 NPU PB 없음 | 온라인 AsyncLLM은 MP 경로. MP=0 요구하는 RBLN Profiler는 오프라인 LLM API로 실행 | H6_fix, opprofile.configure |
| NPU decode PB 종료<시작·미종료 slice | not_collected 유지. 64토큰 재측정도 정상 집계 불가. SDK 진단 자료 제공 후 재측정 필요 | H6b 메모·진단 결과 |
| GPU gzip 잘림 | deep stop_profile 후 gzip 전체 검사와 크기 3초 안정, 최대120초 대기. 실패 경로와 run.json 원인 확인 | run.wait_for_traces, H6_fix |
| analysis_status INCOMPLETE | endpoint 표본수·before/after·ID 매칭 확인. 짧은 H0는 표본 1개로 최소2개 미달 | analysis.validate_measurement, h0_trace |
| T03·R09b 미수집 | 값 0으로 채우지 않음. copy_ 동기성·device 시간 의미 확인 필요 | H4, transfer.py |

비정상 종료 후 재실행은 새 결과 위치를 쓴다. sweep의 성공 건너뛰기·실패 접미사 기능을 활용한다.
오류를 보고할 때 run ID·커밋·설정·실패 단계·관련 로그만 공유하고 환경 파일·실제 경로·비밀 값은 제외한다.
