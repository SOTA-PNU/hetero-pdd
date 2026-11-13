# 설치 안내서

문서 갱신: 2026-11-11. 아래 시험·자료 확인일은 2026-11-10이다.

명령은 저장소 루트 기준이며 대문자 변수의 실제 값은 운영자가 저장소 밖 환경 파일에 설정한다.

## 전제 환경

| 항목 | 준비 사항·근거 |
|---|---|
| OS·Python | Linux x86_64, 번들 python-version과 같은 Python 3.10; venv 지원 |
| 장치 | 실측은 NVIDIA RTX PRO 6000 Blackwell 1장, RBLN-CA22 2장. 다른 장비는 별도 검증 필요 |
| 드라이버·SDK | nvidia-smi, rbln-smi가 정상 동작하는 기존 환경; 번들이 드라이버를 설치하지 않음 |
| CUDA·RBLN venv | 호환 vLLM 0.22.0 및 해당 장치 의존성 사전 설치. CUDA_PY·RBLN_PY 지정 |
| 모델 | 로컬 모델·토크나이저·config·가중치 준비. S1은 Qwen3-0.6B FP16 |
| 시험 의존성 | 시스템 Python의 prometheus_client; PyYAML 없으면 YAML 시험 skip. 도구 venv를 엔진 venv로 사용하지 않음 |
| 공간·권한 | clone·모델·번들·도구·결과·캐시를 구분하고 설치/결과를 저장소 밖에 둠 |

정확한 드라이버·SDK 호환 조합은 수령 환경의 공급자 설치 지침과 서비스 config.json을 확인한다.
본 문서는 모든 드라이버 버전의 호환성을 보증하지 않는다.

## 1. 소스와 번들 준비

팀 저장소를 clone하고 검토할 커밋을 checkout한다. 인터넷 연결 환경에서 번들을 만든다.
결과와 설치 위치는 저장소 밖이어야 한다.

```bash
bash tools/hetero/third_party/make_offline_bundle.sh --out "$OFFLINE_BUNDLE"
```

## 2. 오프라인 도구 설치

TOOLS_VENV·TOOLS_DIR은 저장소 밖 새 위치다. 원래 CUDA·RBLN venv를 덮어쓰지 않는다.

```bash
bash "$CLONE_DIR/tools/hetero/third_party/install_offline.sh"   --bundle "$OFFLINE_BUNDLE" --venv "$TOOLS_VENV" --tools-dir "$TOOLS_DIR"
```

스크립트는 SHA256SUMS·versions.env·산출물 해시와 Python 버전을 검사한 뒤 venv를 만든다.
pip·setuptools를 로컬 wheel로 먼저 갱신하고 고정 requirements를 no-index로 설치한다.
all-smi·Collector·trace_processor_shell을 배치하고 바이너리 버전을 확인한다.
별도 시험 의존성·드라이버·모델·엔진은 이 번들 설치 범위가 아니다.

| 고정 패키지 | 버전 |
|---|---|
| perfetto / matplotlib | 0.58.2 / 3.10.9 |
| Pillow / fonttools | 12.3.0 / 4.65.0 |
| pip / setuptools | 26.2.2 / 84.2.1 |


## 3. 환경 파일

설치기가 출력한 export 값을 저장소 밖 신뢰할 수 있는 환경 파일에 반영한다.
경로 값은 문서·git에 넣지 않는다. 아래는 실제 경로 대신 자리표시자를 쓴 구성 예다.

```bash
export CUDA_PY='<CUDA venv Python>'
export RBLN_PY='<RBLN venv Python>'
export PDD_TOOLS_PY='<도구 venv Python>'
export ALLSMI_BIN='<설치 위치>/all-smi'
export OTELCOL_BIN='<설치 위치>/otelcol-contrib'
export PERFETTO_TP_BIN='<설치 위치>/trace_processor_shell'
export QWEN06_PATH='<로컬 Qwen3-0.6B 모델>'
```

실행 셸에서 `source "$HETERO_ENV_FILE"`로 읽는다. 환경 파일을 임의 배포물에서 받아 실행하지 않는다.
이론 대역폭 변수는 확인된 장비 규격만 사용한다. 미설정이면 관련 비율은 not_collected다.

## 4. 설치 확인과 S1

```bash
cd "$CLONE_DIR"
git status --porcelain
"$PDD_TOOLS_PY" -c 'import matplotlib; from perfetto.trace_processor import TraceProcessor'
"$CUDA_PY" -c 'from tools.hetero.profiler.tools import check_tools; print(check_tools("deep"))'
python3 -m unittest discover -s tests/hetero
python3 -m unittest discover -s tests/pdd
"$CUDA_PY" -m tools.hetero.profiler.sweep   --scenarios "$S1_JSON" --sweep-dir "$S1_RESULTS"   --model "qwen3-0.6b=$QWEN06_PATH"
```

시작 전 장치 점유를 확인하고 시나리오 설정을 검토한다.
결과의 두 상태가 PASS인지, results.csv·comparison.csv가 생겼는지와 서비스 정지를 확인한다.
