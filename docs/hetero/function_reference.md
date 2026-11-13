# 함수·CLI 참조

문서 갱신: 2026-11-11. 아래 시험·자료 확인일은 2026-11-10이다.

기준: 2026-11-10, 커밋 b824255. 경로는 tools/hetero/ 기준이다.
모듈 docstring과 실제 선언·argparse를 대조했다. 밑줄로 시작하지 않는 모듈 함수·클래스 및 공개 메서드를 나열한다.
이 목록이 외부 호환 API 보증을 뜻하지는 않는다. 분석 보조 함수는 CLI에서 직접 호출하지 않는다.
함수별 docstring이 없는 경우 “없음”으로 표시하고 의미를 추측해 채우지 않았다.
기본값·서명은 코드 그대로이며 사용 예는 [사용 안내](user_guide.md)에 있다.

## 모듈별 함수

### finalize_results.py

결과표·쌍 비교·데이터셋 내보내기.

| 선언 | docstring(있으면 원문) |
|---|---|
| `kv_size(config, work)` | 없음 |
| `result_row(root, scenario, attempt, work)` | 없음 |
| `output_tokens(folder, work)` | 없음 |
| `comparisons(records)` | 없음 |
| `datasets(records)` | 없음 |
| `finalize(root, opprofile_dir=None)` | 없음 |
| `main(argv=None)` | 없음 |

### loadgen.py

정확한 토큰 입력·SSE 요청과 원자료 저장.

| 선언 | docstring(있으면 원문) |
|---|---|
| `Slot(index: int, target: str, base_url: str)` | 없음 |
| `make_tokenizer(path)` | 없음 |
| `clock_anchor()` | 없음 |
| `request_succeeded(row)` | 없음 |
| `make_prompt(tokenizer, length: int, request_index: int, seed: int)` | 없음 |
| `parse_sse_line(raw_line: bytes)` | 없음 |
| `issue_request(*, slot: Slot, request_index: int, prompt_ids: list[int], scheduled_arrival: float, model: str, output_tokens: int, seed: int, timeout: float, workload: str)` | 없음 |
| `parse_slots(args: argparse.Namespace)` | 없음 |
| `main(argv=None, *, tokenizer=None)` | 없음 |

### orchestrate.py

서비스 환경·입력 검사와 역할별 프로세스 기동/정지.

| 선언 | docstring(있으면 원문) |
|---|---|
| `python_path(value, variable: str)` | 없음 |
| `validate_role_options(args)` | 없음 |
| `apply_role_options(args, role, env, delta, argv)` | 없음 |
| `run_checked(argv: list[str], *, text: bool=True)` | 없음 |
| `verify_environment(repo: Path, cuda_py: Path, rbln_py: Path, expected_compiler=None)` | 없음 |
| `verify_idle()` | 없음 |
| `reserve_ports(count: int)` | 없음 |
| `proc_starttime(pid: int)` | 없음 |
| `wait_http(url: str, process: subprocess.Popen[bytes], timeout: float)` | 없음 |
| `clean_base_env()` | 없음 |
| `common_server_args(args: argparse.Namespace, port: int)` | 없음 |
| `write_commands(path: Path, commands: list[tuple[str, dict[str, str], list[str]]])` | Preserve the original audit script; process execution always uses argv. |
| `launch(run_dir: Path, role: str, argv: list[str], env: dict[str, str], env_delta: dict[str, str], health_url: str, timeout: float, records: list[dict[str, object]])` | 없음 |
| `stop_records(run_dir: Path)` | 없음 |
| `start(args: argparse.Namespace)` | 없음 |
| `parse_args(argv=None)` | 없음 |
| `main()` | 없음 |

### profiler/analysis.py

원자료·계측 검증, 클라이언트 계산·요청 연결·산출물.

| 선언 | docstring(있으면 원문) |
|---|---|
| `metric(value, unit='s', reason='missing_raw')` | 없음 |
| `validate_raw(folder, rows, config)` | 없음 |
| `client_metrics(rows)` | 없음 |
| `join_requests(root, rows, config, spans=None, events=None, window=(0, 0))` | 없음 |
| `validate_measurement(samples, boundaries, roles, window)` | 없음 |
| `write_csv(path, rows)` | 없음 |
| `analyze(service_run, workload)` | 없음 |

### profiler/common.py

JSONL·지표 객체·구간 합집합·시간가중 계산.

| 선언 | docstring(있으면 원문) |
|---|---|
| `live_rows(path)` | 없음 |
| `metric(value, unit, source, reason='missing_raw', status='measured', **extra)` | 없음 |
| `union(intervals)` | 없음 |
| `active_ratio(intervals, window)` | 없음 |
| `attribute_samples(points, intervals, linear=False)` | Integrate adjacent samples without extrapolation or bridging missing values. |
| `interval_segments(intervals, window)` | Partition a window at clipped interval endpoints. |

### profiler/expected.py

역할 설정·기대 ID·엔진 suffix·span 대기.

| 선언 | docstring(있으면 원문) |
|---|---|
| `service_config(run_dir)` | 없음 |
| `server_request_id(client_id)` | 없음 |
| `engine_request_base(req_id)` | 없음 |
| `expected_ids(rows, config)` | 없음 |
| `attribute(item, key)` | 없음 |
| `wait_for_spans(run_dir, expected, timeout=30)` | 없음 |

### profiler/log.py

UTC·run ID·component 로그.

| 선언 | docstring(있으면 원문) |
|---|---|
| `log_line(component, message)` | 없음 |

### profiler/monitors.py

외부 수집 프로세스 시작·준비·정지.

| 선언 | docstring(있으면 원문) |
|---|---|
| `free_port(port=0, timeout=0)` | 없음 |
| `port_ready(port)` | 없음 |
| `wait_for_layout(root, timeout=30)` | 없음 |
| `class Monitors` | 없음 |
| `Monitors.__init__(self, root)` | 없음 |
| `Monitors.start(self, name, argv, env=None, probe=None)` | 없음 |
| `Monitors.collector(self)` | 없음 |
| `Monitors.allsmi(self)` | 없음 |
| `Monitors.scrape(self, endpoints, out)` | 없음 |
| `Monitors.stop(self, name)` | 없음 |

### profiler/operator_stats.py

Chrome/PB 파싱·무결성·연산 및 유형 집계.

| 선언 | docstring(있으면 원문) |
|---|---|
| `aggregate_types(rows, device)` | 없음 |
| `chrome_rows(path)` | 없음 |
| `pb_rows(path, health=None)` | 없음 |
| `aggregate(rows, top=20)` | 없음 |
| `write_csv(path, rows, fields)` | 없음 |
| `main(argv=None)` | 없음 |

### profiler/opprofile.py

단일 장치 오프라인 LLM API 프로파일.

| 선언 | docstring(있으면 원문) |
|---|---|
| `configure(device, out)` | 없음 |
| `main(argv=None)` | 없음 |

### profiler/overhead.py

PER-001 반복·유효성·통계·CPU·보고.

| 선언 | docstring(있으면 원문) |
|---|---|
| `order(block)` | 없음 |
| `cpu_ticks(stat)` | 없음 |
| `cpu_snapshot(processes)` | 없음 |
| `cpu_usage(before, after, hz, cores)` | 없음 |
| `block_metrics(runs)` | 없음 |
| `decide(blocks)` | 없음 |
| `validate(folder, work, config, on)` | 없음 |
| `trial(folder, work, config, on)` | 없음 |
| `measure_block(number, measure, data)` | 없음 |
| `save(root, data)` | 없음 |
| `main(argv=None)` | 없음 |

### profiler/resources.py

장치 대응·시간축·상태별 자원·에너지.

| 선언 | docstring(있으면 원문) |
|---|---|
| `device_states(phases, window)` | 없음 |
| `classify_devices(text, links)` | 없음 |
| `read_samples(collected, links, window)` | 없음 |
| `mono(ns, anchors)` | 없음 |
| `phase_intervals(rows, ids, spans, events, anchors, trace)` | 없음 |
| `exclusive_phase_shares(parts, window)` | 없음 |
| `request_phase_share(rows, ids, spans, events, anchors, trace)` | 없음 |
| `phase_resources(series, phases, window, trace)` | 없음 |
| `energy_efficiency(power, window, tokens)` | 없음 |

### profiler/run.py

온라인 측정 세션·단계 기록·수집 및 정리.

| 선언 | docstring(있으면 원문) |
|---|---|
| `validate_workloads(workloads, configuration)` | 없음 |
| `flags(**values)` | 없음 |
| `role_options(configuration, mode, root, port)` | 없음 |
| `loadgen_args(work, config)` | 없음 |
| `command(root, name, argv)` | 없음 |
| `profile_control(out, role, url, action)` | 없음 |
| `wait_for_traces(root, roles, previous, timeout=120)` | 없음 |
| `recorded(root, data, name, optional=False)` | 없음 |
| `stage_functions(root, data)` | 없음 |
| `service_session(args, forwarded, service, data)` | 없음 |
| `workload_session(work, config, folder, mode, monitors, call, cli, endpoints)` | 없음 |
| `execute(args, forwarded, service, workloads, argv)` | 없음 |
| `parse_args(argv=None, *, single=False)` | 없음 |
| `main(argv=None)` | 없음 |

### profiler/scrape.py

HTTP 지표 수집·JSONL·경계 스냅샷.

| 선언 | docstring(있으면 원문) |
|---|---|
| `wait_until(condition, timeout, name)` | 없음 |
| `fetch(endpoint)` | 없음 |
| `append_rows(out_path, rows)` | 없음 |
| `boundary_snapshot(endpoints, out_path, label)` | 없음 |
| `main()` | 없음 |

### profiler/server.py

Prometheus 경계 차분·서버 단계 계산.

| 선언 | docstring(있으면 원문) |
|---|---|
| `samples(text)` | 없음 |
| `boundary_deltas(before, after)` | 없음 |
| `server_metrics(boundaries, collected, spans, trace, window, roles)` | 없음 |

### profiler/sweep.py

시나리오 쌍 검사·모델 매핑·캐시·재개.

| 선언 | docstring(있으면 원문) |
|---|---|
| `read_json(path)` | 없음 |
| `outside_repo(path)` | 없음 |
| `validate_pairs(scenarios)` | 없음 |
| `cache_key(scenario)` | 없음 |
| `successful(root, scenario, attempt)` | 없음 |
| `execute(spec, root, models, cache_root=None, only=())` | 없음 |
| `main(argv=None)` | 없음 |

### profiler/tools.py

고정 도구 검사·GPU/NPU 링크 원문 수집.

| 선언 | docstring(있으면 원문) |
|---|---|
| `check_tools(mode)` | 없음 |
| `capture_links(gpu_index, out_path)` | 없음 |

### profiler/transfer.py

전송·device 비율·추정 메모리 대역폭.

| 선언 | docstring(있으면 원문) |
|---|---|
| `device_time_ratio(steps, window)` | 없음 |
| `h2d_bandwidth(size, seconds)` | 없음 |
| `memory_bandwidth(config, steps, running, rows)` | 없음 |
| `transfer_metrics(config, rows, events, servers, links, window)` | 없음 |

### report_figures.py

결과 CSV·JSON의 보고서 그림 생성.

| 선언 | docstring(있으면 원문) |
|---|---|
| `read_csv(path)` | 없음 |
| `num(value)` | 없음 |
| `panel(ax, title, records, kind='bar')` | 없음 |
| `generate(root, operators, overhead, out)` | 없음 |

## CLI 인자

각 Python CLI의 --help로 현재 기본값을 확인할 수 있다. 표의 “필수” 이외 값은 선택이다.
run·overhead의 서비스 인자는 orchestrate start 파서로 전달한다.

| CLI | 필수 | 선택·기본값 |
|---|---|---|
| orchestrate.py start | --configuration baseline/pdd, --run-dir, --model-path, --model-name | 아래 서비스 인자 표 |
| orchestrate.py stop | --run-dir | 없음 |
| loadgen.py | --model, --tokenizer, --input-tokens, --output-tokens, --requests, --workload, --run-dir(--out-dir 별칭), --endpoint 또는 반복 --worker 중 하나 | --target pdd, --concurrency 1, --warmup-per-endpoint 1, --seed 20260924, --warmup-only, --timeout 900 |
| profiler/run.py | --configuration baseline/pdd, --mode none/monitor/trace/deep, --run-dir, --workloads | 서비스 인자 전달 |
| profiler/overhead.py | --configuration baseline/pdd, --run-dir, --workload | 서비스 인자 전달; mode=none 고정 |
| profiler/sweep.py | --scenarios, --sweep-dir | 반복 --model KEY=PATH, --cache-root, 반복 --only NAME |
| profiler/analysis.py | --run-dir | workload 폴더이며 service_run이 아님 |
| profiler/scrape.py | --out, 반복 --endpoint NAME=URL | --interval 1 |
| profiler/opprofile.py | --device gpu/npu, --phase prefill/decode, --model-path, --model-name, --out | --decode-tokens 256, --dtype bfloat16, --kv-cache-dtype auto, 아래 오프라인 설정 |
| profiler/operator_stats.py | --input, --out | 입력 폴더 meta.json으로 장치·phase 식별 |
| finalize_results.py | --sweep-dir | --opprofile-dir |
| report_figures.py | --sweep-dir, --opprofile-dir, --overhead, --out | overhead는 overhead.json 파일 |
| third_party/make_offline_bundle.sh | --out | --python-version(기본 현재 python3 버전) |
| third_party/install_offline.sh | --bundle, --venv, --tools-dir | 모두 저장소 밖 |

### 서비스 인자와 오프라인 기본값

| 인자 | orchestrate start 기본값 | opprofile 기본값 |
|---|---|---|
| --dtype | float16 (float16/bfloat16) | bfloat16 |
| --kv-cache-dtype | float16 (float16/bfloat16) | auto |
| --max-model-len | 1024 | 4352 |
| --max-num-batched-tokens | 128 | 128 |
| --max-num-seqs | 1 | 4 |
| --block-size | 128 | 128 |
| --num-blocks | 32 | 160 |
| --layout-reorder | 0 (0/1) | 해당 옵션 없음 |
| --cache-dir | 미지정 시 실행별 service/cache | 해당 옵션 없음; out/cache |
| --cuda-python, --rbln-python | CLI 우선, 없으면 CUDA_PY/RBLN_PY, 없으면 오류 | 현재 실행한 장치 Python 사용 |
| --expected-compiler | 미지정(None); 지정하면 실제 compiler와 일치 검사 | 해당 옵션 없음 |
| --role-env-json, --role-args-json | {} / {}; 역할별 허용 목록 검사 | 해당 옵션 없음 |

workload 객체 필드는 name/input_tokens/output_tokens/requests/seed/warmup과 구성별 slots 또는 concurrency다.
예약 이름·중복·모르는 키를 거부한다. 상세 형식은 run.validate_workloads와 scenarios/initial.json을 기준으로 한다.
