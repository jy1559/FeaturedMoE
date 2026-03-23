# Phase10~13 Unified Deep Verification Wrapper Code Guide

## 1) 목적

`phase10~13` wide 실험에서 뽑은 후보 setting을 **수동 입력**으로 받아,
`N settings x M hparams x K seeds` 형태의 deep verification을 한 번에 실행한다.

- 자동 best selection은 하지 않는다.
- phase를 섞어서 입력할 수 있다.
- 기본은 `M=2`, `K=4`이지만 가변이다.

---

## 2) 구현 파일

- Python runner: `experiments/run/fmoe_n3/run_phase10_13_verification_wrapper.py`
- Shell wrapper: `experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh`
- Axis/Phase:
  - `AXIS=phase10_13_verification_wrapper_v1`
  - `PHASE=P10_13_2`

---

## 3) Setting 카탈로그 수집 방식

런타임에 기존 phase runner의 `_build_settings`를 직접 호출해 카탈로그를 만든다.

- P10: `_build_settings(... include_extra_24=True)`
  - 즉 `P10-22`, `P10-23` 보강 setting도 기본 포함.
- P11/P12/P13: 각 phase 전체 setting 포함.

카탈로그 canonical key 형식:

- `P10-xx_*`
- `P11-xx_*`
- `P12-xx_*`
- `P13-xx_*`

카탈로그 확인:

```bash
python3 experiments/run/fmoe_n3/run_phase10_13_verification_wrapper.py --list-settings
```

또는

```bash
bash experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh --list-settings
```

---

## 4) Setting 입력 포맷 (mixed tolerant)

### 4.1 허용 포맷

1. canonical key 직접 입력
- 예: `P10-00_FULL`
- 예: `P12-31_BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED`

2. phase-qualified index
- 예: `P11:23`
- 예: `P13:17`

3. phase-qualified short key
- 예: `P10:FULL`
- 예: `P11:LAYER_ONLY_BASELINE`

4. phase+idx tolerant key (suffix 무시)
- 예: `P10-01_TEMPO_ONLY`
- 예: `P12_31_ANY_LABEL`
- `Pxx-idx` / `Pxx_idx` 형태에서 `idx`가 우선이며 뒤 suffix는 매칭에 사용하지 않는다.

5. bare token
- 예: `FULL`, `LAYER_ONLY_BASELINE`
- 단, **전체 카탈로그에서 유일 매칭일 때만 허용**.
- 다중 매칭이면 에러 + 후보 목록 출력.

### 4.2 전달 방법

- `--settings "token1,token2,..."`
- `--settings-json path.json`

`--settings-json`은 JSON list를 받는다.

```json
[
  "P10-00_FULL",
  "P11:23",
  "P12-25_BUNDLE_ALL_ROUTER_CONDITIONED",
  "P13:17"
]
```

중복 token은 canonical key 기준으로 자동 dedupe된다.

---

## 5) Hparam 선택 규칙

## 5.1 기본 우선순위

- 우선순위: `H1, H3, H2, H4, H5, H6, H7, H8`
- `--hparam-count n`이면 위 순서에서 앞 `n`개 선택.
- 기본값: `--hparam-count 2` => `H1,H3`
- 최대 8개.

## 5.2 명시 입력 우선

`--hparams`가 들어오면 `--hparam-count`보다 우선한다.

- 허용: `--hparams 1,3` 또는 `--hparams H1,H3`

## 5.3 Hparam Bank

- H1: emb128, dff256, expert128, router64, wd1e-6, drop0.15
- H2: emb160, dff320, expert160, router80, wd5e-7, drop0.12
- H3: emb160, dff320, expert160, router80, wd2e-6, drop0.18
- H4: emb112, dff224, expert112, router56, wd3e-6, drop0.20
- H5: emb168, dff336, expert168, router84, wd1e-6, drop0.15
- H6: emb144, dff288, expert144, router72, wd1.5e-6, drop0.17
- H7: emb160, dff320, expert160, router80, wd1e-6, drop0.15
- H8: emb128, dff256, expert128, router64, wd2.5e-6, drop0.19

---

## 6) 실행 수와 run naming

총 run 수:

- `N(settings) x M(hparams) x K(seeds)`

기본 seeds:

- `1,2,3,4`

run naming:

- `run_id = <axis_id><setting_uid>_H<hid>_S<sid>`
- `run_phase = P10_13_2_<axis_id><setting_uid>_H<hid>_S<sid>`

로그 파일명:

- `P10_13_2_<axis_id><setting_uid>_<axis_desc>_<setting_desc>_H<hid>_S<sid>.log`

예시:

- `P10_13_2_AP10_00_verification_wrapper_full_H1_S1.log`
- `P10_13_2_BP11_23_verification_wrapper_layer_only_baseline_H3_S4.log`

---

## 7) 저장 경로/요약/재실행

로그 루트:

- `experiments/run/artifacts/logs/fmoe_n3/phase10_13_verification_wrapper_v1/<dataset>/`

요약 파일:

- `summary.csv` (동일 경로)
- append 이벤트:
  - `trigger=trial_new_best`
  - `trigger=run_complete`

매니페스트:

- 기본: `verification_matrix.json` (동일 경로)
- override: `--manifest-out`

resume 규칙:

- strict 완료 마커:
  - 마지막 non-empty line이 `[RUN_STATUS] END status=normal`로 시작하면 skip.
- 완료가 아니면 같은 로그 파일명으로 preamble overwrite 후 재실행.

---

## 8) 주요 CLI

Python:

- `--settings`, `--settings-json`, `--list-settings`
- `--hparam-count`, `--hparams`, `--seeds`, `--seed-base`
- `--dataset`, `--gpus`
- `--max-evals`, `--tune-epochs`, `--tune-patience`
- `--manifest-out`
- `--resume-from-logs / --no-resume-from-logs`
- `--verify-logging / --no-verify-logging`
- `--dry-run`, `--smoke-test`, `--smoke-max-runs`

Shell:

- 위 핵심 옵션들을 pass-through 지원.
- `RUN_PYTHON_BIN` 우선 사용.

---

## 9) 실행 예시

### 9.1 기본(8 settings x 2 hparams x 4 seeds)

```bash
bash experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh \
  --gpus 4,5,6,7 \
  --settings "P10-00_FULL,P10-18_NO_CATEGORY,P11:00,P11:23,P12:25,P12:31,P13:00,P13:17"
```

### 9.2 32 settings 일괄 + dry-run

```bash
bash experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh \
  --settings-json /path/to/selected_32.json \
  --hparam-count 2 \
  --seeds 1,2,3,4 \
  --dry-run
```

### 9.3 hparam 명시 override

```bash
bash experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh \
  --settings "P11:23,P12:25,P13:17" \
  --hparams H1,H3,H2 \
  --seeds 1,2,3,4
```

### 9.4 smoke test

```bash
bash experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh \
  --settings "P10-00_FULL,P11:00" \
  --gpus 4 \
  --smoke-test \
  --smoke-max-runs 2
```

---

## 10) 검증 체크

- `python3 -m py_compile experiments/run/fmoe_n3/run_phase10_13_verification_wrapper.py`
- `bash -n experiments/run/fmoe_n3/phase_10_13_verification_wrapper.sh`
- `--list-settings` 결과 count 확인: `P10=24, P11=24, P12=32, P13=24`
- `--dry-run`에서 run count/manifest/run naming 확인
- resume 동작(완료 skip, 미완료 overwrite) 확인
- summary append 이벤트 2종 확인
