# FMoE Phase Runner 구현 가이드 표준화

작성 목적: `phase*.md` 계획 문서를 기준으로 실행용 `sh/py`를 만들 때, 로그/로깅/summary/파일명/재실행/Slack 규칙을 공통으로 고정해 phase 간 편차를 줄인다.

---

## 0) 적용 범위와 원칙

- 적용 범위: `experiments/run/fmoe_n3` 하위의 신규/수정 phase runner(`run_phase*.py`, `phase_*.sh`)
- 기본 원칙:
  - 로그 경로는 `phase.../<dataset>` 중심으로 통일
  - `summary.csv`는 공통 선두 컬럼 + phase 확장 컬럼 전략
  - 동일 로그 이름 충돌 시: 완료 로그면 skip, 아니면 덮어쓰기 재실행
- 비목표:
  - 기존 phase8/9/10 전체를 즉시 일괄 리팩터링하지 않음

---

## 1) 실행 타입 표준

### 1.1 Wide Search

- 기본 목적: 설정 축(screening) 빠르게 비교
- 기본 run 밀도: `1 setting = 1 hparam x 1 seed`
- 대표 예: `phase9`류의 넓은 조합 탐색

### 1.2 Deep Verification (`_2`)

- 기본 목적: 이전 phase 상위 설정 정밀 검증
- 기본 run 밀도: `선정 setting = 4 hparam x 4 seed`
- 대표 예: `phase9_2`류의 후속 검증

### 1.3 plan 문서 필수 표기

각 phase 계획 문서는 아래를 반드시 명시한다.

- 실행 타입(`wide` 또는 `deep`)
- setting 개수
- hparam 개수
- seed 개수
- 총 run 수 계산식

---

## 2) 표준 저장 경로

### 2.1 로그(`.log`) 경로

기본 로그 루트:

- `experiments/run/artifacts/logs/fmoe_n3/<phase_axis>/<dataset>/`

권장 하위 구조:

- 기본(요청안 우선): model_tag 없이 바로 `.log` 저장
- 선택(phase 필요 시): `/<model_tag>/` 하위 저장 가능

예시:

- `.../logs/fmoe_n3/phase10_feature_portability_v1/KuaiRecLargeStrictPosV2_0.2/P10_A_group_subset_00_FULL.log`

### 2.2 로깅 산출물(logging/special/diag) 경로

기존 운영 유지:

- `run/artifacts/logging` 루트 사용
- `diag/special` 상세 구조는 현재 러너 규칙을 따른다

---

## 3) 로그 파일명 규칙

## 3.1 slug 규칙

- 공백/특수문자 -> `_`
- 연속 `_` -> 단일 `_`
- 허용 문자: 영문, 숫자, `_`
- 식별자(`Pn`, `axis_id`, `setting_id`, `Hn`, `Sn`)는 대문자/숫자 유지
- 설명(`axis_desc`, `setting_desc`)은 소문자 snake_case 권장

### 3.2 Wide 파일명

포맷:

- `Pn_<axis_id>_<axis_desc>_<setting_id>_<setting_desc>.log`

예시(3개):

- `P10_A_group_subset_00_FULL.log`
- `P10_B_compactness_16_TOP1_PER_GROUP.log`
- `P11_C_stage_semantics_05_MID_ONLY.log`

### 3.3 Deep(`_2`) 파일명

포맷:

- `Pn_<axis_id><setting_id>_<axis_desc>_<setting_desc>_H<hparam_id>_S<seed_id>.log`

예시(3개):

- `P10_A00_group_subset_FULL_H1_S4.log`
- `P9_2_B03_canonical_balance_B3_H2_S1.log`
- `P12_D07_layout_compose_MIXED2_H4_S3.log`

---

## 4) skip/overwrite 규칙 (강제)

### 4.1 완료 판정

동일 로그 파일이 존재하고, 마지막 non-empty line이 아래로 시작하면 완료로 간주한다.

- `[RUN_STATUS] END status=normal`

주의:

- 뒤에 `pid/start/end` 같은 추가 메타가 붙어도 완료로 인정한다.

### 4.2 동작 규칙

- 완료 로그: skip
- 미완료/없음: 같은 파일명으로 preamble부터 덮어쓰기 후 재실행

### 4.3 판정 함수 시그니처(템플릿)

```python
def _is_completed_log(log_path: Path) -> bool:
    ...
```

### 4.4 의사코드

```text
if not exists(log_path):
  return run

last = last_non_empty_line(log_path)
if last startswith "[RUN_STATUS] END status=normal":
  return skip
else:
  overwrite log preamble
  return run
```

---

## 5) summary.csv 표준

### 5.1 위치

- 해당 phase 로그 루트의 dataset 폴더
- 경로: `.../artifacts/logs/fmoe_n3/<phase_axis>/<dataset>/summary.csv`

### 5.2 공통 선두 컬럼(고정 순서)

- `global_best_valid_mrr20`
- `run_best_valid_mrr20`
- `run_phase`
- `exp_brief`
- `stage`

### 5.3 공통 메타 컬럼(권장)

- `trigger`
- `dataset`
- `seed_id`
- `gpu_id`
- `status`
- `test_mrr20`
- `n_completed`
- `interrupted`
- `result_path`
- `timestamp_utc`

### 5.4 phase 확장 컬럼

- 위 공통 컬럼 뒤에 phase 전용 컬럼을 추가한다.
- 예: `setting_id`, `setting_key`, `axis_id`, `hparam_id`, `special_ok`, `diag_ok`

### 5.5 append 트리거(강제)

- run 종료 시: 무조건 1행 append
- 실행 중 global best 갱신 시: 즉시 1행 append

예시 이벤트:

- `trigger=run_complete`
- `trigger=trial_new_best`

---

## 6) `py`/`sh` 생성 계약 (필수 인터페이스)

### 6.1 `py` row 메타 필수 키

- `phase_id`
- `axis_id`
- `axis_desc`
- `setting_id`
- `setting_desc`
- `hparam_id`
- `seed_id`
- `run_phase`
- `exp_brief`

### 6.2 `py` 필수 함수 템플릿

- `_log_path_from_row`
- `_is_completed_log`
- `_write_log_preamble`
- `_ensure_summary_csv`
- `_append_summary_event`

### 6.3 `sh` 필수 규약

- 기본 preset: dataset/gpu/seeds/hparam
- `RUN_PYTHON_BIN` 우선 사용
- `--dry-run`, `--resume` 옵션 pass-through
- `set -euo pipefail` 기본 적용

---

## 7) Slack 래핑 규칙

기본 실행 예시는 아래 래퍼를 사용한다.

- `experiments/run/fmoe_n3/run_with_slack_notify.sh`

권장 실행:

```bash
bash experiments/run/fmoe_n3/run_with_slack_notify.sh \
  --on --title "P10 feature portability" -- \
  bash experiments/run/fmoe_n3/phase_10_feature_portability.sh --gpus 4,5,6,7
```

원칙:

- `sh`는 Slack 의존 없이 순수 실행 스크립트로 유지
- 알림은 wrapper가 담당

---

## 8) 문서 검증 체크리스트

### 8.1 문서 자체 검증

- [ ] wide/deep 각각 입력 row -> 기대 로그 파일명 예시 3개 이상
- [ ] skip 판정 예시 3개(정상 완료/비정상 종료/빈 파일)
- [ ] summary append 예시 2개(run_complete, trial_new_best)

### 8.2 적용성 검증(향후 구현 시)

- [ ] 신규 phase py dry-run에서 경로/파일명/summary 헤더 일치 확인
- [ ] Slack wrapper로 START/END 알림 및 exit code 전달 확인

---

## 9) 채택 정책

- 신규/수정 phase부터 본 가이드를 준수한다.
- 기존 phase는 필요 시점에 점진적으로 이관한다.
- 예외가 필요한 경우 phase 계획 문서에 이유/범위를 명시한다.
