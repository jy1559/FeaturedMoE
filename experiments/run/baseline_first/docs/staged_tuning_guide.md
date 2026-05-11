# Baseline2 A→B→C→D 튜닝 가이드

## 1) 핵심 요약
- `Stage A`는 **hyperopt 탐색 단계가 아님**.
- 현재 기본(lean) 기준으로 `Stage A`는 구조 후보 18개를 각각 `max_evals=1`로 실행한다.
- 내부적으로 `hyperopt_tune.py`를 호출하지만, search space가 모두 단일값(`choice` 1개)이라 사실상 **단일 run**과 동일하다.
- 즉, `learning_rate / weight_decay / dropout`도 후보별 단일값으로 고정되고, 주로 **discrete 구조축**(`max_len`, `layers`, `heads`, `hidden`)을 빠르게 비교한다.
- 이후 `Stage B`부터 연속축 hyperopt를 켠다.
- 참고: 기존 장기 실험 축 `ABCD_v1`(30xLR6)도 그대로 재현 가능하며, 현재 스크립트 기본 축은 `ABCD_v2_lean`이다.

## 2) Stage별 목적과 탐색 방식

### Stage A (빠른 구조 스크리닝)
- 목적: 구조 조합을 빠르게 거르고 상위 6개만 남기기.
- 후보 수: 모델/데이터셋당 구조 18개.
- 실행: 구조 18개 각각에 LR grid 4개를 적용 (총 72 run/모델/데이터셋).
- 각 run은 `max_evals=1`, 짧은 epoch (lean 기본: 14).
- 탐색 축:
  - SASRec: `max_len`, `num_layers`, `num_heads`, `hidden_size`
  - TiSASRec: 위 + `time_span`
  - GRU4Rec: `max_len`, `num_layers`, `hidden_size`
- Stage A LR grid(lean 기본): `2e-4,6e-4,1.2e-3,3e-3`
- 동일 구조 후보는 LR별 그룹 폴더(`lr_groups/lr_...`)로 저장.
- 장점: 속도.
- 한계: 연속축 최적화가 덜 되어 순위 노이즈 가능.

### Stage B (상위 6개 연속축 hyperopt)
- 목적: A에서 남긴 구조를 고정하고, 연속축 최적화.
- 입력: A 상위 6개.
- 실행: 후보당 `epochs=32`, `patience=4` (lean), TPE.
- `max_evals`는 ABCD_v2에서 데이터셋별 자동 조정:
  - 가벼운 셋(beauty 등): B 최대 40
  - 무거운 셋(lastfm0.03/movielens1m 등): B 약 20
- 탐색 축:
  - 공통: `learning_rate(loguniform)`, `weight_decay(loguniform)`
  - SASRec/TiSASRec: `dropout_ratio`, `hidden_dropout_prob`, `attn_dropout_prob` (uniform)
  - GRU4Rec: `dropout_prob` (uniform)
- 출력: 상위 3개.

### Stage C (재변이 + 재선별)
- 목적: B 상위 3개 주변에서 논문/공식 문서 강조축을 근방 탐색.
- 입력: B 상위 3개.
- 생성: parent당 2개 변이 => 6개.
- 변이 예:
  - `max_len` 인접값 이동
  - TiSASRec의 `time_span` 인접값 이동
  - `dropout`, `weight_decay` 미세 이동
- 실행: 중간 예산(TPE, lean 기준 `max_evals=8`, `epochs=48`, `patience=6`), 출력 상위 2개.
- 실행: 중간 예산(TPE, lean 기준 `epochs=48`, `patience=6`, `max_evals`는 데이터셋별 자동 조정), 출력 상위 2개.
- 변이 원칙: B 상위 3개의 근방에서 구조축+연속축을 함께 변이
  - 구조축: `max_len`, (모델별) `layers/heads/hidden/time_span` 등
  - 연속축: `lr/dropout/wd`

### Stage D (최종 심화 + 확정)
- 목적: C 상위 2개 주변에서 final 후보 압축.
- 입력: C 상위 2개.
- 생성: parent당 2개 변이 => 4개.
- 실행: 긴 예산(TPE, lean 기준 `epochs=64`, `patience=8`, `max_evals`는 데이터셋별 자동 조정), 기본적으로 최종 seed 재검증(`--final-seeds`, 기본 `1,2`) 가능.
- 변이 원칙: C 상위 2개의 근방 미세 변이(구조축+연속축)로 정밀 확정.
- 출력: 최종 1개.

## 3) Stage A 신뢰도 보완 방식
- A는 의도적으로 “빠른 구조 필터”다. A 결과만으로 결론을 내리지 않는다.
- B에서 연속축 hyperopt(데이터셋별 `max_evals` 정책)로 구조별 편차를 다시 줄인다.
- C/D에서 주변 변이와 더 긴 학습으로 재검증해 A의 초기 편향을 완화한다.
- 최종 우승은 D 단계 기준이며, 필요하면 `--final-seeds` 평균으로 안정성을 확인한다.

## 4) 승급 규칙
- 기본 승급 폭(lean): `18xLR4 → 6 → 3 → (변이6) → 2 → (변이4) → 1`
- 레거시(v1): `30xLR6 → 12 → 4 → (변이12) → 2 → (변이6) → 1`
- 정렬 기준:
  1. `best_valid_mrr20` (seen-main)
  2. `test_mrr20` (seen-main)
  3. `test_unseen_mrr20` (cold target)

## 5) 데이터/평가 프로토콜 (v4)
- 입력 데이터: `feature_added_v3`
- 출력 데이터: `feature_added_v4`
- 변환 규칙:
  - train은 그대로 복사
  - 기존 valid+test 세션 병합 후 session start time strict 정렬, 50/50 재분할
  - valid/test에서 non-target unseen item interaction drop
  - target unseen은 유지
- 평가/로깅:
  - `feature_mode=full_v4`
  - `exclude_unseen_target_from_main_eval=true`
  - `log_unseen_target_metrics=true`
  - main metric은 seen-target 기준, unseen-target은 special/cold 지표로 별도 기록

## 6) 모델별 튜닝 원칙 (1차: 3개 모델)

### SASRec
- Stage A: 구조축 우선
  - `max_len`(20/50/100/200), `layers`(1/2/3), `heads`(1/2/4), `hidden`(96/128/160/192)
- Stage B/C/D: 연속축 보정
  - `lr`, `dropout 계열`, `wd`

### TiSASRec
- Stage A: 구조 + 시간축
  - SASRec 구조축 + `time_span`(64/256/1024/4096)
- Stage B/C/D: `lr/dropout/wd` + `time_span` 근방 재변이

### GRU4Rec
- Stage A: 구조축
  - `max_len`, `num_layers`, `hidden`
- Stage B/C/D: `lr/dropout_prob/wd`

## 7) 데이터셋별 max_len 실무 가이드
- short 계열(`beauty`, `foursquare`, `retail_rocket`, `KuaiRecLargeStrictPosV2_0.2`):
  - Stage A에서 20/50 중심, 필요 시 100 제한적 확인
- long 계열(`lastfm0.03`, `movielens1m`):
  - Stage A에서 50/100 우선, 200 포함 유지

## 8) 실행 스크립트
- Stage별:
  - `experiments/run/baseline_2/stageA_baseline2.sh`
  - `experiments/run/baseline_2/stageB_baseline2.sh`
  - `experiments/run/baseline_2/stageC_baseline2.sh`
  - `experiments/run/baseline_2/stageD_baseline2.sh`
- 통합:
  - `experiments/run/baseline_2/run_all_stages.sh`

기본값(통합 실행):
- 모델: `sasrec,tisasrec,gru4rec` (3개)
- 데이터셋: `KuaiRecLargeStrictPosV2_0.2,beauty,lastfm0.03,foursquare,movielens1m,retail_rocket` (6개)

즉 기본 실행은 `3 models x 6 datasets` 조합을 stage A→B→C→D로 돈다.

실행 예시:
```bash
# 기본 실행 (Slack OFF)
bash experiments/run/baseline_2/run_all_stages.sh

# GPU 2개 라운드로빈
GPU_LIST=0,1 bash experiments/run/baseline_2/run_all_stages.sh

# 모델/데이터셋 제한 (뒤 인자는 run_staged_tuning.py로 전달)
bash experiments/run/baseline_2/run_all_stages.sh -- \
  --models sasrec,tisasrec \
  --datasets beauty,lastfm0.03

# 예산 프로파일 변경
bash experiments/run/baseline_2/run_all_stages.sh -- --budget-profile fast
```

### Slack 알림
- `.env.slack` 위치: `experiments/run/baseline_2/.env.slack`
- 필수: `SLACK_WEBHOOK_URL=...`
- 통합 실행에서 알림 시점:
  - 시작
  - 각 stage(A/B/C/D) 완료
  - 전체 완료
  - 실패/중단

Slack on/off:
```bash
# 명시 ON
bash experiments/run/baseline_2/run_all_stages.sh --slack-on

# 명시 OFF
bash experiments/run/baseline_2/run_all_stages.sh --slack-off

# 환경변수 ON
SLACK_NOTIFY=1 bash experiments/run/baseline_2/run_all_stages.sh
```

제목/노트:
```bash
bash experiments/run/baseline_2/run_all_stages.sh \
  --slack-on \
  --title "Baseline2 ABCD" \
  --note "ablation pre-run"
```

단일 명령 감싸기용(옵션):
- `experiments/run/baseline_2/run_with_slack_notify.sh`
- 예: `bash experiments/run/baseline_2/run_with_slack_notify.sh --on bash experiments/run/baseline_2/stageA_baseline2.sh`

### 주요 인자 정리
- `GPU_LIST` (env): 기본 GPU 리스트. 예: `GPU_LIST=0,1,2`
- `RUNTIME_SEED` (env): stage 런타임 seed
- `FINAL_SEEDS` (env): stage D 다중 seed. 예: `1,2`
- `--models` (passthrough): `sasrec,tisasrec,gru4rec` 중 선택
- `--datasets` (passthrough): 데이터셋 CSV
- `--gpus` (passthrough): `GPU_LIST` 대신 직접 지정 가능
- `--budget-profile` (passthrough): `balanced|lean|fast|deep`
- `--dataset-max-eval-policy` (passthrough): `off|abcd_v2` (기본 `abcd_v2`, axis가 `ABCD_v2`로 시작할 때 dataset별 max_evals 적용)
- `--axis` (passthrough): 결과 폴더 축 이름 (현재 스크립트 기본 `ABCD_v2_lean`)
- `--runtime-seed` / `--final-seeds` (passthrough): seed 직접 지정
- `--resume-from-logs` (passthrough): 기존 완료 run 재사용
- `--fast-first` / `--no-fast-first` (passthrough): 빠른 dataset/model 조합 우선 실행 (기본 ON)
- `--slack-on|--slack-off|--title|--note` (run_all_stages.sh 전용): Slack 제어

GPU 스케줄링 동작:
- `run_staged_tuning.py`는 GPU 개수만큼 워커를 띄워 **동시 실행**한다.
- 워커들은 공통 작업 큐에서 작업을 가져가므로, 먼저 끝난 GPU가 다음 작업을 바로 이어서 처리한다.
- 따라서 `GPU_LIST=0,1,4,5,6,7`이면 최대 6개 run이 병렬로 진행된다(리소스/메모리 허용 범위 내).

## 9) 산출물
- 로그/요약 루트:
  - `experiments/run/artifacts/logs/baseline_2/ABCD_v2_lean/`
- 실행 로그 경로:
  - `ABCD_v2_lean/<dataset>/<model>/stageA/lr_groups/<lr_group>/logs/*.log`
  - `ABCD_v2_lean/<dataset>/<model>/stageB/logs/*.log` (C/D 동일)
- stage 메타/집계:
  - `ABCD_v2_lean/stages/stageA|B|C|D/{summary.csv,promotion.csv,leaderboard.csv,manifest.json}`
- 데이터셋 누적 요약:
  - `ABCD_v2_lean/<dataset>/summary.csv`
  - 필드: `best_valid_mrr20`, `best_test_mrr20`, `cum_best_valid_mrr20`, `cum_best_test_mrr20`
- stage별 파일:
  - `summary.csv`
  - `promotion.csv`
  - `leaderboard.csv`
  - `manifest.json`
- 주요 컬럼:
  - `best_valid_mrr20`, `test_mrr20`
  - `valid_unseen_mrr20`, `test_unseen_mrr20`
  - `valid_main_seen_count`, `valid_main_unseen_count`
  - `test_main_seen_count`, `test_main_unseen_count`
