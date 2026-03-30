# P14 Baseline Final All-Datasets (Wide Hparam)

## 목적
FMoE 최종 비교를 위한 baseline 대조군을 동일 프로토콜로 대규모 실행한다.

- Track: `baseline`
- Axis: `Final_all_datasets`
- Eval mode: `session_fixed` (고정)
- Feature mode: `full_v3` (고정)

## 실행 규모
- 모델: `9`
- 데이터셋: `6`
- 하이퍼파라미터 bank: `12` (`H1..H12`, 모델별 오버레이 + 안전 cap)
- 기본 선택(`AUTO`): 모델별 최대 `12` preset (`--max-hparams-per-model`로 제한 가능)
- Seed 기본값: `1`
- 총 run 수(기본 AUTO): `648` (`9 x 6 x 12 x 1`)

모델 순서:
`SASRec, GRU4Rec, TiSASRec, DuoRec, SIGMA, BSARec, FEARec, DIFSR, FAME`

데이터셋 순서:
`KuaiRecLargeStrictPosV2_0.2, lastfm0.03, amazon_beauty, foursquare, movielens1m, retail_rocket`

## Hparam Bank (Wide + Model Adaptive)
- `H1/H2/H3/H4`: 기존 core bank (H1-like, balanced, deep, outlier-like)
- `H5/H6/H12`: 큰 width 중심 bank (large-dim 시도)
- `H7`: 작은 capacity + 높은 regularization
- `H8`: 긴 sequence + 높은 dropout
- `H9/H10`: 구조는 유사하지만 dropout/WD/LR multiplier를 달리한 LR-space 변형
- `H11`: deeper + regularized 변형

각 모델은 공통 bank 위에 model-specific overlay와 safety cap이 적용된다.
- `TiSASRec`: `max_len/time_span/layers/hidden` 상한으로 OOM 완화
- `DuoRec/FEARec`: contrastive/semantic 설정 분기 + width/length 상한
- `FAME`: `num_experts` 안전 범위 강제
- `SIGMA`: layer/length 상한 적용

## 모델별 키 매핑
- `SASRec/TiSASRec/DuoRec/FEARec`: `n_layers/n_heads/inner_size` (호환 위해 `num_*`도 함께 override)
- `BSARec/FAME/DIFSR`: `num_layers/num_heads/inner_size` (호환 위해 `n_*`도 함께 override)
- `GRU4Rec`: `dropout_prob`
- `SIGMA`: `state_size/conv_kernel/remaining_ratio`
- 추가 outlier knob
  - `TiSASRec`: `time_span`
  - `FAME`: `num_experts`
  - `DIFSR`: `attribute_hidden_size`

## LR 탐색 정책
`learning_rate`만 탐색(로그 스케일), `weight_decay/dropout/layer` 등은 고정 override.

- dataset별 base 구간의 geometric center를 사용해 모델별 narrow band(`hi/lo≈4~6`)로 변환
- model multiplier + hparam multiplier 반영
- 같은 구조 계열이라도(`H9/H10` 등) LR-space가 달라지도록 설계
- `search_space_type_overrides.learning_rate=loguniform`
- `GRU4Rec`: `dropout_prob` 고정
- 그 외 모델: `dropout_ratio` 고정

추가 속도 정책:
- `DuoRec/FEARec`는 contrastive preset을 H별로 분리(`un/su/us_x`)해 평균 trial 시간을 줄임
- `TiSASRec`는 OOM 완화를 위해 H4에서 더 작은 max_len/batch preset 적용
- `FEARec` 로컬 커스텀 구현을 사용해 semantic sampling/초기화 병목을 완화

## 스케줄링/실행 정책
- row 생성: dataset-major
- AUTO hparam 선택: 모델별 우선순위 bank를 기본으로, 해당 dataset `summary.csv`의 기존 성능이 있으면 best-first로 재정렬
- 런타임: shared queue(work-stealing)로 free GPU가 즉시 다음 작업을 가져감
- dry-run 출력: 가독성을 위해 deterministic projected assignment 표시
- 데이터셋은 순서대로 진행(현재 dataset 종료 후 다음 dataset 시작)

## 산출물 경로
- 로그: `experiments/run/artifacts/logs/baseline/Final_all_datasets/<dataset>/<model>/<hparam>/...`
- 결과 JSON: `experiments/run/artifacts/results/baseline/...`
- logging bundle mirror: `experiments/run/artifacts/logging/baseline/Final_all_datasets/<dataset>/<model>/<hparam>/...`
- special metrics: 결과 JSON의 `special_result_file`, `special_log_file`
- dataset 요약: `<dataset>/summary.csv`
- manifest: `<dataset>/final_matrix.json`
- summary.csv 컬럼: `global_best_valid_mrr20`, `model_best_valid_mrr20`, `run_best_valid_mrr20`

## Skip/재개 규칙
- 로그 마지막 non-empty line이 아래와 같을 때만 완료로 간주:
  - `[RUN_STATUS] END status=normal`
  - 또는 `[RUN_STATUS] END status=normal ...`
- `--verify-logging` 활성 시 skip 추가 조건:
  - 대응 result JSON 존재
  - `special_result_file` 존재
  - `special_log_file` 존재
- 로그만 완료이고 special 검증 실패면 재실행

## 예시 실행
### Dry run
```bash
bash experiments/run/baseline/phase_14_final_all_datasets.sh \
  --gpus 0,1,2,3 \
  --dry-run
```

### Smoke test
```bash
bash experiments/run/baseline/phase_14_final_all_datasets.sh \
  --gpus 0 \
  --smoke-test \
  --smoke-max-runs 4
```

### Full run (default)
```bash
bash experiments/run/baseline/phase_14_final_all_datasets.sh \
  --gpus 0,1,2,3
```

### 모델당 최대 hparam 개수 제한(예: 6개)
```bash
bash experiments/run/baseline/phase_14_final_all_datasets.sh \
  --gpus 0,1,2,3 \
  --max-hparams-per-model 6
```

## 기본값
- `seeds=1`
- `hparams=AUTO`
- `max-hparams-per-model=12`
- `max_evals=10`, `tune_epochs=60`, `tune_patience=8`
- `resume_from_logs=true`, `verify_logging=true`
