# Stage B (Structure) - baseline

## 목표
- Stage A에서 잡은 LR 신뢰구간을 유지하면서, 모델의 큰 구조(`dim/layers/heads/ffn`)를 6개 프로파일로 탐색한다.
- Stage A 결과를 사용할 때 `best valid`만 보지 않고, `early_stopped 비율 + epoch 사용률 + completion_ratio`를 함께 반영해 LR 윈도우를 뽑는다.

## Stage A 관측 요약 (현재 로그 기준)
- `lastfm0.03`
  - `SASRec`: LR3~LR4 (`7.2e-4 ~ 4.32e-3`)
  - `GRU4Rec`: LR5~LR6 (`2.45e-3 ~ 7.65e-3`)
  - `DuoRec`: LR2~LR3 (`3.24e-4 ~ 1.94e-3`)
  - `DIFSR`: LR3~LR4 (`6.84e-4 ~ 4.10e-3`)
  - `FAME`: LR3~LR4 (`5.76e-4 ~ 3.46e-3`)
- `amazon_beauty`
  - `SASRec`: LR3~LR4 (`9.6e-4 ~ 5.76e-3`)
  - `GRU4Rec`: LR4~LR5 (`1.63e-3 ~ 7.65e-3`)
  - `DuoRec`: LR2~LR3 (`4.32e-4 ~ 2.59e-3`)
  - `DIFSR`: LR4~LR6 (`1.82e-3 ~ 1.0e-2`)
  - `FAME`: LR4~LR6 (`1.54e-3 ~ 9.6e-3`)

## LR 선택 규칙 (A -> B)
- 입력: `StageA_LR_anchor2_core5/<dataset>/summary.csv` + result JSON `trials[]`.
- band 점수:
  - `score = valid_mrr20 - 0.004*early_stop_ratio + 0.002*epoch_usage_ratio - 0.010*(1-completion_ratio)`
- 의미:
  - `early_stop_ratio`가 높으면 불리
  - `epoch_usage_ratio`(평균 `epochs_run/tune_epochs`)가 높으면 유리
  - trial 완료율이 낮으면 불리
- 선택: 점수 상위 2개 band 병합 -> Stage B의 profile별 LR base window.

## Stage B 6개 프로파일
- `B1 compact_reg`
  - hidden `x0.75`, layers `+0`, inner ratio `2`, dropout `+0.04`, wd `x1.6`, lr center `x0.85`
- `B2 balanced`
  - hidden `x1.00`, layers `+0`, inner ratio `2`, dropout `+0.00`, wd `x1.0`, lr center `x1.00`
- `B3 wide`
  - hidden `x1.25`, layers `+1`, inner ratio `3`, dropout `-0.02`, wd `x0.85`, lr center `x1.05`
- `B4 deeper`
  - hidden `x1.00`, layers `+1`, inner ratio `2`, dropout `+0.02`, wd `x1.1`, lr center `x0.95`
- `B5 thin_deep_outlier`
  - hidden `x0.85`, layers `+2`, inner ratio `2`, dropout `+0.05`, wd `x1.8`, lr span 확대
- `B6 wide_shallow_outlier`
  - hidden `x1.40`, layers `-1`, inner ratio `4`, dropout `-0.03`, wd `x0.7`, lr span 확대

## 실행
```bash
# dry-run
bash experiments/run/baseline/stageB_structure.sh --dry-run

# smoke
bash experiments/run/baseline/stageB_structure.sh --smoke-test --gpus 0

# full
bash experiments/run/baseline/stageB_structure.sh \
  --datasets lastfm0.03,amazon_beauty \
  --models sasrec,gru4rec,duorec,difsr,fame \
  --gpus 0,1,2,3 --seeds 1
```

## 산출물
- 로그: `experiments/run/artifacts/logs/baseline/StageB_Structure_anchor2_core5/<dataset>/<model>/...`
- 결과: `experiments/run/artifacts/results/baseline/normal/stageb_structure_anchor2_core5/...`
- summary: `.../StageB_Structure_anchor2_core5/<dataset>/summary.csv`
- 후보: `.../StageB_Structure_anchor2_core5/<dataset>/stageB_candidates.json`

## 승급 규칙 (B -> C)
- 모델별 `Top-2 + Stability` (`score_mean` 기준)
- Stage C는 기본적으로 이 `selected_profiles`를 자동 사용 (`--b-topk 2`).
