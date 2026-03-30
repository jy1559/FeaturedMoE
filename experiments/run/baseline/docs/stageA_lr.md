# Stage A LR Scan (Anchor-2, Core-5)

## 목적
Stage A는 "좋은 LR 대역을 빠르게 찾는 단계"다.

- 실행 축: `2 datasets x 5 models x 6 LR bands x seed`
- run 내부 탐색: `learning_rate`만 (`loguniform`)
- 나머지 hparam은 모델별 고정값(singleton choice)

## 대상
- Datasets: `lastfm0.03`, `amazon_beauty`
- Models: `SASRec`, `GRU4Rec`, `DuoRec`, `DIFSR`, `FAME`

## LR band 정책
기본 6개 band:
- `LR1`: `[2e-4, 7e-4]`
- `LR2`: `[4e-4, 1.2e-3]`
- `LR3`: `[8e-4, 2.4e-3]`
- `LR4`: `[1.6e-3, 4.8e-3]`
- `LR5`: `[3.2e-3, 7.5e-3]`
- `LR6`: `[5e-3, 1e-2]`

보정:
- dataset multiplier:
  - `lastfm0.03 x0.90`
  - `amazon_beauty x1.20`
- model multiplier:
  - `SASRec x1.00`
  - `GRU4Rec x0.85`
  - `DuoRec x0.90`
  - `DIFSR x0.95`
  - `FAME x0.80`
- clamp: `LR in [8e-5, 1e-2]`

## 고정 hparam (LR 외)
공통:
- 기본 `MAX_ITEM_LIST_LENGTH=10`
- `session_fixed + full_v3`

모델별:
- SASRec:
  - hidden 128, layers 2, heads 4, inner 256, dropout 0.20, wd `3e-4`
- GRU4Rec:
  - hidden 160, layers 1, dropout_prob 0.20, wd `2e-4`
- DuoRec:
  - hidden 96, layers 1, heads 2, inner 192, dropout 0.20, wd `3e-4`
  - `contrast=un`, `tau=0.30`, `lmd=0.04`, `lmd_sem=0.0`
- DIFSR:
  - hidden 128, layers 1, heads 4, inner 256, dropout 0.20, wd `2e-4`
  - `fusion_type=gate`, `use_attribute_predictor=true`, `lambda_attr=0.10`
- FAME:
  - hidden 128, layers 1, heads 4, inner 256, dropout 0.20, wd `3e-4`
  - `num_experts=2`

## 예산 (속도 우선)
- 기본(default):
  - `max_evals=6`, `tune_epochs=36`, `tune_patience=5`
- DuoRec (경량):
  - `max_evals=4`, `tune_epochs=24`, `tune_patience=3`
- FAME:
  - `max_evals=5`, `tune_epochs=32`, `tune_patience=4`

## 실행기
- Python: `experiments/run/baseline/run_stageA_lr.py`
- Shell: `experiments/run/baseline/stageA_lr.sh`

예시:
```bash
bash experiments/run/baseline/stageA_lr.sh --dry-run
```

```bash
bash experiments/run/baseline/stageA_lr.sh --gpus 0,1,2,3
```

```bash
bash experiments/run/baseline/stageA_lr.sh --gpus 0 --smoke-test --smoke-max-runs 4
```

## 산출물/해석
dataset 폴더 기준 산출:
- `summary.csv`
- `final_matrix.json`
- `stageA_candidates.json`

`stageA_candidates.json` 선별 규칙:
1. 모델별 band score 계산
   - `score = valid_mrr20 - 0.5*std(top3_trial_valid_mrr20) - 0.01*(1-completion_ratio)`
2. `Top-2 + Stability` 선별
3. best가 경계 band이거나 best LR가 경계 10% 이내면 인접 band 1개 추가

Stage B 입력은 `stageA_candidates.json`의 `selected_bands`만 사용한다.
