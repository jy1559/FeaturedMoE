# Stage C (Focus Knobs) - baseline

## 목표
- Stage B 상위 구조(`Top-k`)를 고정하고, 모델별 중간 중요 knob + `max_len {10,15,20}`을 6개 프로파일로 탐색한다.
- LR 탐색은 계속 `learning_rate only`로 유지한다.

## 입력 (자동)
- 파일: `experiments/run/artifacts/logs/baseline/StageB_Structure_anchor2_core5/<dataset>/stageB_candidates.json`
- 모델별 `selected_profiles`에서 기본 `top-2`를 자동 로드 (`--b-topk 2`).
- Stage B 후보가 없으면 Stage A LR 윈도우 + 기본 config로 fallback.

## Stage C 6개 프로파일
- `C1 short_reg`
  - `max_len=10`, dropout↑, wd↑, 보수 LR
  - DuoRec: `contrast=un, tau=0.25, lmd=0.02`
  - DIFSR: `fusion=sum, lambda_attr=0.05`
  - FAME: `num_experts=2`
- `C2 short_balanced`
  - `max_len=10`, 기준형
  - DuoRec: `su, tau=0.45, lmd=0, lmd_sem=0.06`
  - DIFSR: `gate, lambda_attr=0.10`
  - FAME: `num_experts=3`
- `C3 mid_aggressive`
  - `max_len=15`, dropout↓, wd↓, 공격적 LR
  - SASRec류: inner ratio↑
  - GRU4Rec: layers +1
  - DuoRec: `us_x, tau=0.8, lmd=0.10, lmd_sem=0.08`
  - DIFSR: `gate, lambda_attr=0.15`
  - FAME: `num_experts=4`
- `C4 long_reg`
  - `max_len=20`, dropout↑, wd↑, 보수 LR
  - DuoRec: `un, tau=0.30, lmd=0.06`
  - DIFSR: `concat, use_attribute_predictor=false`
  - FAME: `num_experts=6`
- `C5 mid_sparse_outlier`
  - `max_len=15`, 강한 정규화 outlier
  - SASRec류: head 축소/ffn 축소
  - DIFSR: `sum + predictor off`
- `C6 long_dense_outlier`
  - `max_len=20`, 저정규화 outlier
  - SASRec류: head/ffn 확장
  - DuoRec: `su, lmd_sem 강화`
  - DIFSR: `gate, lambda_attr=0.20`
  - FAME: `num_experts=6`

## 실행
```bash
# dry-run
bash experiments/run/baseline/stageC_focus.sh --dry-run

# smoke
bash experiments/run/baseline/stageC_focus.sh --smoke-test --gpus 0

# full (B top-2 자동 사용)
bash experiments/run/baseline/stageC_focus.sh \
  --datasets lastfm0.03,amazon_beauty \
  --models sasrec,gru4rec,duorec,difsr,fame \
  --profiles C1,C2,C3,C4,C5,C6 \
  --b-topk 2 --gpus 0,1,2,3 --seeds 1
```

## B/C 연속 실행
```bash
bash experiments/run/baseline/stageBC.sh \
  --datasets lastfm0.03,amazon_beauty \
  --models sasrec,gru4rec,duorec,difsr,fame \
  --gpus 0,1,2,3 --seeds 1
```

## 산출물
- 로그: `experiments/run/artifacts/logs/baseline/StageC_Focus_anchor2_core5/<dataset>/<model>/...`
- 결과: `experiments/run/artifacts/results/baseline/normal/stagec_focus_anchor2_core5/...`
- summary: `.../StageC_Focus_anchor2_core5/<dataset>/summary.csv`
- 후보: `.../StageC_Focus_anchor2_core5/<dataset>/stageC_candidates.json`

## summary 주요 컬럼
- 앞쪽 고정: `global_best_valid_mrr20, global_best_test_mrr20, model, model_best_valid_mrr20, model_best_test_mrr20`
- Stage C 추가 추적: `parent_profile_id`, `parent_run_phase`
