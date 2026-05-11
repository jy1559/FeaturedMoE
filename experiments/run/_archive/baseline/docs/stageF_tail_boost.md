# Stage F Tail-Boost (GRU4Rec / FAME / DuoRec)

## Goal
`lastfm0.03`, `amazon_beauty`에서 낮게 나온 tail 모델(`GRU4Rec`, `FAME`)을 다시 끌어올리고,
`DuoRec`은 Stage E에서 꺾인 구간을 회복/상향하는 목적의 재탐색 stage.

## What We Learned From Stage A~E

### lastfm0.03
- `GRU4Rec`: A `0.1897` -> B `0.2040` -> D `0.2048` -> E `0.1982`
  - 고 LR(대체로 `1e-3~1e-2`) + 더 깊은 층에서 좋아짐.
  - E에서 재-LR/seed 과정에서 일부 감소.
- `FAME`: A `0.2150` -> D `0.2231` -> E `0.2242`
  - 이미 안정적으로 상승. `hidden≈208`, `layers=1`, `heads=2`, `experts=3~6` 계열이 강함.
- `DuoRec`: B `0.2316` -> D `0.2321` -> E `0.2324`
  - `max_len=10`, 중간 LR(`~2e-4~3e-3`) + `contrast=un` 계열에서 매우 안정적.

### amazon_beauty
- `GRU4Rec`: A `0.0088` -> D `0.0197` -> E `0.0118`
  - Stage E parent가 과수축되며 성능 하락. Stage D 강한 설정 복구 필요.
- `FAME`: A `0.0190` -> D `0.0265` -> E `0.0226`
  - 고LR/작은 모델(88-3layer) 쪽이 상대적으로 나음.
- `DuoRec`: C `0.1226`(best) -> E `0.1057`
  - E에서 `hidden=64/layer=3/dropout=0.45` 쪽으로 몰리며 명확히 성능 하락.
  - C의 `B3-C1` 류(`hidden=120,layers=2,max_len=10`) 복구가 필요.

## Stage F Design

### 1) Parent Candidate Widening
Stage F는 parent를 Stage E만 쓰지 않고 아래를 합쳐서 선별함.
- Stage E candidates
- Stage D candidates
- Stage C candidates
- Stage B candidates
- summary 상위 run 복구 후보
- manual recovery anchors (특히 amazon GRU4Rec/FAME/DuoRec)

선별 규칙:
- `Top-K`(기본 3) + `must_keep` anchor 우선
- valid/test/completion/source를 합친 점수로 정렬

### 2) LR Strategy
- 기본은 각 parent LR window 주변 local search
- Stage F profile에서 `lr_mult`, `lr_span_mult`로 미세 이동
- clamp 유지: `[8e-5, 1e-2]`

### 3) Runtime Consistency Fix
- Stage F에서는 `MAX_ITEM_LIST_LENGTH`를 반드시 singleton search로 고정:
  - `++search.MAX_ITEM_LIST_LENGTH=[fixed_max_len]`
  - `++search_space_type_overrides.MAX_ITEM_LIST_LENGTH=choice`
- 목적: 불필요한 max_len 랜덤 탐색 제거, 속도/일관성 개선.

### 4) Profiles (F1~F8)
- `F1`: stable-mid
- `F2`: high-lr push
- `F3`: low-lr safe
- `F4`: regularized outlier
- `F5`: FAME expert boost
- `F6`: DuoRec recover
- `F7`: capacity probe (저정규화/고LR)
- `F8`: long-context probe (`max_len=15`)

기본 탐색 규모(현재 기본값):
- `parents(topk)=3`
- 모델별 profile shortlist 4개
  - GRU4Rec: `F1,F2,F3,F7`
  - FAME: `F1,F2,F5,F7`
  - DuoRec: `F1,F2,F4,F6`
- 따라서 모델/데이터셋당 `12 runs` (seed 1 기준)

## Files
- Runner: `experiments/run/baseline/run_stageF_tail_boost.py`
- Wrapper: `experiments/run/baseline/stageF_tail_boost.sh`

## Run Examples

Dry-run:
```bash
bash experiments/run/baseline/stageF_tail_boost.sh --dry-run
```

Smoke:
```bash
bash experiments/run/baseline/stageF_tail_boost.sh --smoke-test --smoke-max-runs 4 --gpus 0
```

Full (target models only, aggressive default):
```bash
bash experiments/run/baseline/stageF_tail_boost.sh \
  --models gru4rec,fame,duorec \
  --datasets lastfm0.03,amazon_beauty \
  --gpus 0,1,2,3
```

## Expected Output
- logs: `experiments/run/artifacts/logs/baseline/StageF_TailBoost_anchor2_core5/<dataset>/...`
- summary: dataset별 `summary.csv`
- candidates: dataset별 `stageC_candidates.json` (Stage C 엔진 재사용)
