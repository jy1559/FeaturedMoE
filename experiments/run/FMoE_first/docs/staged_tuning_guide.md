# FMoE_N4 Staged Tuning Guide (A12-fixed)

## 1) 목표
- `fmoe_n4`는 baseline식 A->B->C->D 파이프라인을 쓰되, 구조는 `fmoe_n3`의 A12를 고정한다.
- 즉 아키텍처 실험이 아니라, 하이퍼파라미터 중심 실험이다.
- 핵심 튜닝 축:
  - capacity/dim: `embedding_size`, `d_ff`, `d_expert_hidden`, `d_router_hidden`
  - feature/capacity coupling: `d_feat_emb`, `expert_scale`
  - regularization/optimization: `learning_rate`, `fixed_weight_decay`, `fixed_hidden_dropout_prob`
  - stage dropout: `stage_family_dropout_prob`, `stage_feature_dropout_prob`

## 2) A12 고정 규칙
`fmoe_n4` track에서 `featured_moe_n3_tune` 실행 시 아래를 강제한다.

- `stage_router_wrapper={macro:w5_exd,mid:w5_exd,micro:w5_exd}`
- `layer_layout=[attn, macro_ffn, mid_ffn, attn, micro_ffn]`
- `stage_router_granularity={macro:session,mid:session,micro:token}`
- `stage_feature_dropout_scope={macro:token,mid:token,micro:token}`
- `bias_mode=none`, `rule_bias_scale=0.0`, `feature_group_bias_lambda=0.0`

즉 A12 레이아웃/라우터 계열은 고정하고, 용량/정규화/학습축만 탐색한다.

## 3) Stage 설계 (기본: 2일 scout)

기본 승급/변이 폭:
- Stage A 구조 후보: `8`개 x LR grid `2`개
- 승급: `A->4`, `B->2`, `C->1`, `D->1`
- 변이: `C`는 parent당 `2`개, `D`도 parent당 `2`개

예산:
- 기본 `budget-profile=fast` 사용
- fast 프로필 기준: `A(1/12/3)`, `B(20/45/5)`, `C(14/70/7)`, `D(8/90/9)`
- `final-seeds` 기본은 `1`로 단일 seed 평가
- 표기 순서: `max_evals/epochs/patience`

의도:
- D까지는 수행하되, "최종 확정"보다 "감 잡기"를 우선하는 scout 라운드
- scout 결과를 기반으로 마지막 주에 선택 구간만 deep 재탐색

### Stage A: Broad Capacity Screening
목표:
- A12 고정 상태에서 폭넓은 용량 family를 빠르게 스크리닝.

주요 튜닝:
- 구조/용량 후보 조합:
  - `max_len` in `[20,30,40,50]`
  - `embedding_size` in `[96,128,160,192,256]`
  - `d_ff` in `[192,256,320,384,512]`
  - `d_expert_hidden` in `[96,128,160,192,256]`
  - `d_router_hidden` in `[48,64,80,96,128]`
  - `d_feat_emb` in `[8,12,16,24]`
  - `expert_scale` in `[2,3,4]`
  - `num_heads` in `[2,4]`
- LR grid(기본): `1.6e-4,5e-4`
- dropout/wd는 후보마다 anchor 값으로 고정 후 LR만 비교.

### Stage B: Optimization + Small Capacity Window
목표:
- A 상위 후보에서 연속축 최적화와 소폭 capacity 미세조정.

주요 튜닝:
- 연속축:
  - `learning_rate` (loguniform)
  - `weight_decay`/`fixed_weight_decay` (loguniform)
  - `fixed_hidden_dropout_prob` (uniform)
- 이산 local choice:
  - `d_feat_emb`, `expert_scale`, `d_router_hidden`
- Stage A에서 지나친 구조 이동 없이 optimizer/regularization 정렬을 우선.

### Stage C: Local Architecture Mutation + Re-Search
목표:
- B 상위권 주변에서 dim/capacity를 다시 확장 탐색.

주요 튜닝:
- 변이 + 탐색:
  - `MAX_ITEM_LIST_LENGTH`
  - `embedding_size`/`hidden_size`
  - `d_ff`, `d_expert_hidden`, `d_router_hidden`
  - `d_feat_emb`, `expert_scale`
- 연속축(`lr`, `wd`, `dropout`)은 계속 탐색.
- `stage_family_dropout_prob`, `stage_feature_dropout_prob`도 choice로 함께 탐색.

### Stage D: Final Polish (Scout)
목표:
- C 상위 후보를 빠르게 1개로 압축.

주요 튜닝:
- 좁은 범위 미세조정:
  - `MAX_ITEM_LIST_LENGTH`
  - `d_feat_emb`, `expert_scale`, `d_router_hidden`
  - `lr/wd/dropout` 정밀 재탐색
- `--final-seeds` 기본 `1`.
- 막판 확정 라운드에서만 `--final-seeds 1,2,3` 또는 그 이상으로 확장 권장.

## 4) 실행
- 전체 실행:
  - `bash experiments/run/fmoe_n4/run_all_stages.sh`
- Stage별 실행:
  - `bash experiments/run/fmoe_n4/stageA_fmoe_n4.sh`
  - `bash experiments/run/fmoe_n4/stageB_fmoe_n4.sh`
  - `bash experiments/run/fmoe_n4/stageC_fmoe_n4.sh`
  - `bash experiments/run/fmoe_n4/stageD_fmoe_n4.sh`

자주 조정하는 환경변수:
- `GPU_LIST=0,1,2,3`
- `BUDGET_PROFILE=fast`
- `STAGE_A_STRUCT_COUNT=8`
- `PROMOTE_A_TO_B=4`, `PROMOTE_B_TO_C=2`, `PROMOTE_C_TO_D=1`
- `FINAL_SEEDS=1`

막판 deep 재탐색 예시:
- `BUDGET_PROFILE=deep`
- `STAGE_A_STRUCT_COUNT=16`
- `PROMOTE_A_TO_B=6`, `PROMOTE_B_TO_C=3`, `PROMOTE_C_TO_D=2`
- `FINAL_SEEDS=1,2,3`

## 5) 산출물 경로
- 로그 루트:
  - `experiments/run/artifacts/logs/fmoe_n4/ABCD_A12_hparam_v1/`
- Stage별:
  - `stages/stageA|stageB|stageC|stageD/{summary.csv,promotion.csv,leaderboard.csv,manifest.json}`
- 데이터셋 누적:
  - `<dataset>/summary.csv`

## 6) 평가 규칙
- `feature_mode=full_v4`
- `exclude_unseen_target_from_main_eval=true`
- `log_unseen_target_metrics=true`
- main(seen) + cold(unseen) 지표 동시 저장
