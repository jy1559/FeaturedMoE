# FMoE_HGR Run Entrypoints

`run/fmoe_hgr`는 `FeaturedMoE_HGR` 전용 트랙입니다.
HGR는 HiR와 비슷한 stage scaffold를 쓰지만, 핵심 축이 `group_router_mode`, `group_top_k`, `stage_merge_mode`이므로
phase 설계도 layout-only보다 routing-first로 둡니다.

## 기본값 철학
- 기본 router: `group_router_mode=per_group`
- 기본 stage merge: `stage_merge_mode=serial`
- 기본 top-k: `group_top_k=0`, `moe_top_k=0`
- 기본 layout: `[1,1,1,1,0]` (`global_pre=1`, `macro=1`, `mid=1`, `micro=1`, `post=0`)
- 기본 capacity: `embedding_size=128`, `d_feat_emb=16`, `d_expert_hidden=160`, `d_router_hidden=64`, `expert_scale=3`
- 기본 batch:
  - ML1M: `train_batch_size=4096`, `eval_batch_size=8192`
  - RetailRocket: `train_batch_size=3072`, `eval_batch_size=6144`

## 스크립트
- `train_single.sh`: 단일 학습(P0 스모크 / 재현)
- `tune_hparam.sh`: 고정된 HGR combo에 대해 LR/WD/dropout/balance 중심 튜닝
- `p1_wide_shallow.sh`: routing/layout combo를 GPU에 분산하고 각 combo 내부에서는 LR/WD만 넓고 얕게 탐색
- `p2_confirm_top.sh`: P1 top result를 동일 combo로 다시 좁은 space에서 재검증
- `p3_dim_batch_combo.sh`: best routing combo를 고정하고 dim/router/batch/expert_scale combo를 GPU 분산 탐색
- `tune_routing.sh`: best 구조 위에서 `group_top_k`, `moe_top_k`, temp/dropout, optional schedule 축 튜닝
- `EXPERIMENT_PLAN_hgr.md`: phase별 설계 배경과 권장 순서

## 권장 phase 흐름
- `P0`: `train_single.sh`로 serial/per_group 스모크
- `P1`: `p1_wide_shallow.sh`
- `P2`: `p2_confirm_top.sh`
- `P3`: `p3_dim_batch_combo.sh`
- `P4`: `tune_routing.sh --mode routing|schedule|combined`

## P1 구성
- 총 `80 combos`
- `40 combos`: base layout `[1,1,1,1,0]` 고정, routing/merge/capacity만 다양화
- `40 combos`: layout 20종을 base capacity로 훑고 `serial/per_group` vs `parallel/hybrid` 비교
- 기본 budget: `4 GPUs x 20 combos`, `epochs=20`, `patience=5`, `max_evals=10`

## 예시
```bash
bash experiments/run/fmoe_hgr/train_single.sh \
  --dataset movielens1m --layout-id 0 --stage-merge-mode serial --group-router-mode per_group --gpu 0 --dry-run

bash experiments/run/fmoe_hgr/p1_wide_shallow.sh \
  --datasets movielens1m --gpus 0,1,2,3 --combos-per-gpu 20 --max-evals 10 --tune-epochs 20 --tune-patience 5 --dry-run

bash experiments/run/fmoe_hgr/p2_confirm_top.sh \
  --datasets movielens1m --gpus 0,1 --topn 4 --source-phase-prefix P1HGR --dry-run

bash experiments/run/fmoe_hgr/p3_dim_batch_combo.sh \
  --dataset movielens1m \
  --parent-result experiments/run/artifacts/results/fmoe_hgr/<best_p2>.json \
  --gpus 0,1 --combos-per-gpu 2 --dry-run

bash experiments/run/fmoe_hgr/tune_routing.sh \
  --dataset movielens1m \
  --parent-result experiments/run/artifacts/results/fmoe_hgr/<best_p3>.json \
  --mode routing --gpu 0 --dry-run
```
