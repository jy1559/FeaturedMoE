# FMoE_HGR Run Entrypoints

`run/fmoe_hgr`는 `FeaturedMoE_HGR` 전용 트랙입니다.
HGR는 HiR와 비슷한 stage scaffold를 쓰지만, 핵심 축이 `group_router_mode`, `group_top_k`, `stage_merge_mode`이므로
phase 설계도 layout-only보다 routing-first로 둡니다.

## 기본값 철학
- 기본 router: `group_router_mode=per_group`
- 기본 stage merge: `stage_merge_mode=serial`
- 기본 top-k: `group_top_k=0`, `expert_top_k=1`
- 기본 layout: `[1,0,1,0,1,0,1,0]` (`global_pre=1`, macro/mid/micro에 MoE 1회씩)
- 기본 capacity: `embedding_size=128`, `d_feat_emb=16`, `d_expert_hidden=160`, `d_router_hidden=64`, `expert_scale=3`
- 기본 batch:
  - ML1M: `train_batch_size=4096`, `eval_batch_size=8192`
  - RetailRocket: `train_batch_size=3072`, `eval_batch_size=6144`

## 스크립트
- `train_single.sh`: 단일 학습(P0 스모크 / 재현)
- `tune_hparam.sh`: 고정된 HGR combo에 대해 LR/WD/dropout/balance 중심 튜닝
- `p1_wide_shallow.sh`: routing/layout combo를 GPU에 분산하고 각 combo 내부에서는 LR/WD만 넓고 얕게 탐색
- `p2_confirm_top.sh`: P1 top result를 동일 combo로 다시 좁은 space에서 재검증
- `p2_dim_focus.sh`: `P15HGR` 이후 상위 layout/route만 유지하고 dim/batch/LR coupling을 8 combo로 빠르게 확인
- `p3_router_teach.sh`: `P2HGR` 이후 상위 anchor 2개만 고정하고 distill/spec/group_top_k 축으로 router teaching을 비교
- `p3_dim_batch_combo.sh`: 예전 capacity 확장용 러너. 현재 기본 권장 흐름에서는 사용하지 않음
- `tune_routing.sh`: best 구조 위에서 `group_top_k`, `moe_top_k`, temp/dropout, optional schedule 축 튜닝
- `EXPERIMENT_PLAN_hgr.md`: phase별 설계 배경과 권장 순서

## 권장 phase 흐름
- `P0`: `train_single.sh`로 serial/per_group 스모크
- `P1`: `p1_wide_shallow.sh`
- `P1.5`: `p15_layout_focus.sh`
- `P2`: `p2_dim_focus.sh`
- `P3`: `p3_router_teach.sh`
- `Final`: `tune_hparam.sh` 또는 `tune_routing.sh`로 narrow squeeze

## P1 구성
- 총 `32 combos`
- `8 combos`: layout/depth와 dim/capacity를 joint하게 묶은 arch anchor 8종
- `4 combos/anchor`: `serial+per_group`, `serial+per_group+alpha_cold`, `serial+hybrid`, `parallel+per_group+alpha_cold`
- 기본 budget: `4 GPUs x 8 combos`, `epochs=25`, `patience=5`, `max_evals=8`
- 기본 LR space: `2e-4,5e-4,1e-3,2e-3,4e-3,8e-3,1.5e-2`

## P1.5 구성
- 총 `24 combos`
- `8` layout anchors: `0, 5, 10, 11, 15, 16, 21` 중심, dim은 `layout0` anchor/wide와 `layout21` heavy control 정도만 변주
- `3 combos/anchor`: `serial+per_group`, `serial+per_group+alpha_cold`, `serial+hybrid`
- 고정 aux: `group_balance_lambda=0.001`, `intra_balance_lambda=0.001`, `group_feature_spec_aux_lambda=1e-4`
- 좁은 search: `lr=5e-4~3.2e-3`, `wd=0~3e-5`, `dropout=0.08/0.10/0.12`, `balance=0.0015/0.003/0.006`

## Post-P15 흐름
- 현재 `P15HGR` readout 기준 main layout은 `15`, strong control은 `16`, heavy control은 `21`
- route는 `serial + hybrid`를 main으로 두고, `serial + per_group`만 control로 둠
- `alpha_cold`는 현재 채택 신호가 약해서 `P2` main에서 제외

### P2 dim-focus
- 총 `8 combos = 4 GPUs x 2 combos`
- main: `L15 + serial + hybrid`
- control: `L15 + serial + per_group`, `L16 + serial + hybrid`, `L21 + serial + per_group`
- dim bucket:
  - `128/16/160/64/scale3/bs4096`
  - `160/16/224/96/scale3/bs3072`
  - `160/16/256/112/scale3/bs2048`
  - `192/16/320/128/scale2/bs1536`
  - `224/16/512/160/scale1/bs1024`

### P3 router-teaching
- 총 `16 combos = 4 GPUs x 4 combos`
- anchor 2개만 사용:
  - `A0 = L15 + serial + hybrid + 128/16/160/64/scale3/bs4096`
  - `A1 = L15 + serial + hybrid + 160/16/256/112/scale3/bs2048`
- 구조는 고정하고 아래 teaching profile만 바꿈:
  - baseline
  - weak/main/strong distill
  - longer distill
  - sharper temperature
  - macro+mid feature specialization
  - `group_top_k=2`
- LR/WD는 anchor별로 다르게 둠:
  - `A0`: `lr=9e-4~2.8e-3`, `wd=1e-6~2e-5`
  - `A1`: `lr=3.5e-4~1.0e-3`, `wd=5e-6~5e-5`

### 이후 phase
- `P3` 이후에는 best distill/profile 하나를 parent로 두고 optimizer-only final squeeze를 권장합니다.
- `expert_use_feature=true`나 schedule on 같은 축은 이 final phase에서 single-control 수준으로만 붙이는 쪽이 안전합니다.

## 예시
```bash
bash experiments/run/fmoe_hgr/train_single.sh \
  --dataset movielens1m --layout-id 0 --stage-merge-mode serial --group-router-mode per_group --gpu 0 --dry-run

bash experiments/run/fmoe_hgr/p1_wide_shallow.sh \
  --datasets movielens1m --gpus 0,1,2,3 --combos-per-gpu 8 --max-evals 8 --tune-epochs 25 --tune-patience 5 --dry-run

bash experiments/run/fmoe_hgr/p15_layout_focus.sh \
  --datasets movielens1m --gpus 0,1,2,3 --combos-per-gpu 6 --max-evals 8 --tune-epochs 25 --tune-patience 5 --dry-run

bash experiments/run/fmoe_hgr/p2_dim_focus.sh \
  --datasets movielens1m --gpus 0,1,2,3 --combos-per-gpu 2 --max-evals 6 --tune-epochs 40 --tune-patience 8 --dry-run

bash experiments/run/fmoe_hgr/p3_router_teach.sh \
  --dataset movielens1m --gpus 0,1,2,3 --combos-per-gpu 4 --max-evals 10 --tune-epochs 100 --tune-patience 15 --dry-run

bash experiments/run/fmoe_hgr/tune_routing.sh \
  --dataset movielens1m \
  --parent-result experiments/run/artifacts/results/fmoe_hgr/<best_p2>.json \
  --mode routing --gpu 0 --dry-run
```
