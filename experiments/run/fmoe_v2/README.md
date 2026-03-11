# FMoE_v2 Run Entrypoints

`run/fmoe_v2`는 `FeaturedMoE_v2` 전용 트랙입니다.
rule-based ablation 전용 엔트리포인트는 `run/fmoe_rule`에서 별도 관리합니다.

## 스크립트
- `train_single.sh`: 단일 학습(P0 스모크/재현)
- `p1_wide_shallow.sh`: 넓고 얕은 P1 스크리닝 (layout/execution 고정 다건, lr/wd 중심)
- `p1_rr_factorized_probe.sh`: RetailRocket용 새 factorized router 기준 blocked-joint P1. 공통 anchor로 layout을 비교하고 top layout에만 dim probe를 더해 LR band를 재파악
- `tune_hparam.sh`: 하이퍼파라미터 탐색(P1/P3, 단일 layout/execution 기준)
- `tune_layout.sh`: layout/execution 축 탐색(P2)
- `tune_schedule.sh`: schedule 축 탐색(P4)
- `p2_dim_batch_combo.sh`: serial/parallel 고정 layout 기반 dim/router/batch 조합 + LR/WD 탐색(P2 확장, OOM 시 batch half 재시도)
- `p2_rr_focus.sh`: RetailRocket용 RR-focused P2. `P1` 상위 serial layout 2~3개를 anchor로 두고 combo를 넓히되 LR/WD를 좁게 탐색
- `p2_rr_factorized_dim.sh`: RetailRocket용 factorized-router P2 narrow. `P1RGI` 상위 layout을 고정하고 dim/batch를 중심으로만 좁게 탐색하되, 일부 seed에는 `expert_top_k` probe와 outlier big-dim combo를 섞음
- `p3_rr_factorized_router.sh`: RetailRocket용 factorized-router P3 narrow. `P2/P1` 상위 layout+dim seed를 고정하고 `expert_top_k`, feature-spec, distill만 짧게 ablation
- `p3_rr_router_teach.sh`: RetailRocket용 RR P3 full-tune. `1 main seed`를 고정하고 `expert_top_k`, feature-spec, distill profile만 12-combo로 비교. `--catalog-profile semantics12`를 주면 `expert_top_k=0/1/2` semantics 비교로 바뀜
- `pipeline_ml1_rr_v2.sh`: ML1M -> RetailRocket 파이프라인
- `final_v2_ml1_rr.sh`: FMoEv2 최종화 계획(ML1 28 + RR 전이 20, spec-aux 축 포함)

## 출력 경로 (artifacts-first)
- Logs: `experiments/run/artifacts/logs/fmoe_v2/*`
- Results: `experiments/run/artifacts/results/fmoe_v2/*.json`
- Timeline: `experiments/run/artifacts/timeline/events.jsonl` (`track=fmoe_v2`)

## 예시
```bash
bash experiments/run/fmoe_v2/p1_wide_shallow.sh --datasets movielens1m,retail_rocket --gpus 0,1 --combos-per-gpu 3 --max-evals 12 --dry-run
bash experiments/run/fmoe_v2/p1_rr_factorized_probe.sh --datasets retail_rocket --gpus 0,1,2,3 --dry-run
bash experiments/run/fmoe_v2/train_single.sh --dataset movielens1m --layout-id 0 --execution serial --gpu 0 --dry-run
bash experiments/run/fmoe_v2/tune_hparam.sh --dataset movielens1m --layout-id 0 --execution serial --gpu 0 --dry-run
bash experiments/run/fmoe_v2/tune_layout.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v2/<p1>.json --gpu 0 --dry-run
bash experiments/run/fmoe_v2/tune_schedule.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v2/<p3>.json --mode alpha --gpu 0 --dry-run
bash experiments/run/fmoe_v2/p2_dim_batch_combo.sh --datasets movielens1m --gpus 0,1 --combos-per-gpu 2 --max-evals 20 --dry-run
bash experiments/run/fmoe_v2/p2_rr_focus.sh --datasets retail_rocket --gpus 0,1 --layout-ids 16,18,7 --combos-per-gpu 3 --dry-run
bash experiments/run/fmoe_v2/p2_rr_factorized_dim.sh --datasets retail_rocket --gpus 0,1,2,3 --dry-run
bash experiments/run/fmoe_v2/p3_rr_factorized_router.sh --datasets retail_rocket --gpus 0,1,2,3 --dry-run
bash experiments/run/fmoe_v2/p3_rr_router_teach.sh --datasets retail_rocket --gpus 0,1,2,3 --dry-run
bash experiments/run/fmoe_v2/p3_rr_router_teach.sh --datasets retail_rocket --gpus 0,1,2,3 --catalog-profile semantics12 --dry-run
bash experiments/run/fmoe_v2/pipeline_ml1_rr_v2.sh --datasets movielens1m,retail_rocket --gpus 0,1 --dry-run
```
