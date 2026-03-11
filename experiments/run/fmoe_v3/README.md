# FMoE_v3 Run Entrypoints

`run/fmoe_v3`는 `FeaturedMoE_v3` 전용 트랙입니다.  
`FeaturedMoE_v3`는 `snapshot: pre-fmoe-v2-router-overhaul-20260309` 시점의 flat-router `FeaturedMoE_v2`를 복원한 baseline track입니다.
rule-based ablation 전용 엔트리포인트는 `run/fmoe_rule`에서 별도 관리합니다.

## 스크립트
- `train_single.sh`: 단일 학습(P0 스모크/재현)
- `p1_wide_shallow.sh`: 넓고 얕은 P1 스크리닝 (layout/execution 고정 다건, lr/wd 중심)
- `tune_hparam.sh`: 하이퍼파라미터 탐색(P1/P3, 단일 layout/execution 기준)
- `tune_layout.sh`: layout/execution 축 탐색(P2)
- `tune_schedule.sh`: schedule 축 탐색(P4)
- `p2_dim_batch_combo.sh`: serial/parallel 고정 layout 기반 dim/router/batch 조합 + LR/WD 탐색(P2 확장, OOM 시 batch half 재시도)
- `p2_rr_focus.sh`: RetailRocket용 RR-focused P2. `P1` 상위 serial layout 2~3개를 anchor로 두고 combo를 넓히되 LR/WD를 좁게 탐색
- `pipeline_ml1_rr_v2.sh`: ML1M -> RetailRocket 파이프라인
- `final_v2_ml1_rr.sh`: FMoEv2 최종화 계획(ML1 28 + RR 전이 20, spec-aux 축 포함)
- `phase_a_ml1_legacy_repro.sh`: ML1 non-rule legacy(`0.0982`) 재현용 exact anchor
- `phase_b_ml1_router_structure.sh`: flat router 입력/구조 6종 비교 (`flat_hidden_group_clone12` 포함)
- `phase_c_ml1_distill_modes.sh`: `flat_hidden_group_clone12` 기준 distill 4종 + comparator 비교
- `phase_bc_ml1_quick.sh`: `B(구조)`와 `C(distill)`를 2 GPU씩 동시에 quick run
- `phase_b_ml1_router_narrow.sh`: `group_plus_clone` distill 고정 + 약한 router 2개 제거 후 router 4종만 좁게 재비교
- `phase_c_ml1_distill_narrow.sh`: `flat_legacy` router 고정 + plain/clone/group+clone/hybrid 4종만 좁게 재비교
- `phase_bc_ml1_narrow.sh`: 위 narrow `B/C`를 2 GPU씩 동시에 실행
- `phase_d_ml1_routing_semantics.sh`: `moe_top_k=0/2/4` semantics 비교
- `phase_e_ml1_dim_robustness.sh`: winning recipe의 dim robustness 확인
- `phase_g_ml1_expert_scale_lr.sh`: ML1 legacy anchor 고정 + `expert_scale`별 좁은 LR 재탐색
- `phase_f_rr_transfer.sh`: ML1 winner를 RetailRocket에 전이
- `phase_rr_rule_sanity.sh`: RR rule/hybrid comparator quick sanity
- `ROUTER_PLAN_v3.md`: flat-router 발전 실험 순서와 기본 정책

## 출력 경로 (artifacts-first)
- Logs: `experiments/run/artifacts/logs/fmoe_v3/*`
- Results: `experiments/run/artifacts/results/fmoe_v3/*.json`
- Timeline: `experiments/run/artifacts/timeline/events.jsonl` (`track=fmoe_v3`)

## 예시
```bash
bash experiments/run/fmoe_v3/p1_wide_shallow.sh --datasets movielens1m,retail_rocket --gpus 0,1 --combos-per-gpu 3 --max-evals 12 --dry-run
bash experiments/run/fmoe_v3/train_single.sh --dataset movielens1m --layout-id 0 --execution serial --gpu 0 --dry-run
bash experiments/run/fmoe_v3/tune_hparam.sh --dataset movielens1m --layout-id 0 --execution serial --gpu 0 --dry-run
bash experiments/run/fmoe_v3/tune_layout.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v3/<p1>.json --gpu 0 --dry-run
bash experiments/run/fmoe_v3/tune_schedule.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v3/<p3>.json --mode alpha --gpu 0 --dry-run
bash experiments/run/fmoe_v3/p2_dim_batch_combo.sh --datasets movielens1m --gpus 0,1 --combos-per-gpu 2 --max-evals 20 --dry-run
bash experiments/run/fmoe_v3/p2_rr_focus.sh --datasets retail_rocket --gpus 0,1 --layout-ids 16,18,7 --combos-per-gpu 3 --dry-run
bash experiments/run/fmoe_v3/pipeline_ml1_rr_v2.sh --datasets movielens1m,retail_rocket --gpus 0,1 --dry-run
bash experiments/run/fmoe_v3/phase_a_ml1_legacy_repro.sh --gpu 0 --dry-run
bash experiments/run/fmoe_v3/phase_b_ml1_router_structure.sh --datasets movielens1m --gpus 0,1,2,3 --dry-run
bash experiments/run/fmoe_v3/phase_c_ml1_distill_modes.sh --base-router-design flat_hidden_group_clone12 --dry-run
bash experiments/run/fmoe_v3/phase_bc_ml1_quick.sh --b-gpus 0,1 --c-gpus 2,3 --dry-run
bash experiments/run/fmoe_v3/phase_b_ml1_router_narrow.sh --gpus 0,1 --dry-run
bash experiments/run/fmoe_v3/phase_c_ml1_distill_narrow.sh --gpus 2,3 --dry-run
bash experiments/run/fmoe_v3/phase_bc_ml1_narrow.sh --b-gpus 0,1 --c-gpus 2,3 --dry-run
bash experiments/run/fmoe_v3/phase_d_ml1_routing_semantics.sh --router-design flat_hidden_group_clone12 --distill-mode clone_only --dry-run
bash experiments/run/fmoe_v3/phase_e_ml1_dim_robustness.sh --router-design flat_hidden_group_clone12 --distill-mode clone_only --dry-run
bash experiments/run/fmoe_v3/phase_g_ml1_expert_scale_lr.sh --gpus 0,1,2,3 --scales 1,3,5,8 --dry-run
bash experiments/run/fmoe_v3/phase_f_rr_transfer.sh --router-design flat_hidden_group_clone12 --distill-mode clone_only --dry-run
bash experiments/run/fmoe_v3/phase_rr_rule_sanity.sh --catalog-profile rr_rule8 --dry-run
```
