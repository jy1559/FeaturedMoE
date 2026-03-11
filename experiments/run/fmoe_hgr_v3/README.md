# HGRv3 Track

`run/fmoe_hgr_v3`는 `FeaturedMoE_HGRv3` 전용 트랙입니다.

핵심 구조는 다음과 같습니다.
- outer router: hidden-only group routing
- inner router: hidden + group feature interaction
- rule teacher: outer가 아니라 inner clone routing에만 적용
- 비교 모드: `off`, `distill`, `fused_bias`, `distill_and_fused_bias`
- 기본 `expert_scale`: `4`

R0 quick probe:
```bash
cd /workspace/jy1559/FMoE/experiments
bash run/fmoe_hgr_v3/p1_inner_rule_quick.sh \
  --datasets movielens1m \
  --gpus 0,1,2,3 \
  --combos-per-gpu 2 \
  --max-evals 10 \
  --tune-epochs 40 \
  --tune-patience 8
```

지원 스크립트:
- `train_single.sh`: 단일 config 학습
- `tune_hparam.sh`: fixed-anchor hyperopt
- `p1_inner_rule_quick.sh`: `R0` 8-combo weak-distill/top-k quick probe
- `r1_distill_topk.sh`: `distill` 고정 후 `expert_top_k`와 `distill lambda`를 보는 8-combo sweep
- `r1_layout_dim.sh`: `off vs distill`를 `layout/dim` 변화와 같이 보는 8-combo sweep
- `r2_outer_teacher4.sh`: `balance off + outer feature on/off + off/weak/strong distill` 4-combo ablation
- `summary_ML1.{md,csv}`: `artifacts/logs/fmoe_hgr_v3/hparam/R0HGRv3/`에 phase 요약 자동 갱신

구조 설명은 [HGRV3_STRUCTURE.md](/workspace/jy1559/FMoE/experiments/run/fmoe_hgr_v3/HGRV3_STRUCTURE.md) 에 정리했습니다.
