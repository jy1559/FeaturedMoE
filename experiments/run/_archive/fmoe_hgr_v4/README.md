# HGRv4 Track

`run/fmoe_hgr_v4`는 `FeaturedMoE_HGRv4` 전용 트랙입니다.

핵심 구조는 다음과 같습니다.
- outer router: old HGR 계열로 복구한 feature-aware group routing
- outer/inner router design: `legacy_concat | group_factorized_interaction`
- 기본값: `outer=legacy_concat`, `inner=legacy_concat`
- inner router: hidden + group feature interaction
- inner teacher: `group_stat_soft`
  - group feature ratio의 `mean/std/max/min/range/peak` 통계로 expert logits 생성
- 기본 `expert_scale`: `4`
- 기본 비교축: `off / weak distill / main distill / strong distill`

R0 distill 4-combo:
```bash
cd /workspace/jy1559/FMoE/experiments
bash run/fmoe_hgr_v4/r0_distill4.sh \
  --datasets movielens1m \
  --gpus 0,1,2,3 \
  --combos-per-gpu 1 \
  --max-evals 10 \
  --tune-epochs 40 \
  --tune-patience 8
```

지원 스크립트:
- `train_single.sh`: 단일 config 학습
- `tune_hparam.sh`: fixed-anchor hyperopt
- `r0_distill4.sh`: `layout15 + serial + hybrid outer`에서 distillation 수준 4개 비교

구조 설명은 [HGRV4_STRUCTURE.md](/workspace/jy1559/FMoE/experiments/run/fmoe_hgr_v4/HGRV4_STRUCTURE.md) 에 정리했습니다.
