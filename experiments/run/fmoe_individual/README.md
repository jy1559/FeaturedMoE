# FMoE Individual Track

`run/fmoe_individual`는 `FeaturedMoE_Individual` 전용 트랙입니다.

핵심 구조:
- outer router: stage feature 개별 항목 top-k=4
- inner router: feature별 expert_scale=4 dense softmax
- 기본 execution: serial
- 기본 layout: macro+mid only, 4개 catalog

지원 스크립트:
- `tune_hparam.sh`: layout/dim 고정 1개 combo용 hyperopt wrapper
- `r1_layout_dim.sh`: `dim 3종 x layout 4종 = 12개` hyperopt job을 GPU round-robin으로 실행
