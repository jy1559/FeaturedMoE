# FMoE Result Summary

- generated_at: 2026-03-10T11:52:53.572312
- metric: mrr@20
- total_records: 233

## Best by Dataset

| dataset | model | metric | run_group | run_axis | run_phase | source | timestamp |
|---|---|---:|---|---|---|---|---|
| movielens1m | FeaturedMoE_v2 | 0.1000 | fmoe_rule | hparam | RULE | log_fallback | 2026-03-06T11:39:25.700547 |

## Notes

- artifacts/results JSON을 우선 사용하고, 누락 시 legacy 결과/로그 fallback을 사용한다.
- 채택 기준은 단일 best score이며 재현성 제약을 기본으로 두지 않는다.
