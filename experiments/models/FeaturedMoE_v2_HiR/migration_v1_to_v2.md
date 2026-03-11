# FeaturedMoE v1 -> v2 마이그레이션

## 범위
이 문서는 `FeaturedMoE`(v1)에서 `FeaturedMoE_v2`로 전환할 때 필요한 키/실행/주의사항을 정리합니다.

## 브레이킹 변경
- v2는 별도 모델명 사용: `model=featured_moe_v2` (내부 클래스 `FeaturedMoE_V2`)
- layout 표현이 5-int 벡터에서 object 스키마로 변경
- v1 일부 키는 v2에서 제거(입력 시 명시적 에러)

## 키 매핑
| v1 키/개념 | v2 키/개념 | 설명 |
|---|---|---|
| `arch_layout_catalog` + `arch_layout_id` | `fmoe_v2_layout_catalog` + `fmoe_v2_layout_id` | v2는 object layout만 사용 |
| `stage_moe_repeat_after_pre_layer` | `stages.<stage>.moe_blocks` | 반복 여부를 layout에 직접 포함 |
| `n_pre_*`, `n_post_layer` | `global_pre_layers`, `global_post_layers`, `stages.<stage>.pass_layers` | MoE 비적용 구간을 명시 |
| (없음) | `fmoe_stage_execution_mode` | `serial | parallel` 실행 오버라이드 |
| (없음) | `router_impl`, `router_impl_by_stage`, `rule_router.*` | learned/rule_soft 및 stage별 혼합 라우팅 |
| (없음) | `fmoe_v2_parallel_stage_gate_*` | parallel branch merge 제어 |
| (없음) | `fmoe_v2_stage_merge_aux_*` | merge aux loss 제어 |
| `hidden_size` 중심 | `embedding_size` 중심 | v2는 embedding 단일 차원 키 정책 |

## v2 미지원(제거) 키
- `stage_moe_repeat_after_pre_layer`
- `n_pre_layer`, `n_pre_macro`, `n_pre_mid`, `n_pre_micro`, `n_post_layer`
- `alpha_warmup_steps`, `temperature_warmup_steps`, `moe_top_k_warmup_steps`
- `fmoe_schedule_log_every`

## v2 layout 예시
```yaml
fmoe_v2_layout_catalog:
  - id: L0
    execution: serial
    global_pre_layers: 1
    global_post_layers: 0
    stages:
      macro: {pass_layers: 1, moe_blocks: 1}
      mid:   {pass_layers: 1, moe_blocks: 1}
      micro: {pass_layers: 1, moe_blocks: 1}
```

## run 전환
- v1: `experiments/run/fmoe/*`
- v2: `experiments/run/fmoe_v2/*`

예시:
```bash
bash experiments/run/fmoe_v2/tune_hparam.sh --dataset movielens1m --gpu 0 --dry-run
```

## 주의사항
- v2는 v1 alias/deprecated 키 자동 호환을 제공하지 않음
- 결과/로그는 `track=fmoe_v2` 경로로 분리 저장
