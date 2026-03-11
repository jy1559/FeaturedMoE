# FeaturedMoE_v2

`FeaturedMoE_v2`는 기존 `FeaturedMoE`를 유지한 상태에서, 구조/설정/실험 트랙을 분리해 새로 설계한 모델입니다.

## 목표
- v1 코드와 실험 자산을 건드리지 않고 v2를 독립 운영
- stage 경계를 `layout object`로 명시 (`pass_layers`, `moe_blocks`)
- `serial`, `parallel`, `parallel + repeat`를 같은 실행 엔진으로 지원
- 파라미터 체계를 단순화하고 런 트랙(`run/fmoe_v2`)을 분리

## 핵심 개념
- 경계 명시형 layout
  - `pass_layers`: MoE 없이 attention만 통과하는 레이어 수
  - `moe_blocks`: `[1-layer attention + 1 MoE]` 반복 횟수
- 실행 모드
  - `serial`: macro -> mid -> micro 순차 적용
  - `parallel`: 동일 base hidden에서 stage branch를 독립 계산 후 merge
- 라우터 백엔드
  - `router_impl=learned|rule_soft`
  - `router_impl_by_stage`로 stage별 혼합 라우팅 지원
  - `rule_router.{n_bins,feature_per_expert,...}`로 rule-soft 규칙 제어
- 병렬 병합 보조 손실(옵션)
  - `fmoe_v2_stage_merge_aux_enable`
  - `fmoe_v2_stage_merge_aux_lambda_scale`

## 파일 맵
- `featured_moe_v2.py`: RecBole entry class (`FeaturedMoE_V2`)
- `config_schema.py`: v2 키 검증/removed key 에러 처리
- `layout_schema.py`: layout 파싱/검증/요약 유틸
- `stage_modules.py`: stage branch와 MoE 블록 실행 단위
- `stage_executor.py`: serial/parallel 통합 실행기
- `merge_router.py`: parallel stage merge router
- `schedule.py`: alpha/temp/top-k 스케줄 해석
- `losses.py`: expert aux + merge aux
- `feature_config.py`: v1 feature 정의 thin wrapper

## 바로가기
- [migration_v1_to_v2.md](migration_v1_to_v2.md)
- [quick_guide.md](quick_guide.md)
- [deep_dive.md](deep_dive.md)
