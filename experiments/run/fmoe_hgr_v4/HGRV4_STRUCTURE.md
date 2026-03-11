# HGRv4 Structure

## Why HGRv4
- `HGRv3`는 outer가 너무 거친 summary feature만 써서 group 선택력이 약했습니다.
- inner teacher도 `mean_ratio` 하나만 써서 clone semantics를 충분히 못 밀어줬습니다.
- `HGRv4`는 outer를 old HGR 쪽으로 복구하고, inner teacher를 richer stat-soft rule로 교체합니다.

## Information Flow
1. `outer group router`
- 입력: stage hidden + stage feature summary
- 방식: `stage_wide / per_group / hybrid`
- 기본: `hybrid`
- router design: `legacy_concat` 기본, 필요 시 `group_factorized_interaction`
- 출력: 4개 group logits / weights

2. `inner learned router`
- 입력: hidden + 해당 group feature
- router design: `legacy_concat` 기본, 필요 시 `group_factorized_interaction`
- 출력: group 내부 `expert_scale`개 expert logits

3. `inner stat-soft teacher`
- 입력: 해당 group feature
- ratio 변환 후 `mean/std/max/min/range/peak` 계산
- expert logits:
  - `expert0`: low-flat
  - `expert1`: mid-flat
  - `expert2`: high-flat
  - `expert3`: peaky/contrast

4. `final intra routing`
- `off`: learned logits만 사용
- `distill`: teacher는 loss로만 사용
- `fused_bias`: teacher logits를 learned logits에 직접 더함
- `distill_and_fused_bias`: 둘 다 사용

## HGR vs HGRv3 vs HGRv4
- `HGR`: feature-aware outer가 강점, teacher 없음
- `HGRv3`: outer hidden-only, inner bin teacher
- `HGRv4`: feature-aware outer 복구 + inner stat-soft teacher

## Quick Probe
- `layout=15`
- `serial`
- `group_router_mode=hybrid`
- `expert_scale=4`
- 4-combo distill level 비교:
  - `off`
  - `weak distill`
  - `main distill`
  - `strong distill`
