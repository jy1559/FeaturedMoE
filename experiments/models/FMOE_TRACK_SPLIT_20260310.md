# FMoE Track Split 2026-03-10

이 문서는 `FeaturedMoE_v2` 실험 트랙이 flat baseline과 hierarchical router 실험이 섞여 혼동되기 시작한 시점에, 코드를 두 갈래로 분리한 이유와 앞으로의 운영 기준을 적는다.

## 왜 나눴나

- 원래 `FeaturedMoE_v2`는 `flat learned router` 기반의 boundary-explicit Stage-MoE였다.
- 2026-03-09 작업에서 `group -> intra-group` factorized router, feature-spec aux, router distill이 `FeaturedMoE_v2` working tree에 직접 들어갔다.
- 그 결과 같은 `v2` 이름 아래에
  - 과거 flat 결과 (`ML1 0.0982`, `RR 0.2720`)
  - 이후 hierarchical/factorized 결과 (`RR 0.261~0.264`)
  가 함께 섞여 해석이 어려워졌다.

## 새 기준

- `FeaturedMoE_v3`
  - 위치: [FeaturedMoE_v3](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE_v3)
  - runner: [run/fmoe_v3](/workspace/jy1559/FMoE/experiments/run/fmoe_v3)
  - 의미: `pre-fmoe-v2-router-overhaul-20260309` snapshot에서 복원한 flat-router lineage
  - 목적: 원래 `v2`의 strong baseline을 다시 밀고 가는 메인 track

- `FeaturedMoE_v2_HiR`
  - 위치: [FeaturedMoE_v2_HiR](/workspace/jy1559/FMoE/experiments/models/FeaturedMoE_v2_HiR)
  - runner: [run/fmoe_v2_hir](/workspace/jy1559/FMoE/experiments/run/fmoe_v2_hir)
  - 의미: factorized hierarchical router, feature-spec, distill을 실험하던 branch
  - 목적: flat과 별도로 hierarchical routing hypothesis를 검증하는 실험 track

- `FeaturedMoE_v2`
  - 의미: historical mixed track
  - 운영 기준: 새 실험의 canonical target으로 쓰지 않는다. 과거 artifacts 참조용으로만 본다.

## 앞으로 어떻게 볼까

- `v3`는 `flat router`를 유지한 채 개선한다.
  - ML1 non-rule anchor는 `L7 serial`, `MRR@20=0.0982`였다.
  - RR legacy best는 `L16`, moderate dim, `MRR@20=0.2720`였다.
  - 따라서 다음 baseline 복구/개선은 `v3`에서 진행하는 게 맞다.

- `v2_HiR`는 계속 볼 수는 있지만, 범위를 좁혀야 한다.
  - 지금까지 strongest signal은 `expert_top_k=2`였다.
  - 반면 distill은 부차적이었고, factorized 전체 성능은 legacy RR best를 넘지 못했다.
  - 다음 한 번은 `expert_top_k=0/1/2 semantics`와 `group-local clone routing` 위주로만 확인하고, 안 뜨면 `HGR` 쪽에 집중하는 게 낫다.

## 실험 운영 원칙

- flat baseline 관련 로그/결과는 `fmoe_v3` track에만 쌓는다.
- factorized/hierarchical 실험은 `fmoe_v2_hir` track에만 쌓는다.
- 문서에서 과거 `v2` 결과를 인용할 때는 반드시 `legacy-flat`인지 `factorized-router`인지 같이 적는다.
