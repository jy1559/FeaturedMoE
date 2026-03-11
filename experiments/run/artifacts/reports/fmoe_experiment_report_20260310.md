# FMoE 실험 변천사 및 결과 정리 보고서

- 작성 기준 시각: 2026-03-10 06:02 UTC
- 기준 지표: `MRR@20`
- 운영 원칙: `movielens1m -> retail_rocket` 순서로 anchor를 잡고, 단일 best score로 채택 여부를 판단
- 수치 소스 우선순위: `experiment_overview.md` -> `results/*.json`의 `trials[].mrr@20` 최고값 -> raw log

## 1페이지 요약

이번 실험의 출발점은 `FMoE_v2`의 flat한 mainline anchor였다. ML1M에서는 `serial + L7` 계열이 빠르게 안정적인 기준선을 만들었고, RetailRocket에서는 `serial + L16 + E128/F24/H160/R64` 조합이 `MRR@20=0.2720`으로 최종 주력 anchor가 됐다.  
하지만 코드와 구조는 고정돼 있지 않았다. `v2`는 초반 flat learned router 중심에서 출발해, `rule hybrid`, `factorized router`, `distill/spec` 아이디어를 점점 흡수하는 쪽으로 바뀌었다.  
`HiR2`, `ProtoX`는 “상위 decision layer를 먼저 두면 더 잘 될까?”를 본 가지였지만 현재 수치상 실패에 가깝다. 반면 `HGR`은 계층형 routing을 제일 그럴듯하게 만들었고, `HGRv3`는 그 구조를 inner-teacher 중심으로 다시 다듬는 현재 진행형 브랜치다.  
ML1M 한정으로는 `rule R1`이 raw log 기준 `0.1000`으로 가장 높았고, `flat v3` rollback baseline도 `0.0969`로 꽤 강했다. 즉, feature semantics는 완전히 틀린 게 아니라 “어디에 얼마나 강하게 넣을지”가 핵심이었다.  
지금까지의 결론은 분명하다. 당장 채택은 `FMoE_v2`이고, `Rule/HGR/HGRv3/v3`는 “왜 되는지”와 “어떻게 supervision을 넣을지”를 설명해 주는 실험들이다.

### 데이터셋별 branch 최고 성능

| dataset | `fmoe_v2` | `fmoe_v3` | `fmoe_rule` | `fmoe_hgr` | `fmoe_hgr_v3` | `fmoe_protox` | `fmoe_hir2` | 핵심 해석 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `movielens1m` | 0.0986 | 0.0969 | **0.1000** | 0.0958 | 0.0933 | 0.0838 | 0.0751 | ML1M에서는 `R1` hybrid rule이 가장 높고, `flat v3` legacy baseline도 의외로 강하다. |
| `retail_rocket` | **0.2720** | - | 0.2625 | - | - | - | - | RR 전이에서는 여전히 `fmoe_v2` mainline이 분명한 우세다. |

### 무엇이 먹혔고 무엇이 안 먹혔는가

1. `serial` 실행과 강한 layout anchor는 먹혔다. ML1M에서는 `L7`, RR에서는 `L16`이 가장 안정적으로 상위권을 유지했다.
2. feature semantics 자체는 틀리지 않았다. `R1`과 `flat v3 legacy`가 이를 보여 줬다. 다만 pure replacement(`R0`)나 무거운 상위 allocator(`HiR2`, `ProtoX`)는 실패했다.
3. hierarchy도 무조건 답은 아니었다. `HGR`는 개선 흐름이 있었지만 mainline을 못 넘었고, `HGRv3`는 현재까지 `off`가 여전히 우세하다. `flat v3`에서는 `hidden-only`가 여전히 낮고, `group_only`는 plain보다 아주 소폭 높지만 legacy baseline은 넘지 못했다. distillation/fusion은 아직 “아이디어는 맞는데 세팅은 더 봐야 하는” 단계다.

> 메모: `fmoe_rule`의 ML1M `0.1000`은 raw log에서 두 번 확인되는 값이다. 초기 `R1 C1` run의 결과 JSON은 generic timestamp 파일이 다른 run과 충돌/덮어쓰기된 흔적이 있어 `experiment_overview.md`에는 `0.0988`까지만 남아 있다. 발표 문서에서는 `best_by_dataset.json`과 raw log를 근거로 `0.1000`을 최종 best로 채택한다.

## 발표 순서 제안

이번 발표는 날짜 나열보다 `무엇이 부족해 보여서 다음 구조를 열었는가`의 순서가 더 좋다. 아래 순서대로 가면 `v2 flat anchor -> 실패한 가지 -> 계층형 보완 -> rule 확인 -> distill/fusion -> 최신 진행 상황` 흐름이 자연스럽다.

### Slide 1. 전체 결론부터

- 한 줄 메시지: 지금 채택은 `FMoE_v2`, 하지만 그 결론에 도달하기까지 `rule`, `hierarchy`, `distill/fusion`을 차례로 확인했다.
- 보여줄 숫자:
  - ML1M: `rule R1 0.1000 > fmoe_v2 0.0986 > flat v3 legacy 0.0969 > HGR 0.0958 > HGRv3 0.0933`
  - RR: `fmoe_v2 0.2720 > rule R1 0.2625`

### Slide 2. 출발점: 초반 `FMoE_v2`는 flat anchor를 만드는 실험이었다

- 2026-03-05 `P1S/P2DB`에서 `serial + L7`가 빠르게 anchor가 됨
- ML1M best가 `0.0982`, later `FINALV2`가 `0.0986`
- 말할 포인트: 처음 목표는 “잘 설명되는 구조”가 아니라 “일단 강한 anchor를 하나 만드는 것”이었다.

### Slide 3. 그런데 v2 코드도 계속 바뀌었다

- 초반 anchor 단계는 사실상 flat learned router 중심이었다.
- 후반으로 갈수록 `router_impl_by_stage`, `rule_soft`, RR용 `group_factorized_interaction`, `distill/spec`가 코드에 들어왔다.
- 2026-03-10의 `flat v3`는 이 변화 때문에 다시 만든 rollback control이다.
- 말할 포인트: `v2`는 하나의 고정 모델이 아니라, 좋은 아이디어를 계속 흡수한 evolving mainline이었다.

### Slide 4. 첫 번째 실패 가지: `HiR2`, `ProtoX`

- 질문: 상위 allocator나 prototype latent를 먼저 만들면 expert routing이 더 쉬워질까
- 결과:
  - `HiR2 0.0751`
  - `ProtoX 0.0838`
- 말할 포인트: 상위 decision layer를 하나 더 두는 것만으로는 안 됐다. 구조는 깔끔했지만 optimization burden이 너무 컸다.

### Slide 5. 그래서 계층형을 다시 제대로 해 본 게 `HGR`

- 의도: flat router를 `group -> clone`으로 분해해 outer/inner 역할을 나누자
- 결과: `P1 0.0946 -> P1.5 0.0937 -> P2 0.0956 -> P3 0.0958`
- 말할 포인트: 이 브랜치는 mainline을 넘진 못했지만 “hierarchy를 어디에 두고 distill/spec를 어디에 걸어야 하는지”를 제일 잘 보여 줬다.

### Slide 6. 그런데 HGR도 내 의도와 완전히 같진 않았다

- outer group routing 쪽 의미가 너무 강했고, rule teacher도 outer group에 걸려 있었다.
- 즉, 진짜로 보고 싶었던 “inner clone semantics”보다 outer supervision 비중이 더 컸다.
- 말할 포인트: 그래서 다음 단계는 HGR를 버리는 게 아니라, teacher 위치를 바꾸는 쪽이었다.

### Slide 7. 그 보완판이 `HGRv3`, 그리고 먼저 `off`를 본 이유

- `HGRv3`는 outer를 hidden-only로 단순화하고, teacher를 inner clone routing으로 옮겼다.
- 이때 바로 distill부터 보는 대신 `off`를 같이 둔 이유는 “구조 개선”과 “teacher 효과”를 분리하기 위해서다.
- 현재 확인된 숫자:
  - `R0 off` best `0.0933`
  - `R0 distill` best `0.0922`
  - `R1` 후속에서도 `off 0.0931 > weak distill 0.0919`
- 말할 포인트: 지금까지는 구조 보완 자체는 의미가 있지만, distill/fusion이 바로 이득을 준다고 말할 단계는 아니다.

### Slide 8. 반대로 `Rule`은 다른 질문에 답했다

- 질문: 애초에 feature로 expert를 고르는 MoE가 이상한 건가
- 결과:
  - `R0` pure rule `0.0752`
  - `R1` hybrid rule raw log `0.1000`
- 말할 포인트: feature-based expert selection 자체가 이상한 건 아니었다. 문제는 learned router를 완전히 대체하느냐, teacher/보조 의미로 쓰느냐였다.

### Slide 9. 그래서 나온 결론이 `distillation / fusion`

- hard replacement 대신, feature semantics를 learned router에 “가르치거나 더하는” 방향으로 바뀌었다.
- 여기서 파생된 아이디어:
  - `v2 P3RRT`: RR에서 outer router distill
  - `HGRv3`: inner clone distill / 이후 fused bias 예정
  - `flat v3`: `group_only`, `clone_only`, `group_plus_clone`
- 말할 포인트: rule은 최종 구조라기보다 distillation/fusion 아이디어의 근거 실험이었다.

### Slide 10. 이 아이디어를 기존 mainline에 붙여 본 결과

- RR `P1RFI -> P1RGI2 -> P2RGI -> P3RRT`
- 최고는 `0.2644`
- 해석: factorized/distill은 설명력은 늘렸지만, mainline RR best `0.2720`을 넘지는 못했다.
- 말할 포인트: “좋은 아이디어”와 “최종 채택”은 다르다.

### Slide 11. 그래서 다시 old flat router를 분리해서 보는 게 `flat v3`

- `v3`는 `pre-fmoe-v2-router-overhaul-20260309` snapshot rollback이다.
- 이미 완료된 baseline:
  - `V3A_L7_LEGACY = 0.0969`
- 1차 비교 결과:
  - `P2ROUTER`: `flat_legacy 0.0953 > flat_hidden_only 0.0926`
  - `P2DISTILL`: `group_only 0.0956 > plain 0.0953`, but both `< legacy 0.0969`
- 말할 포인트: v2가 강했던 이유를 “새 router semantics 때문”이라고 단정하기 어렵고, 동시에 단순 hidden-only는 여전히 손해가 크며, coarse group teacher는 약간의 이득은 줘도 아직 legacy baseline은 못 넘는다는 점이 확인됐다.

### Slide 12. 지금 결론과 다음 액션

- 현재 결론:
  - 채택: `FMoE_v2`
  - 가장 좋은 분석 브랜치: `HGR`
  - 가장 중요한 확인 실험: `Rule`
  - 현재 진행형: `HGRv3`, `flat v3`
- 결과에 따른 다음 분기:
  - `HGRv3`에서 distill/fused bias가 `off`를 넘기면 inner-teacher 쪽을 확장
  - `flat v3`에서 `clone_only/group_plus_clone`이 plain을 넘기면 teacher supervision을 flat mainline에 다시 이식
  - 둘 다 못 이기면 `FMoE_v2 RR anchor`를 유지하고 supervision만 최소한으로 선택 이식

## 왜 이 실험을 했는가

기본 목적은 순차 추천에서 FMoE 계열 구조가 실제로 어느 조합에서 성능 이득을 내는지 확인하는 것이었다. 운영 방식은 처음부터 일관됐다. 먼저 `movielens1m`에서 구조 anchor를 만들고, 그 anchor가 `retail_rocket`로 얼마나 잘 전이되는지를 본다. 채택 기준은 seed 평균이 아니라 `MRR@20` 단일 best score였다.

이 방식의 장점은 의사결정이 빠르다는 점이다. 반대로 단점은 과적합된 best-run이 섞일 수 있다는 것인데, 이번 트랙은 연구 탐색 단계였기 때문에 재현성보다 구조 가설의 빠른 pruning을 우선했다.

## 모델 구조 변천사

구조를 더 자세히 설명한 토글형 보강 문서: [fmoe_model_architecture_details_20260310.md](fmoe_model_architecture_details_20260310.md)

| 브랜치 | 핵심 구조 | 실험 의도 | 현재 해석 |
|---|---|---|---|
| `FMoE_v2` | layout object 기반 stage 분리, `serial/parallel`, `learned/rule_soft` router 지원 | v1을 건드리지 않고 구조/설정/실험 트랙을 분리한 mainline 구축 | 현재까지 가장 실전적인 기준선이다. |
| `Rule` | `R0` pure rule, `R1` macro learned + mid/micro rule hybrid | rule-soft routing이 learned router를 대체/보완할 수 있는지 확인 | pure rule은 실패, hybrid만 ML1M에서 약한 이득을 보였다. |
| `ProtoX` | prototype-first routing, session prototype mixture가 stage allocation과 routing을 동시에 조건화 | session prototype을 먼저 잡으면 stage/expert routing이 더 안정화될 것이라는 가설 | 아이디어는 분명하지만 optimization 난도가 높고 현재 점수는 낮다. |
| `HiR2` | stage allocator + stage 내부 router의 2단 게이팅 | stage-level allocation을 먼저 결정하면 routing 해석력이 올라갈지 확인 | 구조는 깔끔하지만 현재 성능은 mainline과 격차가 크다. |
| `HGR` | group router, `stage_merge_mode`, `group_router_mode`, distill/spec 축 | HiR류 scaffold를 유지하면서 routing-first로 구조를 재설계 | 내부 phase progression은 좋았지만 mainline을 넘지 못했다. |
| `HGRv3` | outer hidden-only router + inner clone routing teacher | HGR의 outer distill을 버리고 inner clone semantics를 분리해 보려는 보완안 | 현재까지는 `off` baseline이 `distill`보다 약간 더 낫다. |
| `HiR` | serial/parallel x schedule on/off 비교 중심의 기존 hierarchical routing 트랙 | FMoE 대비 계층형 routing 효과를 격리해서 보기 위함 | 현재 artifacts 기준 정량 결과가 비어 있다. |
| `v3` | `pre-fmoe-v2-router-overhaul-20260309` 시점의 flat-router rollback baseline | router overhaul 이전 설계를 복원해 비교 baseline으로 활용 | legacy baseline `0.0969`, 1차 router/distill 비교는 완료됐고 다음은 `clone_only/group_plus_clone`이다. |

## 코드 / 라우터 변화 메모

| 시기 | 실제로 바뀐 것 | 발표에서 어떻게 말하면 좋은가 |
|---|---|---|
| 2026-03-05 ~ 2026-03-07 | `v2`는 `layout + serial/parallel` 엔진 위에 flat learned router 중심으로 anchor를 만든 단계 | “처음엔 잘 되는 anchor를 만드는 게 목적이었고, 구조적으로는 아직 flat 쪽에 가까웠다” |
| 2026-03-06 ~ 2026-03-07 | `rule_soft`를 stage별로 섞는 hybrid가 들어오고, ML1 final에서도 일부 흡수됨 | “rule은 새 모델이 아니라 v2가 feature semantics를 흡수해 가는 과정이었다” |
| 2026-03-08 ~ 2026-03-09 | RR 전이 과정에서 `group_factorized_interaction`, feature-spec, router distill 같은 router-overhaul 아이디어가 v2 안으로 들어옴 | “v2는 고정 모델이 아니라, 좋은 아이디어를 계속 먹은 evolving mainline이었다” |
| 2026-03-10 | 그래서 `v3`를 따로 열어 old flat router를 rollback baseline으로 다시 확인 중 | “지금 보는 v3는 새 주력이 아니라, v2가 너무 많이 바뀌어서 만든 control” |

## 실험 진행 흐름

| 날짜 | 트랙 / phase | 가설 또는 변경점 | best `MRR@20` | 해석 |
|---|---|---|---:|---|
| 2026-03-05 | `fmoe_v2` `P0 -> P1` | 로깅, 환경, wandb, report 체계를 먼저 안정화 | 0.0151 -> 0.0243 | 의미 있는 성능 단계가 아니라 실험 기반을 정리한 단계였다. |
| 2026-03-05 | `fmoe_v2` `P1S` | ML1M에서 layout/execution을 넓고 얕게 스크린 | 0.0975 | `serial + L7`이 빠르게 anchor로 떠올랐다. |
| 2026-03-05~06 | `fmoe_v2` `P2DB` | `serial`과 `parallel`에서 dim/router/batch coupling 확인 | 0.0982 vs 0.0969 | `serial`이 더 높고 OOM도 적어 mainline 방향이 고정됐다. |
| 2026-03-06 | `fmoe_rule` `R0/R1` | pure rule과 hybrid rule을 같은 ML1M anchor 위에서 비교 | 0.1000 vs 0.0752 | `R0`는 크게 무너졌고 `R1`만 의미 있는 개선을 보였다. |
| 2026-03-07 | `fmoe_v2` `FINALV2` | ML1M mainline 최종 squeeze | 0.0986 | mainline도 후반부에는 `mid/micro rule-soft`를 일부 흡수해 rule best와 사실상 수렴했다. |
| 2026-03-07 | `fmoe_hir2`, `fmoe_protox` | stage-first / prototype-first 대안 구조 탐색 | 0.0751 / 0.0838 | 구조 실험으로는 의미 있지만 mainline 후보는 아니었다. |
| 2026-03-08 | `fmoe_v2` `P1S -> P2RRF` on RR | RR 전이용 layout와 capacity를 다시 anchor 근처에서 재탐색 | 0.2699 -> 0.2720 | `L16 + serial + 128/24/160/64`가 명확한 RR best가 됐다. |
| 2026-03-09 | `fmoe_v2` `P1RFI/P1RGI2/P2RGI/P3RRT` | factorized router, feature-spec, distill, top-k를 RR에서 세밀하게 비교 | 0.2617~0.2644 | 해석용으론 유익했지만 mainline RR best를 넘지 못했다. |
| 2026-03-09 | `fmoe_hgr` `P1 -> P1.5 -> P2 -> P3` | group-router 계열을 단계적으로 고도화 | 0.0946 -> 0.0937 -> 0.0956 -> 0.0958 | phase progression은 건강했지만 absolute best는 아직 낮다. |
| 2026-03-10 | `fmoe_rule` `RRRULE` | ML1M rule hybrid 상위권이 RR에서도 먹히는지 quick probe | 0.2625 | RR에서는 learned mainline의 flexibility를 대체하지 못했다. |
| 2026-03-10 | `fmoe_hgr_v3` `R0HGRv3` | inner-teacher 구조의 pure architecture probe: `off` vs weak `distill` | 0.0933 / 0.0922 | 현재까지는 `off`가 `distill`보다 우세하다. |
| 2026-03-10 | `fmoe_v3` `V3A_L7_LEGACY` | router overhaul 이전 flat router를 rollback baseline으로 재현 | 0.0969 | old flat baseline도 여전히 강해서, router 변화 효과를 다시 따져볼 필요가 생겼다. |
| 2026-03-10 | `fmoe_v3` `P2ROUTER` | `flat_legacy` vs `flat_hidden_only` 입력 단순화 비교 | 0.0953 / 0.0926 | hidden-only는 일부 회복했지만 여전히 legacy보다 낮아, old flat interaction 구조의 가치가 남아 있다. |
| 2026-03-10 | `fmoe_v3` `P2DISTILL` | no-distill vs `group_only` teacher 비교 | 0.0953 / 0.0956 | group-only teacher는 plain보다 아주 소폭 높았지만 legacy baseline은 넘지 못했다. |
| 2026-03-10 | `fmoe_hgr_v3` `R1L/R1D` | layout/dim/off 재확인 + weak inner distill 재탐색 | 0.0931 / 0.0919 | 후속 R1에서도 `off`가 우세해, weak distill의 즉시 이득은 확인되지 않았다. |

## 발표용 비교 포인트

발표에서는 전체 best만 보여 주기보다, 아래처럼 “같은 실험 안에서 무엇을 바꿨더니 몇 점 차이가 났는가”를 짚어 주면 흐름이 훨씬 잘 보인다.

### 1. `FMoE_v2` ML1 anchor를 어떻게 잡았나

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-05 | `P1S` wide-shallow | `serial L7` vs `serial L18` vs `serial L16` | `0.0975` vs `0.0973` vs `0.0972` | `L7`가 초반 anchor로 보여서 P2DB를 `serial/L7` 중심으로 좁혔다. |
| 2026-03-05~06 | `P2DB` ML1 | `serial/L7` vs `parallel/L13` | `0.0982` vs `0.0969` | `serial`이 더 높고 OOM도 적어서 mainline 방향을 `serial`로 고정했다. |
| 2026-03-07 | `FINALV2` ML1 | `P2DB best` vs `FINALV2 hybrid control` | `0.0982` -> `0.0986` | rule hybrid 힌트를 mainline에 일부 흡수한 최종형으로 마무리했다. |

### 2. `Rule`: pure replacement는 실패, hybrid만 먹혔다

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-06 | `RULE` ML1 | `R0 pure rule` vs `R1 hybrid rule` | `0.0752` vs raw log `0.1000` | “feature로 expert를 고르는 것” 자체는 틀리지 않다고 보고, pure replacement 대신 hybrid/teacher 방향으로 갔다. |
| 2026-03-10 | `RRRULE` RR | `L16F24` vs `L15MED` vs `L16BIG` | `0.2625` vs `0.2610` vs `0.2599` | RR에서는 rule hybrid가 mainline `0.2720`을 못 넘어서, rule은 주력보다 설명용/teacher용으로 남겼다. |

### 3. `FMoE_v2` RR 전이: mainline vs factorized/distill

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-08 | `P1S` RR | `L16` vs `L7` vs `L5` | `0.2699` vs `0.2695` vs `0.2690` | RR anchor를 `serial/L16`로 잡고 narrow P2로 넘어갔다. |
| 2026-03-08 | `P2RRF` RR | `L16 main serial` best | `0.2720` | RR control anchor를 확정했다. |
| 2026-03-09 | `P1RFI -> P1RGI2 -> P2RGI -> P3RRT` | factorized/dim/distill 누적 | `0.2617 -> 0.2618 -> 0.2621 -> 0.2644` | distill/spec는 도움은 됐지만 mainline anchor `0.2720`보다 낮아서 채택은 보류했다. |

### 4. `HGR`: 계층형을 가장 그럴듯하게 만든 트랙

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-09 새벽 | `P1 joint fast32` | 초기 router redesign quick screen | `0.0920` | 구조가 아주 망하진 않는다고 보고 더 넓은 P1로 넘어갔다. |
| 2026-03-09 오전 | `P1 wide-shallow` | `serial/per_group` 중심 구조 screen | `0.0946` | anchor를 넓힌 뒤 layout ranking만 따로 정리하는 `P1.5`로 갔다. |
| 2026-03-09 오전 | `P15 layout focus` | `serial/hybrid` vs `serial/per_group` at L15 | `0.0937` vs `0.0933` | `hybrid + L15`가 조금 더 좋아 P2 dim focus 기준점이 됐다. |
| 2026-03-09 오후 | `P2 -> P3` | no-distill/spec vs distill/spec | `0.0956` -> `0.0958` | distill/spec가 아주 작지만 이득이 있어, HGRv3에서도 teacher 위치를 다시 보게 됐다. |

### 5. `HGRv3`: inner teacher로 옮겼더니 구조는 괜찮지만 weak distill은 아직 약하다

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-10 | `R0HGRv3` A1 | `off k1` vs `weak distill k2` | `0.0933` vs `0.0922` | 구조 probe 단계에서는 `off`가 더 좋아서 distill 세팅을 바로 채택하지 않았다. |
| 2026-03-10 | `R0HGRv3` A0 | `off k1` vs `weak distill k2` | `0.0930` vs `0.0919` | anchor가 바뀌어도 같은 패턴이라 off 우세가 더 분명해졌다. |
| 2026-03-10 | `R1LHGRv3/R1DHGRv3` | `off` vs `weak distill` follow-up | `0.0931` vs `0.0919` | weak distill 대신 `fused_bias`나 stronger distill을 봐야겠다는 근거가 생겼다. |

### 6. `flat v3`: old flat router를 다시 보니 baseline이 여전히 셌다

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-10 | `V3A_L7_LEGACY` | rollback baseline | `0.0969` | old flat router 자체가 약한 건 아니라는 control을 확보했다. |
| 2026-03-10 | `P2ROUTER` | `legacy` vs `hidden_only` | `0.0953` vs `0.0926` | hidden-only로 단순화하면 여전히 손해가 나서, interaction 구조는 유지하는 쪽으로 갔다. |
| 2026-03-10 | `P2DISTILL` | `plain` vs `group_only` | `0.0953` vs `0.0956` | coarse group teacher도 완전히 틀리진 않다고 보고, 다음 후보를 `clone_only/group_plus_clone`로 잡게 됐다. |

### 7. `ProtoX`와 `HiR2`: 아이디어는 있었지만 주력 후보까지는 못 갔다

| 날짜 | 실험 단위 | 비교 | 결과 | 그래서 다음에 한 것 |
|---|---|---|---:|---|
| 2026-03-07~08 | `ProtoX` | seed baseline -> `P1_fast_wide` -> `P2_ml1_focus` | `0.0756 -> 0.0743 -> 0.0838` | tuning으로는 올랐지만 여전히 mainline과 격차가 커서 주력 대신 구조 아이디어로 남겼다. |
| 2026-03-07 | `HiR2` | best config vs next configs | `0.0751` vs `0.0724/0.0706/0.0690` | stage allocator만 따로 둬도 성능 이득으로 안 이어져, 후속은 HGR/HGRv3 쪽으로 넘어갔다. |

## 트랙별 결과와 해석

### 1. `FMoE_v2` 메인라인

- 실험 의도: `v2` 구조를 mainline으로 세우고, ML1M에서 layout anchor를 만든 뒤 RR로 전이 가능한지 확인
- 비교한 주요 하이퍼파라미터/구조 축: `serial/parallel`, `layout_id`, `embedding/d_feat/d_expert/d_router`, batch, LR/WD, RR에서의 `feature-spec`, `distill`, `expert_top_k`
- 최고 성능:
  - `movielens1m`: `0.0986` at `FINALV2_ML1_G2_C1_R1_CTRL_A128_B8192_R2`
  - `retail_rocket`: `0.2720` at `P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096`
- 해석:
  - ML1M에서는 `serial`이 `parallel`보다 안정적이었고, `P2DB` 기준 OOM도 더 적었다. `L7` 계열이 가장 강한 anchor였다.
  - ML1M final best는 완전 learned가 아니라 `mid/micro rule-soft`를 유지한 hybrid였다. 즉, rule ablation에서 얻은 힌트가 후반 mainline에도 일부 흡수됐다.
  - RR로 넘어가면 best layout이 `L16`으로 이동한다. 즉, ML1M anchor와 RR transfer anchor는 완전히 같지 않았고, RR에서 다시 anchor-near tuning이 필요했다.
  - `factorized router / distill / feature-spec` 계열은 상위권을 만들었지만 최고점은 `0.2644`로 mainline RR best보다 낮았다. 해석 가치는 컸지만 채택 사유는 만들지 못했다.
- 다음 액션: RR 기준선은 계속 `L16 + serial + E128/F24/H160/R64`를 control로 유지하는 것이 맞다.

### 2. `fmoe_rule`

- 실험 의도: learned router를 pure rule 또는 mixed rule로 바꿨을 때 성능/전이성이 좋아지는지 확인
- 비교한 주요 하이퍼파라미터/구조 축: `R0` vs `R1`, `router_impl_by_stage`, `n_bins`, `feature_per_expert`, merge aux, parallel gate, LR/WD
- 최고 성능:
  - `movielens1m`: raw log 기준 `0.1000` at `RULE_R1_P2DB_G7_C1_movielens1m_E128_R64_B6144`
  - `retail_rocket`: `0.2625` at `RRRULE_R1_G5_C01_L16BASE`
- 해석:
  - ML1M에서는 `R1` hybrid가 raw log 기준 `0.1000`으로 가장 높았다. 다만 이 값은 early run log에 남은 best이고, 자동 overview는 JSON/집계 불일치 때문에 `0.0988`까지만 잡고 있다.
  - 다만 late-stage `fmoe_v2` best 자체도 `mid/micro rule-soft`를 포함하므로, ML1M에서는 결국 mainline과 rule branch의 경계가 일부 합쳐졌다.
  - `R0` pure rule은 `0.0752`로 크게 밀렸다. macro까지 rule로 고정하는 것은 현재 구조에서 지나치게 경직적이다.
  - RR quick probe에서는 `R1`이 `0.2604~0.2625` 범위에 머물렀고, mainline anchor `0.2720`을 넘지 못했다. ML1M에서 보인 작은 이득이 RR 전이로 이어지지 않았다.
- 다음 액션: rule은 현재 기준으로 mainline 대체가 아니라, `왜 성능이 변하는가`를 설명하는 ablation/control로 유지하는 것이 맞다.

### 3. `fmoe_hgr`

- 실험 의도: HiR 계열 scaffold를 유지하되 `group_router_mode`, `group_top_k`, `stage_merge_mode`, distill/spec를 중심으로 routing-first 재설계
- 비교한 주요 하이퍼파라미터/구조 축: `serial/per_group`, `serial/hybrid`, layout depth, dim bucket, `group_top_k`, `router_distill_*`, `group_feature_spec_aux_*`
- 최고 성능:
  - `movielens1m`: `0.0958` at `P3HGR_router16_C14_A1_M6_serial_hybrid`
  - phase별 최고: `P1=0.0946`, `P1.5=0.0937`, `P2=0.0956`, `P3=0.0958`
- 해석:
  - 이 트랙의 가장 좋은 점은 progression이 명확하다는 것이다. `P1`에서 빠르게 상위 구조를 고른 뒤 `P1.5`에서 layout rank를 정제하고, `P2`에서 dim을 고정하고, `P3`에서 distill/spec를 붙이는 흐름이 논리적으로 맞았다.
  - `P3`에서 distill/spec가 소폭 이득을 줬고, `serial + hybrid + L15`가 일관된 main anchor였다.
  - 그래도 absolute best는 `fmoe_v2`나 `rule R1`보다 낮다. 즉, 구조 실험으로는 건강하지만 주력 채택용 수치는 아니다.
- 다음 액션: HGR는 바로 채택하기보다, 여기서 먹힌 `distill/spec` 힌트를 mainline 또는 HGRv3로 이식하는 편이 낫다.

### 4. `fmoe_protox`

- 실험 의도: session prototype mixture를 먼저 추정한 뒤 stage allocation과 expert routing을 동시에 조건화
- 비교한 주요 하이퍼파라미터/구조 축: capacity profile `C0~C3`, routing/stability profile `R0~R2`, `proto_usage_lambda`, `proto_entropy_lambda`, LR/WD/dropout/balance
- 최고 성능:
  - `movielens1m`: `0.0838` at `P2_ml1_focus_ML1_R1_G7_C04_S04`
- 해석:
  - prototype-first라는 아이디어는 설명력은 좋지만 현재 optimization이 어렵다. best run에서도 dropout과 balance를 강하게 써야 했고, 학습 안정성도 mainline보다 떨어진다.
  - 현재 수치는 HGR보다도 낮다. 따라서 “새 가설의 가능성”은 남지만, mainline 후보로 올리기엔 이르다.
- 다음 액션: 계속 본다면 architecture search보다 optimization simplification이 먼저다.

### 5. `fmoe_hir2`

- 실험 의도: stage allocator와 stage 내부 router를 분리한 2단 게이팅이 실제로 도움이 되는지 확인
- 비교한 주요 하이퍼파라미터/구조 축: `serial_weighted/parallel_weighted`, allocator `top_k`, layer profile `L0~L5`, arch profile `A0~A3`, stage control `S0~S2`
- 최고 성능:
  - `movielens1m`: `0.0751` at `HIR2_ML1_G5_C1_SER_TK0_L0_A2_S0_R1`
- 해석:
  - stage-first allocation 자체는 해석하기 좋은 구조지만, 현재 설정에서는 추천 정확도를 끌어올리는 데 실패했다.
  - allocator를 따로 둔 복잡도가 이득으로 바뀌지 않았고, best score가 mainline과 꽤 멀다.
- 다음 액션: 독립 mainline보다는 구조 아이디어 아카이브로 남기는 편이 맞다.

### 6. `fmoe_hgr_v3`

- 실험 의도: rule teacher를 outer group routing이 아니라 inner clone routing에만 적용하고, outer router는 hidden-only로 단순화
- 비교한 주요 하이퍼파라미터/구조 축: `off` vs weak `distill`, anchor `A0/A1`, `layout 15`, `expert_top_k`, LR/WD
- 최고 성능:
  - `movielens1m`: `0.0933` at `R0HGRv3_router8_C04_A1_off_k1`
  - 현재 follow-up: `R1L off=0.0931`, `R1D weak distill=0.0919`
- 해석:
  - 초기 quick probe 기준으로는 `off`가 `distill`보다 높다. 후속 `R1`에서도 `off 0.0931 > weak distill 0.0919`라서, weak distill이 즉시 이득을 준다고 보기 어렵다.
  - 그래도 `ProtoX`, `HiR2`보다 출발점이 낫고, HGR의 구조적 약점을 정확히 겨냥한 보완안이라는 점에서 의미가 있다.
  - 현재 숫자는 “HGRv3가 맞다”가 아니라 “HGRv3 구조는 볼 만하지만 distill/fusion은 더 세밀하게 튜닝해야 한다”에 가깝다.
- 다음 액션: `off > distill` 관계가 유지되는지 먼저 확인하고, 그 뒤에만 `fused_bias`나 stronger distill로 넘어가는 것이 맞다.

### 7. `fmoe_v3`

- 실험 의도: router overhaul 이전 flat-router snapshot을 rollback control로 복원하고, old flat baseline과 새 distill 아이디어를 분리해서 보기
- 비교한 주요 하이퍼파라미터/구조 축: `flat_legacy`, `flat_hidden_only`, `flat_hidden_group_clone12`, no-distill vs `group_only`, 이후 `clone_only/group_plus_clone`
- 최고 성능:
  - `movielens1m`: `0.0969` at `V3A_L7_LEGACY`
  - 1차 비교: `P2ROUTER legacy=0.0953`, `P2ROUTER hidden_only=0.0926`, `P2DISTILL plain=0.0953`, `P2DISTILL group_only=0.0956`
- 해석:
  - rollback baseline이 `0.0969`라는 건 old flat router가 지금 봐도 충분히 강한 control이라는 뜻이다.
  - 즉, v2가 좋아진 이유를 “router가 더 똑똑해졌기 때문”이라고 단정할 수 없고, 기존 flat anchor도 상당히 강했다는 사실을 같이 봐야 한다.
  - 방금 끝난 `P2ROUTER/P2DISTILL`도 이 해석을 강화한다. hidden-only는 회복하더라도 legacy보다 낮고, `group_only` teacher는 plain보다 아주 소폭 높지만 여전히 legacy를 넘지는 못했다. 즉, semantics를 넣더라도 base flat logits를 깨지 않는 방식이어야 한다.
- 다음 액션: 다음 후보는 `clone_only`나 `group_plus_clone`처럼 base logits를 유지한 채 보조 의미만 더하는 방향이다.

### 8. 결과 미정리 또는 미실행 트랙

- `fmoe_hir`
  - README와 phase 설계는 존재하지만, 현재 `artifacts/results/fmoe_hir`에는 scored result JSON이 없고 overview도 비어 있다.
  - 발표에서는 “초기 비교 트랙으로 설계됐지만 현재 정량 근거는 비어 있음” 정도로만 언급하는 것이 적절하다.

## 실험 중 / 예정

### `HGRv3`

- 지금 보고 있는 축:
  - outer hidden-only router
  - inner clone routing teacher `off` vs `distill`
  - anchor `A0(128/16/160/64)` vs `A1(160/16/256/112)`
  - `expert_top_k=1/2`
  - 이후 후보: `fused_bias`, `distill_and_fused_bias`
- 현재 상태:
  - `R0` 정리 기준 best는 `off 0.0933`, weak `distill 0.0922`
  - `R1L/R1D` 1차 완료 결과는 각각 `0.0931`, `0.0919`
- 결과에 따른 다음 분기:
  - `distill` 또는 이후 `fused_bias`가 `off`를 넘기면 inner-teacher 방향을 확장
  - 거기까지 못 올라오면 HGR의 distill/spec 아이디어만 mainline에 이식하고 브랜치는 축소

### `flat v3`

- 지금 보고 있는 축:
  - `flat_legacy` rollback control
  - `flat_hidden_only`
  - `flat_hidden_group_clone12`
  - distillation mode `none -> group_only -> clone_only/group_plus_clone`
- 현재 상태:
  - 완료된 baseline은 `V3A_L7_LEGACY = 0.0969`
  - `P2ROUTER`: `legacy 0.0953`, `hidden_only 0.0926`
  - `P2DISTILL`: `plain 0.0953`, `group_only 0.0956`
- 결과에 따른 다음 분기:
  - hidden-only가 무너진 만큼, 다음 flat 실험은 interaction 구조를 유지하는 전제에서만 보는 게 맞다
  - `group_only`가 plain보다 아주 소폭 높았지만 legacy를 못 넘었으므로, 다음 후보는 `clone_only`나 `group_plus_clone`처럼 더 국소적인 teacher다
  - 이것마저 약하면 rollback은 control로만 두고 `HGRv3` 또는 mainline distill 쪽에 자원을 집중

## 최종 비교 및 발표 메시지

메인라인 승자는 현재까지 분명히 `FMoE_v2`다. RR 전이 기준으로는 다른 브랜치가 아직 이 기준선을 건드리지 못했다.  
ML1M에서는 `rule R1`이 raw log 기준 `0.1000`으로 가장 높았고, `flat v3 legacy`도 `0.0969`로 강했다. 즉, feature semantics나 old flat router 자체가 틀렸던 건 아니다. 다만 그 이득이 RR 전이와 최종 채택으로 이어지느냐는 별개 문제였다.  
`HGR`는 구조적 progression이 가장 깔끔했고, `distill/spec`가 완전히 무의미하지 않다는 점을 보여 줬다. 반면 `ProtoX`, `HiR2`는 현재 상태로는 exploratory branch 이상이 아니다.  
`HGRv3`와 `flat v3`는 둘 다 중요한 control이다. 특히 오늘 나온 `flat v3` 1차 결과는 hidden-only가 legacy보다 낮고, coarse `group_only` teacher는 plain보다 아주 소폭 높지만 legacy는 못 넘는다는 점을 보여 줬다. 지금 단계에서는 “무엇이 더 좋다”보다 “rule semantics를 hard replace가 아니라, base logits를 유지한 distill/fusion으로 넣는 게 맞는지”를 검증하는 과정이라고 보는 편이 정확하다.

발표에서 가져갈 메시지는 세 가지다.

1. 주력 채택안은 현재도 `FMoE_v2 mainline`, 특히 RR 기준 `L16 + serial` anchor다.
2. `Rule`은 “feature로 expert를 고르는 MoE가 이상한가?”라는 질문에 대한 확인 실험이었고, 답은 “pure replacement는 아니지만 hybrid/teacher로는 유효하다”였다.
3. hierarchy와 distillation/fusion은 아직 탐색 중이지만, 지금까지 데이터만 보면 mainline을 갈아탈 수준의 정량 근거는 없다.

다음 실험 방향은 `FMoE_v2 RR anchor`를 control로 고정한 상태에서, `HGR/HGRv3`에서 나온 router supervision 아이디어만 선별적으로 다시 mainline에 이식하는 것이다. 새 브랜치를 또 여는 것보다, 이미 효과가 보인 작은 축을 mainline control 위에 재삽입하는 쪽이 더 빠르다.

## 부록

### A. 주요 best run

| track | dataset | best `MRR@20` | best phase | 핵심 설정 요약 | 소스 |
|---|---|---:|---|---|---|
| `fmoe_v2` | `movielens1m` | 0.0986 | `FINALV2_ML1_G2_C1_R1_CTRL_A128_B8192_R2` | `L7`, `serial`, mid/micro rule-soft hybrid 유지 | `artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_finalv2_ml1_g2_c1_r1_ctrl_a128_b8192_r2_20260307_104708_748816_pid194632.json` |
| `fmoe_rule` | `movielens1m` | 0.1000 | `RULE_R1_P2DB_G7_C1_movielens1m_E128_R64_B6144` | `R1`, `L7`, `serial`, macro learned + mid/micro rule-soft, log-verified best | `artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_084521_974_hparam_RULE_R1_P2DB_G7_C1_movielens1m_E128_R64_B6144.log` |
| `fmoe_v3` | `movielens1m` | 0.0969 | `V3A_L7_LEGACY` | rollback `flat_legacy`, `L7`, `serial`, distill off baseline | `artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_v3a_l7_legacy_20260310_024154_205487_pid559858.json` |
| `fmoe_v2` | `retail_rocket` | 0.2720 | `P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096` | `L16`, `serial`, `128/24/160/64`, `expert_scale=3` | `artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rrf_l16_g4_c04_e128_f24_h160_r64_b4096_20260308_191445_359236_pid382700.json` |
| `fmoe_rule` | `retail_rocket` | 0.2625 | `RRRULE_R1_G5_C01_L16BASE` | `R1`, `L16`, `serial`, RR quick probe base profile | `artifacts/results/fmoe_rule/retail_rocket_FeaturedMoE_v2_rrrule_r1_g5_c01_l16base_20260310_011928_658397_pid542173.json` |
| `fmoe_hgr` | `movielens1m` | 0.0958 | `P3HGR_router16_C14_A1_M6_serial_hybrid` | `L15`, `serial`, `hybrid`, distill/spec 포함 | `artifacts/results/fmoe_hgr/movielens1m_FeaturedMoE_HGR_p3hgr_router16_c14_a1_m6_serial_hybrid_20260309_213418_478049_pid529617.json` |
| `fmoe_hgr_v3` | `movielens1m` | 0.0933 | `R0HGRv3_router8_C04_A1_off_k1` | `L15`, `serial`, inner rule `off`, hidden-only outer, current best | `artifacts/results/fmoe_hgr_v3/movielens1m_FeaturedMoE_HGRv3_r0hgrv3_router8_c04_a1_off_k1_20260310_023631_122810_pid555970.json` |
| `fmoe_protox` | `movielens1m` | 0.0838 | `P2_ml1_focus_ML1_R1_G7_C04_S04` | prototype-first routing, `proto_usage=0.001`, `proto_entropy=0.001` | `artifacts/results/fmoe_protox/movielens1m_FeaturedMoE_ProtoX_p2_ml1_focus_ml1_r1_g7_c04_s04_20260308_101121_190745_pid301598.json` |
| `fmoe_hir2` | `movielens1m` | 0.0751 | `HIR2_ML1_G5_C1_SER_TK0_L0_A2_S0_R1` | `serial_weighted`, `top_k=0`, `L0`, `A2`, `S0` | `artifacts/results/fmoe_hir2/movielens1m_FeaturedMoE_HiR2_hir2_ml1_g5_c1_ser_tk0_l0_a2_s0_r1_20260307_104708_576725_pid194607.json` |

### B. 주요 log / result 경로

- `fmoe_v2` overview: `artifacts/logs/fmoe_v2/experiment_overview.md`
- `fmoe_rule` overview: `artifacts/logs/fmoe_rule/experiment_overview.md`
- `fmoe_rule` ML1 best raw log:
  - `artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_084521_974_hparam_RULE_R1_P2DB_G7_C1_movielens1m_E128_R64_B6144.log`
- `fmoe_hgr` overview: `artifacts/logs/fmoe_hgr/experiment_overview.md`
- `fmoe_hgr` summary: `experiments/models/FeaturedMoE_HGR/experiment_summary.md`
- `fmoe_rule` RR quick probe log:
  - `artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011925_822_hparam_RRRULE_R1_G5_C01_L16BASE.log`
- `fmoe_hgr_v3` quick probe log:
  - `artifacts/logs/fmoe_hgr_v3/hparam/R0HGRv3/ML1/FMoEHGRv3/router8_C04_A1_off_k1.log`
- `fmoe_hgr_v3` current R1 logs:
  - `artifacts/logs/fmoe_hgr_v3/hparam/R1LHGRv3/ML1/FMoEHGRv3/000_layout8_C00_L15_D0_off_r2.log`
  - `artifacts/logs/fmoe_hgr_v3/hparam/R1DHGRv3/ML1/FMoEHGRv3/000_topk8_C00_A0_weak_k1.log`
- `fmoe_v3` legacy baseline result:
  - `artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_v3a_l7_legacy_20260310_024154_205487_pid559858.json`
- `fmoe_v3` current router/distill logs:
  - `artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_052919_286_hparam_P2ROUTER_C00_LEGACY.log`
  - `artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_052919_283_hparam_P2ROUTER_C01_HONLY.log`
  - `artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_052958_730_hparam_P2DISTILL_base_plain_cmp.log`
  - `artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_052958_730_hparam_P2DISTILL_base_group_cmp.log`
- `fmoe_v3` current router/distill results:
  - `artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2router_c00_legacy_20260310_052922_110009_pid584683.json`
  - `artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2router_c01_honly_20260310_052922_169681_pid584682.json`
  - `artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2distill_base_plain_cmp_20260310_053001_695125_pid585030.json`
  - `artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2distill_base_group_cmp_20260310_053001_708249_pid585029.json`
- `ProtoX` 구조 설명: `experiments/models/FeaturedMoE_ProtoX/quick_guide.md`
- `HiR2` 구조 설명: `experiments/models/FeaturedMoE_HiR2/quick_guide.md`
- `HGRv3` 구조 설명: `experiments/run/fmoe_hgr_v3/HGRV3_STRUCTURE.md`
- `flat v3` 계획: `experiments/run/fmoe_v3/ROUTER_PLAN_v3.md`

### C. 실험 수 / 포함 범위 / 비고

| track | 집계 기준 | 수량 | 데이터셋 | 비고 |
|---|---|---:|---|---|
| `fmoe_v2` | `experiment_overview.md` included runs | 198 | `movielens1m`, `retail_rocket` | summarized experiments 20개, RR factorized/distill 포함 |
| `fmoe_rule` | overview + RR quick probe result JSON | ML1 12 + RR 4 | `movielens1m`, `retail_rocket` | overview 본문은 ML1 중심, RR은 2026-03-10 quick probe 보강 |
| `fmoe_hgr` | `experiment_overview.md` included runs | 58 | `movielens1m` | `P1 -> P3` phase 흐름이 가장 명확한 보조 트랙 |
| `fmoe_hgr_v3` | result JSON + raw log | result JSON 14, hparam/log-support 16 | `movielens1m` | partial result 포함, 현재 해석은 `R0/R1` raw status까지 반영 |
| `fmoe_protox` | scored result JSON files | 76 | `movielens1m` | auto overview는 없고 result JSON 중심으로 해석 |
| `fmoe_hir2` | scored result JSON files | 7 | `movielens1m` | first-pass 수준의 제한된 탐색 |
| `fmoe_hir` | result JSON + overview | 0 | - | README만 있고 현재 정량 근거 없음 |
| `fmoe_v3` | result JSON + raw log | result JSON 5, hparam logs 5 | `movielens1m` | `V3A_L7_LEGACY=0.0969`, `P2ROUTER legacy>honly`, `P2DISTILL group_only>=plain` |

### D. 발표용 한 줄 정리

- `채택`: 현재 배포/메인 후보는 `FMoE_v2`
- `보조 가설`: `rule R1`, `HGR`, `HGRv3`, `flat v3`
- `진행 중 검증`: `HGRv3 stronger distill/fused bias`, `flat v3 clone-only/group-plus-clone`
- `아이디어 저장소`: `ProtoX`, `HiR2`, `HiR`

### E. 집계 차이 메모

- `fmoe_rule` ML1 `R1 C1` run은 raw log에서 `0.1000`이 확인된다.
- 하지만 초기 rule run 일부는 결과 JSON 파일명이 generic timestamp 기반이라 서로 덮어쓴 흔적이 있다.
- 그래서 `experiment_overview.md`는 `0.0988`을 top으로 보지만, `best_by_dataset.json`과 raw log는 `0.1000`을 가리킨다.
- 이 보고서는 발표 목적상, 더 보수적인 JSON 값이 아니라 실제 run log로 검증된 `0.1000`을 최종 best로 채택했다.
