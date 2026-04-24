# 데이터셋 Appendix 분석

## 측정 항목

- 기본 구조: interaction 수, session 수, user 수, item 수, user당 평균 session 수, session당 평균 interaction 수.
- feature를 쓰지 않는 비교용 지표만 사용: session volatility, transition branching, context availability.
- 간단한 요약 점수: Simple Routing Score = mean(Session Volatility, Transition Branching, Context Availability).
- 보조로 Raw Behavioral Heterogeneity와 Raw Routing Opportunity도 남기지만, 해석의 중심은 세 개의 직관적인 성분입니다.

## 요약 표

| Dataset | Interactions | Sessions | Users | Avg sess/user | Avg inter/session | Users>=2 sessions | Users>=5 sessions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRec | 287,411 | 24,458 | 1,122 | 21.80 | 11.75 | 0.938 | 0.862 |
| Foursquare | 145,238 | 25,369 | 1,083 | 23.42 | 5.73 | 0.997 | 0.967 |
| LastFM | 470,408 | 25,089 | 130 | 192.99 | 18.75 | 0.992 | 0.977 |
| ML-1M | 575,281 | 14,539 | 6,038 | 2.41 | 39.57 | 0.551 | 0.121 |
| Retail Rocket | 821,243 | 153,092 | 124,922 | 1.23 | 5.36 | 0.099 | 0.009 |
| Beauty | 33,488 | 4,243 | 3,603 | 1.18 | 7.89 | 0.109 | 0.008 |

## Feature-Free Heterogeneity 요약

| Dataset | SessionVol | Branching | ContextAvail | SimpleScore | RawHet | RoutingOpp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRec | 0.443 | 0.903 | 0.933 | 0.760 | 0.869 | 0.811 |
| Foursquare | 0.472 | 0.688 | 0.988 | 0.716 | 0.734 | 0.725 |
| LastFM | 0.370 | 0.630 | 0.990 | 0.663 | 0.773 | 0.765 |
| ML-1M | 0.191 | 0.839 | 0.264 | 0.431 | 0.501 | 0.133 |
| Retail Rocket | 0.472 | 0.744 | 0.056 | 0.424 | 0.578 | 0.033 |
| Beauty | 0.347 | 0.843 | 0.059 | 0.416 | 0.497 | 0.029 |

## RouteRec 성능 요약

- Full table 기준 RouteRec은 54개 row 중 34개에서 최고 성능을 기록했습니다.
- 데이터셋별 승수는 KuaiRec 9/9, LastFM 9/9, Retail Rocket 6/9, Beauty 5/9, Foursquare 5/9, ML-1M 0/9 입니다.
- best baseline 대비 평균 격차는 KuaiRec 0.0182, Retail Rocket 0.0055, LastFM 0.0049, Beauty 0.0020, Foursquare 0.0009, ML-1M -0.0031 입니다.
- RouteRec의 가장 잦은 경쟁 baseline은 FDSA(27회)와 FEARec(18회)였습니다. 즉 strongest competitor는 대체로 feature-aware 계열(FDSA) 또는 강한 sequence encoder 계열(FEARec)입니다.
- KuaiRec과 LastFM에서는 9개 metric 전부에서 1위를 기록해, 단일 cutoff 우연이 아니라 전반적인 ranking quality 개선으로 해석할 수 있습니다.
- Foursquare에서는 HR/NDCG 계열 우위가 분명하지만 MRR 계열은 FDSA와 DIF-SR가 근소하게 강해, 상위 구간 recall 개선은 크고 earliest-hit precision 이득은 더 제한적이라고 볼 수 있습니다.
- Beauty에서는 HR/NDCG에서는 우세하지만 MRR은 FDSA가 앞서므로, RouteRec이 더 넓은 hit coverage에는 기여했지만 최상위 첫 정답 위치까지 일관되게 끌어올린 것은 아니라고 정리하는 편이 정확합니다.
- ML-1M에서는 전 metric에서 FDSA, DuoRec, FAME 같은 강한 shared-path baseline이 비슷하거나 더 높아, 길고 비교적 안정적인 sequence에서는 추가 routing의 여지가 제한적이라고 해석하는 편이 자연스럽습니다.

## 간단한 해석 가이드

- SessionVol은 session 길이가 얼마나 들쭉날쭉한지를 나타냅니다. 짧은 세션과 긴 세션이 섞여 있으면 높아집니다.
- Branching은 같은 item 이후 다음 행동이 얼마나 여러 방향으로 갈라지는지를 나타냅니다. 높을수록 shared-path encoder 하나로 설명하기 어려운 전이 다양성이 큽니다.
- ContextAvail은 RouteRec의 macro routing이 실제로 쓸 수 있는 반복 session 문맥이 충분한지를 나타냅니다.
- SimpleScore는 위 세 요소의 평균으로, 과도한 설계 없이도 'RouteRec이 도움될 가능성'을 비교하는 간단한 dataset-level 지표로 쓸 수 있습니다.
- Beauty와 ML-1M이 같은 low-score 구간이라도 이유는 다릅니다. Beauty는 context 부족이 핵심이고, ML-1M은 context는 일부 있지만 행동 branching보다 긴 안정적 sequence 구조가 더 지배적입니다.

## 논문 반영 제안

- 본문에서는 복잡한 composite 수식을 전면에 두기보다, SessionVol, Branching, ContextAvail 세 축을 먼저 설명하고 SimpleScore는 appendix 표 정렬용 요약치로만 쓰는 편이 자연스럽습니다.
- 가장 좋은 위치는 Appendix의 dataset 분석 파트입니다. 여기서 raw-log 기반 진단 표를 제시하고, 본문 Q1에서는 'RouteRec의 이득은 branching이 크고 multi-session context가 충분한 데이터에서 가장 크다'는 한두 문장만 가져가는 편이 좋습니다.
- 약점처럼 보이지 않게 쓰려면, Beauty와 ML-1M을 '실패 사례'로 규정하지 말고, 각각 context scarcity와 strong shared-path suitability가 더 지배적인 조건이라고 설명하는 편이 안전합니다.
- baseline 해석은 가볍게 유지하는 것이 좋습니다. FDSA는 feature-aware/shared-path strong baseline으로서 가장 자주 RouteRec와 경쟁했고, FEARec은 강한 sequence encoder로서 동적 데이터에서도 꾸준히 근접했습니다.

## 논문용 서술 초안

- We summarize dataset-level routing demand from raw logs using three descriptive factors only: session volatility, transition branching, and multi-session context availability. This keeps the analysis model-agnostic and avoids tying the dataset study to RouteRec's internal hand-crafted features.
- Under this view, KuaiRec and Foursquare provide the clearest combination of dynamic local transitions and sufficient repeated-session context, which is consistent with the stronger and broader gains of RouteRec in those datasets.
- Beauty and ML-1M should be interpreted differently rather than grouped as simple failure cases: Beauty offers little repeated-session context for most users, whereas ML-1M is a longer and more preference-stable setting in which strong shared-path sequential encoders are already highly competitive. In both cases, the smaller RouteRec margin is therefore better read as limited routing headroom than as a categorical weakness of the method.
