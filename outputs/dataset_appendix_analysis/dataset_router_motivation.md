# Router Motivation From Raw Logs

## Motivation

- 핵심 질문은 'heterogeneity가 큰 dataset이 무엇인가'가 아니라, sequential input이 어떤 축에서 반복적으로 heterogeneous해지는가이다.
- 이를 위해 raw interaction log만 사용해 여러 candidate axis를 계산했고, dataset-specific한 특수 현상보다 여러 dataset에서 공통적으로 관찰되고 router 설계로 연결될 수 있는 축을 우선적으로 골랐다.
- 따라서 아래의 4(+1)축은 결과를 가장 잘 맞춘 후행 score가 아니라, raw logs에서 반복적으로 드러나는 heterogeneity pattern을 정리한 motivation-oriented axis set이다.

## Recurring Heterogeneity Axes

- Tempo / volatility: 입력의 pace와 session form이 얼마나 빠르고 불규칙하게 바뀌는가
- Transition ambiguity: 비슷한 local state에서 다음 행동이 얼마나 여러 방향으로 갈라지는가
- Memory regime: 반복 소비와 persistence가 얼마나 강하고 또 얼마나 가변적인가
- Exposure regime: 행동이 head-heavy exposure 쪽에 더 묶이는지, 더 preference-driven한지
- Context availability: 위 heterogeneity가 존재하더라도, router가 활용할 repeated-session context가 충분한가

## Why These Axes

- 첫째, dataset 간 변동폭이 충분히 커야 한다. 그래야 axis가 실제 dataset profile을 나누는 설명 변수로 기능할 수 있다.
- 둘째, raw logs에서 직접 계산 가능해야 한다. 그래야 model-agnostic motivation이 된다.
- 셋째, cue family로 자연스럽게 연결되어야 한다. 즉 predictor feature가 아니라 router control 축으로 해석될 수 있어야 한다.
- 넷째, 서로 완전히 같은 현상을 중복해서 설명하지 않아야 한다. 그래서 short/long share 같은 지표는 tempo의 보조 관측치로 두고, 메인 axis는 더 응축된 형태로 잡았다.

## How Each Candidate Axis Is Computed

- 공통 입력: 모든 axis는 `session_id`, `user_id`, `item_id`, `timestamp` 네 컬럼만 사용한다. 먼저 interaction log를 `(user_id, session_id, timestamp)` 순으로 정렬한 뒤, session 단위와 user-session sequence 단위 집계를 만든다.
- Session summary: 각 session에 대해 `session_len`, `session_start`, `session_end`, `unique_items`, `duration=session_end-session_start`, `repeat_ratio=1-(unique_items/session_len)`를 계산한다.
- User-level session chain: 각 user의 session을 시간순으로 놓고, 인접 session 쌍마다 item-set overlap과 session start 간 gap을 계산한다.
- Transition summary: 각 session 내부에서 `(current_item, next_item)` transition을 만들고, 동일한 `current_item`에서 다음 item 분포가 얼마나 퍼지는지 본다.

### Candidate Axis Definitions

- `avg_sessions_per_user`: user별 session 수를 센 뒤 그 평균을 사용한다.
- `avg_session_len`: session별 interaction 수의 평균이다.
- `session_volatility`: session length의 변동계수로 계산한다. 즉 `std(session_len) / mean(session_len)`이다.
- `short_session_share`: 길이 5 이하 session의 비율이다.
- `long_session_share`: 길이 20 이상 session의 비율이다.
- `repeat_intensity`: session별 `repeat_ratio`의 평균이다. 한 session 안에서 item 재방문이 많을수록 커진다.
- `repeat_variability`: session별 `repeat_ratio`의 변동계수다. 즉 `std(repeat_ratio) / mean(repeat_ratio)`이며, 반복 강도가 session마다 얼마나 들쭉날쭉한지 본다.
- `carryover_strength`: 같은 user의 연속한 두 session에 대해, 두 item set의 Jaccard overlap `|A∩B| / |A∪B|`를 계산하고 그 평균을 취한다.
- `cross_session_drift`: `1 - carryover_strength`로 둔다. 즉 연속 session이 얼마나 다른 item set으로 이동하는지 보는 반대 방향 지표다.
- `timing_irregularity`: user의 연속 session pair에 대해 `next_session_start - previous_session_start` gap을 모은 뒤, 그 변동계수 `std(gap) / mean(gap)`를 사용한다.
- `duration_irregularity`: session duration의 변동계수 `std(duration) / mean(duration)`이다.
- `transition_branching`: 각 source item에서 다음 item 분포의 entropy를 계산하고, 이를 가능한 분기 수에 맞춰 정규화한 뒤 transition 수로 가중 평균한다. 값이 클수록 같은 local state에서 다음 행동이 더 여러 방향으로 갈라진다.
- `popularity_concentration`: 전체 item frequency 분포의 Gini coefficient로 계산한다. 값이 클수록 interaction이 소수 head item에 더 집중된다.
- `head_item_share`: 가장 많이 등장한 단일 item의 interaction 비중이다. 즉 `max_item_frequency / total_interactions`다.
- `context_availability`: `(users>=2 sessions 비율 + users>=5 sessions 비율 + clip(avg_sessions_per_user/20, 0, 1)) / 3`으로 만든 보조 지표다. 반복 session 문맥이 router에 얼마나 제공되는지 보기 위한 것이다.
- `simple_router_demand`: `mean(clip(session_volatility/2, 0, 1), transition_branching, context_availability)`로 만든 단순 종합 지표다. 최종 선택축이라기보다 screening용 reference score에 가깝다.

### Interpretation Notes

- 이 값들은 모두 dataset-level aggregate다. 즉 per-session 또는 per-user raw statistics를 먼저 만든 뒤, dataset마다 하나의 요약값으로 압축한다.
- 대부분의 axis는 절대량보다 scale-free 비교가 중요하다고 보고 평균보다 변동계수(CV)를 사용했다. 그래서 domain별 interaction 규모 차이보다 pattern 차이를 더 보도록 설계했다.
- `transition_branching`과 `popularity_concentration`은 각각 local next-step ambiguity와 global exposure concentration을 잡기 위한 축이라, session length 계열과는 다른 정보를 준다.

## Selected Axes and Raw Indicators

| Axis | Raw-log indicators | Why kept for motivation | Cue-family link |
| --- | --- | --- | --- |
| Tempo / volatility | `session_volatility`, `timing_irregularity`, short/long session share | dataset 간 변동폭이 크고 (`session_volatility` range 0.564, `timing_irregularity` range 5.217), '계산 경로를 빨리 바꿔야 하는 입력'을 가장 직관적으로 설명함 | Tempo |
| Transition ambiguity | `transition_branching` | 다음 행동 분기 구조를 직접 보여주며, hidden-only router가 놓치기 쉬운 local multimodality를 설명함 (`branching` range 0.274) | Focus |
| Memory regime | `repeat_intensity`, `repeat_variability`, supplementary `carryover_strength` | 반복/재등장 패턴은 gain과도 비교적 잘 맞고, dataset 간 차이도 큼 (`repeat_variability` range 2.214) | Memory |
| Exposure regime | `popularity_concentration`, supplementary `head_item_share` | gain과의 직접 정렬은 약하지만, browsing-heavy vs preference-driven regime 차이를 설명하는 독립 축으로 유지할 가치가 있음 (`popularity_concentration` range 0.243) | Exposure |
| Context availability | `users>=2 sessions`, `users>=5 sessions`, `avg_sessions_per_user` | heterogeneity 자체보다 'routing이 실제로 배울 수 있는 문맥'을 설명하는 보조축으로 필요함 (`context_availability` range 0.933) | macro/mid routing support |

## Secondary Candidates That Were Not Promoted

- `duration_irregularity`: 값의 range는 크지만 시간 단위와 sessionization 규칙에 지나치게 민감해, 공통 motivation 축으로 쓰기엔 domain effect가 강하다.
- `short_session_share`, `long_session_share`: tempo/volatility를 보조적으로 보여주는 좋은 지표이지만, 메인 axis라기보다 session volatility를 풀어 설명하는 보조 통계로 두는 편이 깔끔하다.
- `cross_session_drift`: 직관은 좋지만 현재 sessionization에서는 overlap이 거의 0에 수렴하는 dataset가 많아, 메인 motivation 축으로 쓰기엔 너무 거칠다. carryover strength를 보조 evidence로 두는 편이 더 안정적이다.
- `head_item_share`: exposure regime을 설명하는 보조 통계로는 유효하지만, popularity concentration보다 정보량이 적다.

## From Axes To Cue Groups

- Tempo / volatility axis는 세션의 속도와 형태가 얼마나 흔들리는지를 보여주므로 Tempo cue family로 연결된다.
- Transition ambiguity axis는 local next-step branching을 드러내므로, router가 현재 intent concentration vs switching을 보게 하는 Focus family와 연결된다.
- Memory regime axis는 repeat intensity와 repeat variability를 통해 persistence/recurrence 구조를 보여주므로 Memory family로 연결된다.
- Exposure regime axis는 popularity concentration을 통해 head-heavy vs preference-driven browsing 차이를 보여주므로 Exposure family로 연결된다.
- Context availability는 독립 cue family라기보다, 위 cue들이 실제로 macro/mid routing에서 활용될 수 있는 조건을 설명하는 보조축이다.

## Dataset Table

- 아래 첫 표는 본문에서 빠르게 보기 위한 compact table이고, 뒤의 두 표는 이 값들이 무엇의 축약인지 풀어서 보여주는 상세 표다.
- 즉 compact table의 숫자는 임의 축약이 아니라, 뒤에 있는 raw component와 candidate axis 통계를 다시 묶어 놓은 요약값이다.

### Compact Header Glossary

| Compact header | Full metric name | How it is computed | Interpretation |
| --- | --- | --- | --- |
| `SessVol` | `session_volatility` | `std(session_len) / mean(session_len)` | 세션 길이가 dataset 안에서 얼마나 들쭉날쭉한가 |
| `Branch` | `transition_branching` | source item별 next-item entropy를 정규화한 뒤 transition 수로 가중 평균 | 같은 local state에서 다음 행동이 얼마나 여러 갈래로 갈라지는가 |
| `RepeatVar` | `repeat_variability` | `std(repeat_ratio) / mean(repeat_ratio)` | 반복 소비 강도가 session마다 얼마나 불균일한가 |
| `Carryover` | `carryover_strength` | 연속 session item-set의 Jaccard overlap 평균 | 이전 session item set이 다음 session으로 얼마나 이어지는가 |
| `Drift` | `cross_session_drift` | `1 - carryover_strength` | 연속 session이 얼마나 다른 item set으로 이동하는가 |
| `PopConc` | `popularity_concentration` | item frequency 분포의 Gini coefficient | interaction이 소수 인기 item에 얼마나 집중되는가 |
| `CtxAvail` | `context_availability` | `mean(users>=2 ratio, users>=5 ratio, clip(avg_sessions_per_user/20,0,1))` | 반복 session 문맥이 router에 얼마나 제공되는가 |
| `Gain` | `avg_gain_to_best_baseline` | 각 metric row에서 `RouteRec - best baseline`을 구한 뒤 평균 | strongest baseline 대비 평균 우위/열위 |
| `WinRate` | `route_win_rate` | 전체 metric row 중 RouteRec이 1등인 비율 | dataset 내 여러 지표에서 얼마나 자주 최고 성능을 냈는가 |

### Compact Dataset Table

| Dataset | SessVol | Branch | RepeatVar | Carryover | Drift | PopConc | CtxAvail | Gain | WinRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beauty | 0.694 | 0.843 | 0.000 | 0.000 | 1.000 | 0.509 | 0.059 | 0.0020 | 0.556 |
| Foursquare | 0.944 | 0.688 | 1.961 | 0.159 | 0.841 | 0.647 | 0.988 | 0.0009 | 0.556 |
| KuaiRec | 0.887 | 0.903 | 2.214 | 0.003 | 0.997 | 0.671 | 0.933 | 0.0182 | 1.000 |
| LastFM | 0.739 | 0.630 | 2.036 | 0.072 | 0.928 | 0.456 | 0.990 | 0.0049 | 1.000 |
| ML-1M | 0.381 | 0.839 | 0.000 | 0.000 | 1.000 | 0.699 | 0.264 | -0.0031 | 0.000 |
| Retail Rocket | 0.945 | 0.744 | 0.858 | 0.184 | 0.816 | 0.674 | 0.056 | 0.0055 | 0.667 |

### Raw Components Behind The Table

| Dataset | Interactions | Sessions | Users | Items | AvgSess/User | AvgSessLen | Short<=5 | Long>=20 | RepeatInt | TimeIrreg | DurIrreg | HeadItemShare |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beauty | 33,488 | 4,243 | 3,603 | 3,625 | 1.178 | 7.893 | 0.378 | 0.040 | 0.000 | 1.085 | 1.644 | 0.006 |
| Foursquare | 145,238 | 25,369 | 1,083 | 30,588 | 23.425 | 5.725 | 0.713 | 0.029 | 0.064 | 2.214 | 1.059 | 0.005 |
| KuaiRec | 287,411 | 24,458 | 1,122 | 6,477 | 21.799 | 11.751 | 0.204 | 0.137 | 0.086 | 2.571 | 0.871 | 0.007 |
| LastFM | 470,408 | 25,089 | 130 | 52,510 | 192.992 | 18.750 | 0.084 | 0.345 | 0.083 | 6.302 | 0.747 | 0.001 |
| ML-1M | 575,281 | 14,539 | 6,038 | 3,533 | 2.408 | 39.568 | 0.032 | 0.838 | 0.000 | 3.735 | 3.277 | 0.005 |
| Retail Rocket | 821,243 | 153,092 | 124,922 | 90,211 | 1.226 | 5.364 | 0.748 | 0.021 | 0.279 | 2.390 | 1.552 | 0.001 |

### Full Candidate-Axis Value Table

| Dataset | SessVol | Branch | RepeatInt | RepeatVar | Carryover | Drift | TimeIrreg | DurIrreg | PopConc | HeadShare | CtxAvail | RouterDemand | Gain | WinRate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Beauty | 0.694 | 0.843 | 0.000 | 0.000 | 0.000 | 1.000 | 1.085 | 1.644 | 0.509 | 0.006 | 0.059 | 0.416 | 0.0020 | 0.556 |
| Foursquare | 0.944 | 0.688 | 0.064 | 1.961 | 0.159 | 0.841 | 2.214 | 1.059 | 0.647 | 0.005 | 0.988 | 0.716 | 0.0009 | 0.556 |
| KuaiRec | 0.887 | 0.903 | 0.086 | 2.214 | 0.003 | 0.997 | 2.571 | 0.871 | 0.671 | 0.007 | 0.933 | 0.760 | 0.0182 | 1.000 |
| LastFM | 0.739 | 0.630 | 0.083 | 2.036 | 0.072 | 0.928 | 6.302 | 0.747 | 0.456 | 0.001 | 0.990 | 0.663 | 0.0049 | 1.000 |
| ML-1M | 0.381 | 0.839 | 0.000 | 0.000 | 0.000 | 1.000 | 3.735 | 3.277 | 0.699 | 0.005 | 0.264 | 0.431 | -0.0031 | 0.000 |
| Retail Rocket | 0.945 | 0.744 | 0.279 | 0.858 | 0.184 | 0.816 | 2.390 | 1.552 | 0.674 | 0.001 | 0.056 | 0.424 | 0.0055 | 0.667 |

- `Raw Components Behind The Table`는 compact table에 직접 안 들어간 보조 값들까지 포함한다. 예를 들어 `RepeatVar`가 왜 큰지 보려면 `RepeatInt`와 함께 읽는 편이 좋고, `CtxAvail`는 `AvgSess/User` 및 multi-session user 비율과 같이 읽어야 해석이 자연스럽다.
- `Full Candidate-Axis Value Table`는 실제 correlation 계산에 들어간 candidate axis를 거의 전부 모은 표다. 그래서 본문용 compact table, appendix용 full table, 계산 정의 섹션이 서로 연결되도록 구성했다.

## Directional Comparison With Results

- 아래 rank correlation은 dataset 수가 6개뿐이라 강한 통계 검정보다 directional evidence로 읽는 것이 적절하다.
- Motivation의 중심은 correlation 자체가 아니라, 위에서 정의한 axis가 실제 RouteRec gain과도 완전히 어긋나지 않는다는 점을 확인하는 것이다.

| Candidate axis | Rank corr. with gain | Rank corr. with win rate |
| --- | ---: | ---: |
| repeat_intensity | 0.841 | 0.761 |
| repeat_variability | 0.667 | 0.851 |
| session_volatility | 0.543 | 0.441 |
| carryover_strength | 0.377 | 0.403 |
| short_session_share | 0.314 | 0.088 |
| simple_router_demand | 0.257 | 0.441 |
| transition_branching | 0.257 | -0.088 |
| timing_irregularity | 0.086 | 0.353 |
| head_item_share | 0.086 | -0.177 |
| avg_sessions_per_user | 0.029 | 0.441 |
| context_availability | -0.086 | 0.353 |
| popularity_concentration | -0.143 | -0.441 |
| avg_session_len | -0.314 | -0.088 |
| long_session_share | -0.314 | -0.088 |
| cross_session_drift | -0.377 | -0.403 |
| duration_irregularity | -0.600 | -0.883 |

- `repeat_intensity`, `repeat_variability`, `session_volatility`는 gain과 비교적 잘 정렬된다. 즉 memory regime와 tempo volatility는 실제로 routing-relevant한 축으로 볼 근거가 있다.
- `transition_branching`은 correlation 수치만 보면 아주 강하지 않지만, KuaiRec처럼 실제 gain이 가장 큰 데이터에서 매우 높게 나타나고, hidden-only router의 한계를 설명하기에 가장 해석이 좋은 축이므로 motivation에서 유지할 가치가 있다.
- `context_availability`는 단독 상관보다 Beauty/Retail Rocket vs KuaiRec/Foursquare/LastFM의 차이를 설명하는 gating condition으로 더 유용하다.

## Clean Story Flow

- Step 1: raw logs를 보면 sequential input은 여러 dataset에서 반복적으로 tempo, branching, repeat regime, exposure 차이 위에서 heterogeneous해진다.
- Step 2: hidden-only router는 입력이 어느 behavioral axis에서 다른지를 직접 보지 못한다.
- Step 3: 따라서 router input은 richer predictive feature가 아니라, 이 recurring heterogeneity axis를 operationalize한 lightweight cue여야 한다.
- Step 4: RouteRec은 이 축에 맞춰 Tempo, Focus, Memory, Exposure의 네 cue group을 구성하고, context availability는 그 cue가 실제 routing에 활용될 수 있는 조건을 설명하는 보조축으로 둔다.
- Step 5: 이후 결과를 보면, 위 축이 상대적으로 강한 dataset일수록 RouteRec의 이득이 더 크게 나타난다.

## Introduction-Style Paragraph

- Rather than starting from a single scalar notion of heterogeneity, we inspect raw interaction logs and find that sequential inputs differ recurrently along a small set of behavioral axes: temporal volatility, transition ambiguity, repetition and carryover structure, and exposure regime. These axes appear across datasets even though their relative strength varies by domain.
- This suggests that the central MoE question in sequential recommendation is not only whether to route, but along which behavioral axis routing should specialize. Hidden-only routing can respond to representation differences, yet it does not explicitly expose where the heterogeneity comes from. We therefore design router cues as lightweight controls aligned with these recurring axes of heterogeneity rather than as richer predictive side information.
- Under this view, the four cue groups in RouteRec are not arbitrary feature bundles: they are operationalizations of recurring raw-log heterogeneity patterns. The smaller margin on some datasets is then better interpreted as limited routing headroom under scarce repeated-session context or strong shared-path suitability, rather than as evidence against the routing premise itself.

## Paper Use

- Motivation / Introduction에서는 위 Recurring Heterogeneity Axes와 Introduction-Style Paragraph만 가져가고, correlation과 dataset table은 appendix로 내리는 편이 좋다.
- Method에서는 'why these cues'를 설명할 때 Selected Axes and Raw Indicators 표를 축약해서 사용하면 된다.
- Appendix에서는 Dataset Table과 Directional Comparison With Results를 통해, 이 축이 결과와도 대체로 맞물린다는 보조 evidence를 제시하면 충분하다.
