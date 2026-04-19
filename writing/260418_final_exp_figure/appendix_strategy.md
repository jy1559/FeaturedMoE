# RouteRec Appendix Strategy v2

## 핵심 판단

지금 appendix의 문제는 `A06~A09에 뭘 더 넣을까`가 아니다.

더 큰 문제는, 현재 묶음이 `실험 슬롯` 중심이라서 독자가 appendix를 읽을 때

- 어떤 그림은 구조 정당화,
- 어떤 그림은 경계 조건,
- 어떤 그림은 diagnostics,
- 어떤 그림은 behavior slice,

를 한 흐름으로 이해하지 못한다는 점이다.

따라서 revision 기준은 패널 수가 아니라 `주제별 묶음`이다.

appendix는 이제 아래 다섯 묶음으로 재정리하는 것이 맞다.

## 권장 Appendix Taxonomy

### A06 Structural Sanity Checks

역할:

- 본문 Q3의 구조 선택을 짧고 직접적으로 방어
- `semantic grouping`과 `temporal role assignment`만 남김

넣을 것:

- family prior intact vs reduced vs shuffled vs flattened
- original temporal role vs identical scope vs scope swap vs extra attention

넣지 않을 것:

- top-k
- sparse routing
- operating summary를 과도하게 섞는 패널

핵심 메시지:

- 지금 구조가 단순히 parameterization 때문이 아니라, semantic grouping과 temporal role 분리가 실제로 필요하다.

### A07 Top-k Routing Regimes

역할:

- `dense RouteRec이 왜 main choice인가`를 sparse/top-k와의 operating-regime 비교로 설명
- 성능표가 아니라 `quality-selectivity frontier`를 보여주는 appendix

넣을 것:

- quality vs active experts frontier
- global top-k sweep
- grouped top-k sweep
- average quality vs active experts vs top-share summary

넣지 않을 것:

- cue PCA
- behavior cases
- generic routing heatmap dump

핵심 메시지:

- sparse routing은 흥미로운 control이지만, 현재 paper setting에서는 dense routing이 가장 안정적인 operating point다.

### A08 Behavior Case Studies

역할:

- behavior slice를 paper-friendly case study로 번역
- `어떤 case가 gain을 만들고`, `그 case가 어떤 routing/cue 특성을 가지는가`를 보여줌

넣을 것:

- per-case grouped bar + line with only essential comparators
- gain vs routing concentration scatter
- one cue-space prototype panel or one cue-profile panel

넣지 않을 것:

- pure/permissive jargon-heavy split
- threshold robustness panel 여러 개
- top-k와 섞인 analysis

핵심 메시지:

- RouteRec gain은 모든 case에서 균일하지 않고, 해석 가능한 일부 behavior case에서 집중된다.

### A09 Boundary Conditions and Lightweight Portability

역할:

- `어디까지 버티고 어디서 약해지는가`를 compact하게 보여주는 boundary-condition appendix

넣을 것:

- low-data budget curves
- 매우 compact한 transfer-style reuse comparison

넣지 않을 것:

- 넓은 transfer matrix
- operating panel을 여러 장으로 늘리는 구성

핵심 메시지:

- low-data와 reuse setting은 RouteRec의 승리 선언이 아니라, 경계 조건을 점검하는 공간이다.

### A10 Cue Semantics and Intervention Analysis

역할:

- `metric appendix`가 아니라 `analysis appendix`
- feature family, portable cue setting, intervention response를 한 묶음으로 정리

넣을 것:

- portable cue family profile under different cue settings
- cue-setting별 routing sharpness summary
- intervention score drop + family-mass shift map

넣지 않을 것:

- stage-by-stage expert heatmap
- 구조 ablation과 섞인 패널
- low-data 결과와 혼합된 portability analysis

핵심 메시지:

- RouteRec의 router는 단순 hidden-state artifact가 아니라, semantic cue families와 intervention에 일관되게 반응하는 control interface다.

## 묶음 기준

appendix panel은 아래 셋 중 하나에만 속해야 한다.

1. `sanity check`: main design을 짧게 방어하는가
2. `operating regime`: dense vs sparse, rich vs selective의 tradeoff를 보여주는가
3. `analysis`: routing이 무엇에 반응하는지, 어떤 case를 설명하는지 보여주는가

이 셋 중 어디에도 명확히 속하지 않으면 빼는 편이 낫다.

## 지금 구조에서 줄여야 할 것

- metric 종류만 바꾼 유사 plot 반복
- subfigure마다 전혀 다른 시각 언어를 쓰는 구성
- 내부 코드베이스를 알아야 이해되는 naming
- 본문과 연결이 약한 generic diagnostic dump
- stage heatmap을 중심으로 한 appendix 구성

## 시각화 원칙

- x축은 변수 이름이 아니라 해석 문장처럼 써야 한다.
- bar + line은 `quality + behavior summary`처럼 역할이 분명할 때만 쓴다.
- heatmap은 `delta`나 `matrix-like relation`일 때만 쓴다.
- analysis appendix에는 scatter, projection, family-profile line plot 같은 비-metric 중심 plot을 의도적으로 섞는다.
- 한 notebook 안의 패널들은 같은 질문에 답해야 한다.

## 현재 권장 파일 매핑

- A06: structural sanity only
- A07: top-k regimes only
- A08: behavior cases only
- A09: low-data and compact transfer only
- A10: cue semantics and intervention analysis

## 바로 반영할 변경

- A06은 현재 방향 유지, 축 이름만 더 직관적으로 유지
- A07은 top-k 전용 notebook으로 고정
- A08은 case study + cue-space analysis만 남김
- A09는 transfer panel을 더 compact한 bar + line 방식으로 정리
- A10을 새로 만들어 `analysis-only appendix` 역할을 맡김

<details>
<summary>Archived Earlier Slot-by-Slot Notes</summary>

# RouteRec Appendix Strategy

## 목적

이 문서는 `writing/260418_final_exp_figure` 아래의 appendix 실험을 다시 설계하는 메모다.

기준은 명확하다.

- 본문 실험의 최신 기준점은 `sample-sigconf.tex`의 placeholder가 아니라 `02~05` notebook이다.
- appendix는 본문을 반복하는 공간이 아니라, 본문 claim이 약해 보일 수 있는 지점을 보강하는 공간이다.
- 따라서 appendix의 핵심은 `전체 서사`보다 `어떤 실험을 더 해야 하는가`, `어떤 비교군을 넣어야 하는가`, `어떻게 시각화해야 의도가 바로 읽히는가`다.

이번 revision에서는 특히 다음을 반영한다.

- appendix에는 baseline 및 대조군이 더 많이 들어가도 된다.
- appendix figure에도 `test metric bar + line` 비교를 적극적으로 써도 된다.
- dense routing이 너무 soft-mixing처럼 보일 수 있으므로 `top-k / sparse-like routing` 실험을 구조 appendix의 핵심 후보로 올린다.
- low-data 혹은 cutoff 실험은 가치가 있지만, MoE가 불리할 가능성도 있으므로 `경계 조건 확인`으로 프레이밍한다.
- 이미 저장된 aux-loss 및 diagnostics artifact는 최대한 재활용한다.

## 지금 상태의 전체 문제

현재 본문 02~05 notebook은 꽤 정리되어 있다.

- Q2는 routing source와 representative routing evidence를 직접 보여준다.
- Q3는 final structure가 reduced alternatives보다 낫다는 점에 집중한다.
- Q4는 portability를 metric retention보다 router operating behavior로 읽게 만든다.
- Q5는 family heatmap보다 semantic intervention과 gain vs selectivity를 더 직접적으로 보여준다.

반면 appendix 초안은 아직 몇 군데가 옛 placeholder 논리에 머물러 있다.

- A06은 구조 확장판이긴 하지만, 지금 더 중요한 `dense vs sparse-like routing`, `왜 current group prior가 과도하지 않은가`가 약하다.
- A07은 indirect diagnostic 비중이 높고, 본문에서 이미 밀어낸 consistency-type 분석을 여전히 중심에 놓고 있다.
- A08은 현재로서는 main Q5의 큰 확대판이라 appendix 추가 가치가 약하다.
- A09는 transfer를 많이 하려는 반면, 실제로는 low-data portability와 sparse routing의 boundary condition을 보는 편이 paper와 더 잘 맞는다.

3. routing diagnostic이 trivial artifact가 아니라는 방어
4. behavior slice claim의 dataset-level 및 threshold robustness 보강
5. low-data regime에서 RouteRec이 어디까지 유지되고 어디서 약해지는지 확인
### 1. appendix는 본문보다 더 많은 대조군을 써도 된다

본문은 압축해야 하지만 appendix는 그렇지 않다. 따라서 appendix figure에서는 다음이 허용된다.

- baseline 2~3개를 함께 넣는 비교
- `NDCG@20` bar + `HR@10` line 형태의 multi-variant plot
- dataset별 small multiple
- same claim에 대한 dense / hidden-only / shared FFN / RouteRec / sparse RouteRec 동시 비교

즉 appendix는 `깔끔한 1-message`보다 `비교군을 충분히 보여주는 보강`이 더 중요하다.

### 2. appendix는 average 하나로 끝내지 않는다

aggregate 평균만 반복하면 appendix가 약해진다. 가능하면 아래 중 하나를 추가해야 한다.

- dataset-level heterogeneity
- stage-level behavior
- capacity or sparsity sensitivity
- intervention response
- pure/permissive robustness

### 3. 현실적으로 가능한 실험만 넣는다

현재 현실 제약도 문서에 명시해야 한다.

- `group count = 8`까지 새로 늘리는 것은 현재 현실적으로 어렵다.
- 따라서 group count 축은 `4보다 더 적게` 보는 쪽이 현실적이다.
- 반대로 `top-k`, `topk_scope_mode`, `expert_scale`, `group_top_k` 계열은 이미 구현 흔적이 있고, 이전 실험도 존재하므로 appendix 후보로 훨씬 현실적이다.

### 4. negative result도 appendix에서 가치가 있다


를 보여주면 된다.

즉 appendix는 `항상 이긴다`가 아니라 `어디서 잘 되고 어디서 약한가`까지 보여줘도 된다.

### 5. existing artifact를 최대한 재활용한다

현재 이미 저장된 진단값이 많다.

- intervention shift placeholder schema

따라서 appendix 전략은 `새 logging을 최소화하고, 집계 및 시각화 설계를 더 잘하는 쪽`으로 잡는 것이 좋다.

## dataset 및 비교군 우선순위

appendix에서 모든 dataset를 다 같은 비중으로 다루기보다, claim별로 대표 dataset를 고르는 편이 낫다.
- KuaiRec
- Retail Rocket

이 세 dataset는 본문 narrative상 RouteRec gain이 더 잘 읽히고, sparse/dense routing의 차이도 상대적으로 해석하기 좋다.

### boundary-condition 확인용 dataset

- ML-1M
- LastFM
### 기본 comparator

- `best baseline`
- `shared_ffn`
- `RouteRec dense` (현재 main 설정)
### 지금 상태의 문제

이 슬롯은 실험 설계보다는 결과 정리 슬롯이다. 따라서 여기서 큰 디자인 논의를 할 필요는 없다.

### 어떻게 개선할지

- main table과 동일한 selection rule 유지
- full cutoff를 appendix에서 복원
- 필요하면 sparse routing variant 한두 개의 full cutoff는 작은 supplementary table로만 추가
- strongest baseline
- sparse routing 대표 1개만 추가 가능하면 추가

### 어떻게 시각화할지

- 기본은 table
- notebook은 styled dataframe까지만 준비

- A05는 redesign 우선순위가 낮다.
- 대신 다른 A 슬롯에서 새 실험이 확정되면, 그 결과의 full cutoff를 여기에 흡수하는 구조가 맞다.

## A06 Extended Structural Ablations

### 지금 상태의 문제

현재 A06은 structural variant를 넓게 다루지만, 지금 reviewer가 더 물을 지점은 따로 있다.

- dense routing이 정말 RouteRec 의도와 맞는가
- current capacity가 단순히 크기만 해서 이긴 것은 아닌가

반면 `8 groups까지 sweep`은 현실적으로 어렵다. 이 축을 무리하게 키우는 것은 비효율적이다.

A06은 `order/search extension`보다 `routing sparsity and capacity behavior` 중심으로 재설계하는 편이 좋다.

핵심은 세 패널이다.

1. semantic family prior validity
2. stage semantics validity
3. dense vs sparse-like routing and active-capacity sensitivity
#### A06-(a) semantic grouping validity

유지할 실험:

- full semantic grouping
- reduced family set
- shuffled family assignment
이 패널은 `semantic family prior가 정말 의미가 있는가`를 보여준다.

#### A06-(b) stage semantics and layout validity

유지 또는 소폭 확장:

- original scope
- identical scope
- scope swap

이 패널은 `3 stage가 있다는 사실`보다 `각 stage가 다른 temporal role을 맡는다`를 보여준다.

#### A06-(c) dense vs top-k / sparse-like routing

이 패널이 이번 revision의 핵심이다.

- global top-k: top-2 / top-4 / top-8
- group-dense: 모든 group 사용, group 내부는 dense
- group-top1-pergroup: 모든 group 사용, 각 group에서 expert 1개만 활성
- group-top2-pergroup: 모든 group 사용, 각 group에서 expert 2개 활성

가능하면 추가할 실험군:

- active group 2개만 사용 + 내부 dense
- active group 3개만 사용 + 내부 dense

- dense global
- global top-2
- global top-4
- group-dense
- group-top1-pergroup
- group-top2-pergroup

이 축은 줄여서 본다.

- `2 groups` vs `4 groups`
- 혹은 `expert_scale 1/2/3/4` 정도

여기서 중요한 건 `8 groups`까지 새로 열지 않는다는 점을 문서에 명시하는 것이다. 이건 공격 포인트가 될 수 있지만, 지금은 현실적으로 받아들이는 편이 맞다.

#### A06-(a), A06-(b)

- dataset별 `NDCG@20` bar + `HR@10` line small multiples
- appendix에서는 comparator가 많아도 된다.

- panel left: dataset별 `NDCG@20` bar + `HR@10` line for dense/top-k variants
- panel right: routing richness summary
  - x축: variant
  - y축 1: `effective experts`
  - y축 2: `entropy` or `top-share`

즉 단순 성능 비교가 아니라 `얼마나 sparse하게 썼는지`를 같이 읽게 해야 한다.

### 왜 이 구성이 좋은가

- RouteRec이 dense soft mix라서 MoE스럽지 않다는 우려를 정면으로 다룬다.
- sparse하게 해도 되는지, 오히려 더 나쁜지, 어느 sparsity가 타협점인지 보여줄 수 있다.
- negative result가 나와도 `dense를 선택한 이유`를 appendix에서 방어할 수 있다.

## Objective / Aux Loss / Regularization

### 지금 상태의 문제

현재 objective appendix는 table placeholder 수준이다. 그런데 이 슬롯은 본문 Q2/Q5 해석을 방어하는 데 중요하다.
가 여기서 다뤄져야 한다.

### 어떻게 개선할지

이 슬롯은 figure보다 table 중심이 맞다. 대신 table 내부 의미는 더 선명해야 한다.

### 어떤 실험을 할지

기본 variant:

- CE only
- CE + consistency
- CE + z-loss
- CE + balance
- CE + consistency + z-loss
- full objective

가능하면 sparse routing 대표 1개에 대해서도 아래 둘만 짧게 확인한다.

- dense full objective
- sparse representative full objective

이건 `sparse routing이 instability 때문에 지는지`를 보는 보조 evidence가 된다.

### 어떤 값을 넣을지

main column은 3축이면 충분하다.

- ranking: `MRR@20` 또는 `NDCG@20`
- semantics: route consistency score
- stability: `entropy_mean`, `n_eff floor`, `max logit scale`, 또는 failure count 중 하나

실제로는 CSV에 더 많은 raw value를 넣고, paper table은 요약형으로 만드는 편이 좋다.

### 어떻게 시각화할지

- paper에는 compact table
- notebook 내부에는 heatmap-style styled dataframe
- 필요하면 보조 bar+line plot 하나로 `quality vs stability`만 확인

### 왜 이 구성이 좋은가

- 이미 저장된 diagnostics를 재활용하기 좋다.
- 본문의 routing 해석이 auxiliary loss artifact가 아니라는 방어가 가능하다.

## A07 Routing Diagnostics

### 지금 상태의 문제

현재 A07은 consistency 및 feature-bucket 같은 indirect evidence가 중심이다. 지금 본문 논리는 더 직접적이므로 appendix도 거기에 맞춰야 한다.

### 어떻게 개선할지

A07은 `router가 어떻게 움직이는가`를 보여주는 쪽으로 재구성한다.

핵심은 다음 세 가지다.

1. stage/family usage
2. dense vs sparse or full vs reduced-cue operating behavior
3. semantic intervention이 routing mass를 어떻게 바꾸는가

### 어떤 실험을 할지

#### A07-(a) stage-wise family usage heatmap

기존 expert id heatmap보다 family heatmap이 낫다.

- row: dataset 또는 behavior slice
- column: `macro/mid/micro x family`
- value: mean routing mass

#### A07-(b) routing richness comparison

한 모델 절대값이 아니라 comparator를 둔다.

추천 비교군:

- `RouteRec dense`
- `global top-k representative`
- `group-top1-pergroup`
- `sequence-only portable`
- `hidden-only` 또는 `shared_ffn` 중 1개

이 패널은 `effective experts`, `entropy`, `group_n_eff`, `top-share`를 비교한다.

#### A07-(c) semantic intervention shift map

현재 `05_feature_intervention_shift.csv`가 잘 맞는다.

- row: intervention
- column: family 또는 `stage x family`
- value: routing mass delta

이건 main Q5 intervention performance와 바로 연결된다.

#### A07-(d) optional legacy consistency support

이건 optional이다.

- similarity bucket vs consistency
- pure reviewer support용

공간이 부족하면 이 패널은 버리고, 대신 `cue family profile shift under cue removal`을 넣는 편이 더 좋다.

### 어떻게 시각화할지

#### A07-(a)

- heatmap
- dataset 또는 slice row 정렬을 narrative 순서에 맞춘다.

#### A07-(b)

- appendix에서는 `bar + line`을 적극적으로 써도 된다.
- 예: bar=`effective experts`, line=`entropy`, category=`variant`, small multiple=`dataset`

#### A07-(c)

- heatmap 또는 diverging heatmap
- positive/negative delta가 직관적으로 보여야 한다.

### 왜 이 구성이 좋은가

- 이미 저장된 diag 값이 많아서 새 실험 비용이 낮다.
- dense vs sparse-like routing 차이를 내부 동작으로 설명할 수 있다.
- Q5 semantic intervention panel을 appendix에서 더 설득력 있게 받쳐준다.

## A08 Behavior Slices and Qualitative Cases

### 지금 상태의 문제

현재 A08은 main Q5를 조금 더 크게 보는 수준이다. appendix만의 추가 정보가 약하다.

### 어떻게 개선할지

A08은 `dataset-level heterogeneity`와 `threshold robustness`를 보여주는 쪽으로 바꾼다.

또한 `pure / permissive` tier를 분리해서 쓰는 것이 중요하다.

- `pure`: 본문 primary evidence
- `permissive`: appendix robustness evidence

### 어떤 실험을 할지

#### A08-(a) per-dataset slice quality comparison

여기서는 appendix답게 comparator를 충분히 넣는다.

- x축: behavior slice
- bar: `NDCG@20`
- line: `HR@10`
- hue/model: `best baseline`, `shared_ffn`, `RouteRec dense`, 필요 시 `sparse representative`
- small multiple: dataset

즉 본문보다 더 직접적으로 `slice별 baseline 대비 차이`를 보여준다.

#### A08-(b) per-dataset gain heatmap

- row: dataset
- column: slice
- value: RouteRec gain over comparator

이건 `어느 dataset에서 어떤 regime가 특히 중요했는가`를 빠르게 보여준다.

#### A08-(c) pure vs permissive robustness curves

각 slice family에 대해

- pure tier gain
- permissive tier gain
- route concentration

을 함께 본다.

#### A08-(d) percentile trend view

binary slice threshold 대신 quantile trend도 좋다.

- repeat ratio quantile bins
- mean gap quantile bins
- focus entropy quantile bins
- switch rate quantile bins

그리고 각 bin에서 아래를 본다.

- RouteRec gain over baseline
- route concentration

이건 appendix에서 cherry-pick 우려를 줄이는 데 유용하다.

#### qualitative cases

tex의 qualitative case table은 A08과 묶어서 준비한다.

추천 사례:

- repeat-heavy session
- fast-tempo exploratory session
- narrow-focus session
- preference-stable session

### 어떻게 시각화할지

- (a)는 dataset small-multiple grouped `bar + line`
- (b)는 heatmap
- (c), (d)는 line or paired line
- qualitative case는 compact table + small routing strip

### 왜 이 구성이 좋은가

- appendix에서도 baseline 비교가 충분히 들어간다.
- 본문 Q5가 average claim에만 의존하지 않는다는 점을 보여준다.
- pure/permissive를 분리하면 본문과 appendix 역할도 자연스럽게 나뉜다.

## A09 Low-Data and Portability Variants

### 지금 상태의 문제

현재 A09는 transfer 전체를 너무 넓게 잡고 있다. 하지만 지금 paper narrative에서는 `low-data에서 얼마나 유지되는가`, `dense보다 sparse-like routing이 오히려 나은가`가 더 중요하다.

### 어떻게 개선할지

A09는 `low-data / cutoff / portability boundary condition` 중심으로 재구성한다.

핵심은 다음이다.

- 적은 데이터에서 RouteRec dense가 얼마나 버티는가
- sparse-like routing이 그 setting에서 도움이 되는가
- sequence-only portable cue가 low-data에서 오히려 덜 무너지는가

### 어떤 실험을 할지

#### A09-(a) low-data cutoff curves

추천 training ratio:

- 5%
- 10%
- 20%
- 가능하면 50%

추천 비교군:

- `shared_ffn`
- `RouteRec dense`
- `RouteRec sequence-only`
- `RouteRec sparse representative`

여기서 sparse representative는 아래 둘 중 하나면 충분하다.

- `global top-k` best one
- `group-top1-pergroup`

이 패널은 negative result도 괜찮다. 오히려 `dense MoE는 low-data에서 더 불안정할 수 있다`는 경계 조건을 보여줄 수 있다.

#### A09-(b) cutoff-specific operating behavior

low-data 성능 하락이 왜 생기는지 같이 봐야 한다.

- x축: training ratio
- bar: `NDCG@20`
- line: `effective experts` 또는 `entropy`
- model별 line split

이렇게 하면 데이터가 줄수록 router가 collapse하는지, 과도하게 diffuse한지 읽을 수 있다.

#### A09-(c) optional lightweight transfer

transfer는 줄여서 본다.

- full matrix는 후순위
- representative pair 2~3개만
- related pair 1개, mismatched pair 1개 정도

비교군은 다음 정도면 충분하다.

- finetune all
- freeze router
- anchor init
- group-level router reuse

이건 진짜 시간이 남을 때만 한다.

### 어떻게 시각화할지

#### A09-(a)

- training ratio x metric curve
- appendix에서는 model 4개까지 한 plot에 넣어도 된다.

#### A09-(b)

- dual-axis line plot
- 또는 left quality / right entropy, n_eff pair plot

#### A09-(c)

- grouped `bar + line`
- 혹은 very small matrix table

### 왜 이 구성이 좋은가

- Q4 portability claim을 더 harsh한 환경에서 점검할 수 있다.
- sparse routing이 low-data에서 오히려 regularization처럼 작동하는지 볼 수 있다.
- RouteRec의 약점이 드러나도 appendix narrative로는 충분히 유효하다.

## 각 슬롯별 바로 실행할 실험 우선순위

### 1차 우선순위

1. A06 dense vs sparse-like routing
2. objective / aux-loss table
3. A07 intervention-aware routing diagnostics

### 2차 우선순위

4. A08 per-dataset behavior slices with strong comparators
5. A09 low-data cutoff curves

### 3차 우선순위

6. A09 lightweight transfer pairs
7. A06 reduced group-count sweep

## 필요한 export 및 집계 체크리스트

### 이미 있는 값에서 바로 가져오기 쉬운 것

- `router_diag`의 `entropy_mean`, `n_eff`, `group_n_eff`, `top-share`
- `special_metrics`의 slice metric
- objective variant의 diagnostics summary
- intervention shift placeholder schema

### 새로 집계가 필요한 것

- dense / sparse routing variant summary CSV
- low-data ratio별 metric summary CSV
- low-data ratio별 routing richness CSV
- per-dataset slice gain heatmap용 CSV
- pure/permissive split summary CSV

### 특히 재활용 가치가 높은 현재 asset

- `data/04_cue_family_profile.csv`
- `data/05_feature_intervention_shift.csv`
- `data/A08_behavior_slice_quality.csv`
- `data/A08_behavior_slice_gain.csv`

## 최종 판단

이번 appendix에서 가장 중요한 변화는 두 가지다.

- 구조 appendix의 중심을 `order variant 추가`에서 `dense vs sparse-like routing`으로 옮기는 것
- portability appendix의 중심을 `넓은 transfer`에서 `low-data cutoff and boundary condition`으로 옮기는 것

이렇게 바꾸면 appendix는 지금 본문 02~05 notebook의 논리와 더 잘 맞는다.

- Q3는 `왜 이 구조인가`를 더 강하게 방어할 수 있고
- Q4는 `적은 신호와 적은 데이터에서 어디까지 버티는가`를 더 직접적으로 볼 수 있고
- Q5는 `semantic intervention과 behavior regime`를 appendix에서 더 설득력 있게 확장할 수 있다.

즉 다음 단계에서 ipynb 예제를 다시 만들 때는, A06과 A09를 특히 크게 바꾸는 것이 맞다.

</details>