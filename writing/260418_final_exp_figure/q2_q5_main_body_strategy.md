# RouteRec Main-Body Experiment Strategy for Q2-Q5

## 목적

이 문서는 본문 Q2-Q5를 지금 draft 기준에서 다시 설계한 메모다.

핵심 질문은 단순히 “무슨 실험을 더 넣을까”가 아니다.

- reviewer가 RouteRec을 왜 믿어야 하는가
- 각 본문 figure가 어떤 claim을 직접적으로 증명하는가
- 어떤 그림은 본문에 두고, 어떤 그림은 appendix나 case study로 보내야 하는가

이 문서는 [writing/260417_references/meeting_summary_only_260417.txt](/workspace/FeaturedMoE/writing/260417_references/meeting_summary_only_260417.txt)의 피드백을 중심으로, [writing/ACM_template/sample-sigconf.tex](/workspace/FeaturedMoE/writing/ACM_template/sample-sigconf.tex)의 현재 Q2-Q5 구성을 비판적으로 다시 정리한 것이다.

## 260418 revision note

최근 피드백 기준으로는 다음이 중요하다.

- Q2는 패널 수를 늘리는 대신 dataset 안에서 variant를 비교하고, 나머지 diagnostic은 stage-average로 더 압축해야 한다.
- Q3는 stage-removal과 stage-count가 겹치므로 main text에서는 `final vs reduced-stage alternatives` 하나로 정리하는 편이 낫다.
- Q4는 metric 외의 내부 동작 근거가 필요하며, 현재로서는 family profile보다 `routing spread`와 `effective experts` 같은 footprint summary가 더 직접적이다.
- Q5는 family-level response heatmap보다, 실제 behavioral regime에서 RouteRec gain이 routing selectivity와 함께 커지는지를 보여주는 편이 논지에 맞다.
- 별도로 method 쪽에서 `왜 4 semantic groups이고 왜 group당 4 experts 근처가 적절한가`를 방어할 supplementary experiment가 필요하다.

## 전체 판단

현재 Q2-Q5의 큰 방향 자체는 맞다. 즉 본문을 아래 네 질문으로 구성하는 것은 유지해도 된다.

1. routing signal은 무엇이어야 하는가
2. staged structure는 실제로 필요한가
3. lightweight cue만으로 충분한가
4. learned routing이 실제 behavioral regime와 맞물리는가

하지만 현재 placeholder 설명은 몇 군데에서 너무 간접적이거나, 논문 독자가 보고 바로 이해하기 어려운 분석에 기대고 있다.

특히 Q2의 `feature-similarity bucket vs routing consistency`는 다음 문제가 있다.

- 모델이 의도한 바를 간접적으로만 보여준다.
- “그럴 듯하다” 수준이지, reviewer가 강하게 납득할 근거로 보기 어렵다.
- similarity 정의와 consistency 정의가 모두 후처리 metric이라, 직관적인 설득력이 약하다.
- 미팅 메모에서도 이 계열 분석은 case study보다 전달력이 떨어질 수 있다는 의견이 있었다.

따라서 본문은 다음 원칙으로 다시 구성하는 것이 좋다.

- 본문은 가능한 한 직접적인 claim 검증에 집중한다.
- 간접 diagnostic은 appendix로 보낸다.
- qualitative 또는 semi-qualitative evidence를 적어도 한 번은 본문에 포함한다.
- feature / stage / routing이라는 세 축이 각각 독립적으로 드러나야 한다.

## Q2. What Should Control Routing?

## 현재 draft의 문제

현재 Q2는 다음 두 패널이다.

- routing control별 average ranking quality
- feature-similarity bucket 대비 route consistency

이 중 첫 번째는 괜찮다. 하지만 두 번째는 본문용으로는 약하다.

이유는 다음과 같다.

- behavior-guided routing의 장점을 “유사한 feature끼리 routing이 비슷하다”로 보여주는 것은 너무 우회적이다.
- reviewer 입장에서는 이 그림을 보고도 “그래서 실제로 feature가 routing을 meaningful하게 바꿨다는 근거가 뭐지?”라는 의문이 남을 수 있다.
- consistency가 높을수록 좋은 것인지도 모델 없이 바로 직관적으로 읽히지 않는다.

## 본문에서 더 설득력 있는 방향

Q2의 본질은 “routing source를 feature로 두는 것이 hidden-only보다 낫다”를 보여주는 것이다. 따라서 본문에서는 다음 두 층을 같이 보여주는 편이 낫다.

### 추천 main claim

- 정량: behavior-guided routing이 hidden-only / mixed / shared FFN보다 더 좋다.
- 정성: behavior-guided routing이 실제 behavioral difference에 맞춰 다른 expert allocation을 만든다.

### 추천 figure 구성

#### Q2 panel (a): routing control performance

유지해도 좋다.

- x축: dataset
- 막대: `NDCG@20`
- 선: `HR@10`
- 비교군: `shared_ffn`, `hidden_only`, `mixed`, `behavior_only`

이 패널은 “성능상으로 무엇이 좋은가”를 바로 보여준다.

#### Q2 panel (b): case-oriented routing evidence

현재 consistency bucket 대신 아래 둘 중 하나로 교체하는 것을 권장한다.

##### 선택지 1. 대표 user/session case study

- 3~4개 representative prefix를 고른다.
- 각 prefix에 대해 macro/mid/micro stage의 top expert 또는 expert probability bar를 보여준다.
- 각 사례 위에는 간단한 설명을 붙인다.
  - 예: repeat-heavy prefix
  - fast switching prefix
  - narrow-focus prefix
  - exploratory prefix

이 방식의 장점은 다음과 같다.

- 독자가 바로 “아 이런 행동에서는 이런 expert를 쓰는구나”를 이해할 수 있다.
- feature-guided routing의 의미가 hidden latent space가 아니라 observable behavior에 대응된다는 점이 잘 드러난다.
- 미팅 메모의 “직관적인 evidence가 필요하다”는 요구와 맞는다.

##### 선택지 2. user group별 routing profile

- user/session을 외부 통계로 3~4개 group으로 나눈다.
- 예: repeat-heavy, fast-tempo, focused, exploratory
- 각 group에 대해 stage별 평균 expert usage heatmap 또는 family-level usage bar를 보여준다.

이 방식의 장점은 다음과 같다.

- case study보다 더 aggregate evidence다.
- “몇 개 사례만 cherry-pick한 것 아니냐”는 방어가 가능하다.

### Q2 본문 추천안

가장 좋은 조합은 아래다.

- panel (a): routing control performance
- panel (b): group-level routing profile 또는 representative case study

이때 current consistency-bucket plot은 appendix diagnostics로 내리는 것이 좋다.

## Q2에서 appendix로 내릴 것

- feature similarity bucket vs consistency
- route consistency metric 자체의 정의와 추가 curve
- hidden+behavior reinjection의 세부 diagnostic

즉 consistency는 버릴 것이 아니라, 본문 main evidence에서 appendix supporting evidence로 내리는 것이 적절하다.

## Q3. Why Is the Staged RouteRec Structure Effective?

## 현재 draft의 장점

Q3는 현재 방향이 가장 안정적이다. 미팅 메모와도 잘 맞는다.

- stage removal
- dense vs staged
- order / wrapper variants

이 구조는 “expert 많이 넣어서 좋아진 것 아님”을 방어하는 데 직접적이다.

## 다만 보완할 점

현재 Q3 설명은 구조 실험이 많아 보이지만, 핵심 메시지는 둘뿐이다.

1. stage를 나눠야 한다.
2. 단순 dense replacement가 아니라 temporal role separation이 중요하다.

order / wrapper variant는 중요하지만, 너무 전면에 오면 독자가 “이 논문이 구조 튜닝 논문인가?”로 읽을 수 있다.

## 본문 추천 방향

### 추천 main claim

- stage removal에서 성능이 떨어진다.
- single / two / three stage 비교에서 full staged design이 가장 좋다.
- 이는 단순 expert insertion이 아니라 time-scope decomposition의 효과다.

### 추천 figure 구성

#### Q3 panel (a): stage removal

유지 권장.

- full
- remove macro
- remove mid
- remove micro

이 패널은 “각 stage가 dead weight가 아니다”를 직접 보여준다.

#### Q3 panel (b): dense vs staged

유지 권장.

- dense FFN
- best single-stage
- best two-stage
- three-stage full

이 패널은 가장 중요하다. reviewer가 가장 먼저 궁금해할 “그냥 FFN 대신 MoE 넣어서 좋아진 것 아냐?”를 막아준다.

#### Q3 panel (c): order / wrapper는 축소

현재처럼 order와 wrapper를 한 패널에 다 넣기보다, 아래처럼 압축하는 쪽이 낫다.

- 본문: final vs one meaningful order variant vs one meaningful wrapper variant
- 나머지 세부 variant: appendix

즉 panel (c)는 “current final design이 완전 임의는 아니다” 정도만 보여주면 충분하다.

### Q3에서 강조할 해석

- macro / mid / micro가 각각 다른 behavioral timescale을 담당한다.
- 그래서 performance gain은 capacity increase보다 decomposition benefit에서 나온다.
- 이 해석이 challenge 2와 method의 multi-stage story와 직접 연결된다.

## Q3에서 appendix로 내릴 것

- full order search
- wrapper 3종 전체 sweep
- cue grouping shuffle / random group variant
- stage placement 세부 비교

## Q4. Are Lightweight Cues Sufficient in Practice?

## 현재 draft의 장점

Q4는 meeting memo와 매우 잘 맞는다. feature contribution을 강하게 보여주라는 피드백과 정확히 연결된다.

## 본문에서 더 강하게 보여줘야 할 것

Q4는 단순히 “몇 개 cue를 뺐더니 점수가 조금 떨어졌다”가 아니다. 본문에서는 아래 메시지를 명확히 해야 한다.

- RouteRec의 feature 설계는 heavy metadata dependency가 아니다.
- ordinary logs만으로도 routing benefit이 유지된다.
- 즉 portable control interface라는 claim이 성립한다.

## 추천 figure 구성

#### Q4 panel (a): cue reduction performance

유지 권장.

- `full`
- `remove_category`
- `remove_time`
- `sequence_only`

다만 단순 `MRR@20` 평균만 보여주는 것보다, 본문용 디자인은 다음이 더 좋다.

- 막대: `NDCG@20`
- 선: `HR@10`

즉 “성능이 얼마나 남는가”를 더 입체적으로 보여준다.

#### Q4 panel (b): routing footprint without metadata

현재는 이쪽이 더 낫다.

- x축에 cue setting을 두고, y축으로 `route entropy`와 `effective experts`를 각각 보여준다.
- dataset별 선을 그려서 full -> remove_category -> remove_time -> sequence_only로 갈 때 router가 급격히 collapse하지 않는지 본다.

이 패널의 장점은 다음과 같다.

- ranking metric을 반복하지 않는다.
- metadata를 빼도 router가 trivial path로 수축하지 않는다는 내부 동작 evidence가 된다.
- portability claim과 더 직접적으로 연결된다.

### Q4에서 추가로 고려할 것

본문 안에서 문장으로 아래 해석이 꼭 들어가야 한다.

- Beauty처럼 category-rich dataset에서는 category removal impact가 클 수 있다.
- Retail Rocket처럼 sparse setting에서는 sequence-only가 상대적으로 더 현실적인 deployment setting이다.

즉 Q4는 단순 ablation이 아니라 dataset-specific interpretation이 붙어야 한다.

## Q4에서 appendix로 내릴 것

- ONLY_MEMORY, ONLY_EXPOSURE 등 세부 family decomposition
- portable subset 전체 sweep

## Q5. Do Routing Patterns Align with Behavioral Regimes?
현재 main text에 더 맞는 panel (b)는 family response map보다 아래다.

#### Q5 panel (b): regime-selective routing gains

- slice별로 `route concentration` 혹은 `routing selectivity`를 x축에 둔다.
- y축에는 RouteRec의 relative gain을 둔다.
- slice label은 `repeat-heavy`, `fast-tempo`, `focused`, `exploration-heavy` 정도가 적절하다.

이 패널이 좋은 이유는 다음과 같다.

- “어느 family가 반응했다”보다, “routing이 잘 작동하는 regime에서 실제 gain이 커진다”는 더 직접적인 claim으로 간다.
- reviewer가 `MoE가 실제로 필요했나?`를 물을 때 더 바로 답할 수 있다.
- intervention panel (a)와 함께 두면, synthetic perturbation과 real held-out regime evidence가 짝을 이룬다.

## Future method-defense: why 4 groups x 4 experts?

이 질문은 method 설명에서 충분히 나올 수 있으므로, 본문이 아니더라도 supplementary 실험 계획은 미리 잡아두는 편이 좋다.

우선순위가 높은 방어 실험은 아래 네 가지다.

- `semantic group count`: 2, 4, 8 groups 비교. 여기서 4가 해석성과 성능의 균형인지 확인한다.
- `experts per group`: 1, 2, 4, 8 experts 비교. 이때 total experts 고정 실험과 group당 capacity 증가 실험을 분리하는 편이 좋다.
- `semantic vs shuffled grouping`: 같은 expert 수에서 semantic grouping이 random grouping보다 낫다는 것을 보여준다.
- `group-conditional vs flat experts`: 4x4 구조가 단순 16 flat experts보다 좋은지 확인한다.

이 축은 지금 바로 main figure로 넣기보다, method question을 방어하는 appendix or rebuttal-ready evidence로 준비하는 것이 적절하다.

## 현재 draft의 방향은 맞지만, 이것도 잘못하면 간접적이 된다

Q5는 본문에서 가장 논문다운 figure가 될 수 있다. 하지만 조건이 있다.

- slice definition이 외부 통계여야 한다.
- slice별 gain과 routing sharpness가 함께 보여야 한다.
- 단순히 “우리 모델이 repeat-heavy에서 좋다”가 아니라 “왜 그런가”가 드러나야 한다.

## 본문 추천 방향

Q5는 사실상 Q2의 qualitative/group evidence를 더 aggregate하게 확장한 그림처럼 설계하는 것이 좋다.

### 추천 main claim

- RouteRec gain은 behaviorally heterogeneous slice에서 특히 크다.
- 그 slice에서 routing은 더 decisive하다.
- 즉 model motivation과 learned behavior가 align한다.

### 추천 figure 구성

#### Q5 panel (a): slice-wise performance

유지 권장.

- x축: repeat-heavy / fast-tempo / focused / exploration-heavy
- hue: `best baseline`, `shared_ffn`, `RouteRec`
- 막대: `NDCG@20`
- 선: `HR@10`

여기서 `shared_ffn`를 함께 두는 것이 중요하다.

- baseline 대비 gain만이 아니라
- routing mechanism이 없는 ablated RouteRec 대비 gain도 보여줄 수 있기 때문이다.

#### Q5 panel (b): gain vs routing concentration

유지 가능하지만, 표현은 더 직관적으로 바꾸는 것이 좋다.

현재 draft의 “relative gain and routing concentration”은 괜찮지만, 실제 plot은 scatter 또는 paired bar보다 다음 형식이 더 좋다.

- x축: slice
- 왼쪽 y축: relative gain
- 오른쪽 y축: routing concentration 또는 inverse entropy

즉 slice별로 두 값이 같이 올라가는지 바로 읽히게 한다.

또는

- scatter plot에서 x축 concentration, y축 gain, label은 slice

이 방법도 가능하지만, 본문에선 category형 paired bar/line이 더 읽기 쉽다.

## Q5의 본문 가치

Q5는 단순 diagnostic이 아니라, introduction의 motivation figure와 method claim을 닫아주는 역할을 해야 한다.

즉 독자가 마지막에 얻어야 하는 인상은 다음이다.

- RouteRec은 그냥 평균적으로 조금 나은 모델이 아니다.
- 특정 행동 양상에서 computation path를 다르게 써서 이득을 본다.

이 메시지가 살아야 논문 제목과 abstract가 설득력을 가진다.

## Q5에서 appendix로 내릴 것

- slice threshold 민감도
- 더 많은 dataset
- 더 세분화된 group

## 본문 Q2-Q5의 추천 최종 구조

## 추천 구성안 A

가장 균형이 좋은 안이다.

### Q2

- panel (a): routing control performance
- panel (b): representative case study 또는 group-level routing profile

### Q3

- panel (a): stage removal
- panel (b): dense vs staged
- panel (c): compressed order/wrapper comparison

### Q4

- panel (a): full vs remove category vs remove time vs sequence-only
- panel (b): retention

### Q5

- panel (a): slice-wise performance
- panel (b): slice-wise gain + concentration

## 추천 구성안 B

만약 본문 페이지가 부족하면 Q2를 더 공격적으로 줄일 수 있다.

### Q2

- panel (a): routing control performance only
- panel (b): 없애고 case study를 독립 작은 figure 또는 boxed example로 처리

이 안은 공간은 절약되지만, qualitative evidence를 별도 위치에 넣어야 한다.

## 무엇을 본문에서 빼야 하는가

현재 기준으로 본문에서 빼는 것이 좋은 항목은 아래다.

1. feature similarity bucket vs routing consistency
2. 너무 세세한 wrapper/order sweep
3. objective regularization 세부 비교
4. diagnostics 전량

이것들은 모두 useful하지만, main claim을 직접적으로 증명하는 그림은 아니다.

## 무엇을 본문에서 반드시 남겨야 하는가

1. hidden-only 대비 behavior-guided superiority
2. dense replacement가 아니라 staged decomposition benefit
3. sequence-only portable cue에서도 gain retention
4. behavior slice에서 gain과 routing sharpness alignment
5. 적어도 한 번의 직관적인 qualitative/group evidence

이 다섯 가지가 본문 설득력의 핵심이다.

## 실무적으로 권장하는 바로 다음 단계

### 1. Q2 재설계

현재 consistency bucket plot은 appendix 후보로 내리고, 대신 아래 둘 중 하나를 본문 후보로 설계한다.

- representative case study panel
- behavior group별 stage-wise routing profile panel

### 2. Q3 panel (c) 축소

본문은 2~3개 variant만 남기고, 나머지는 appendix로 넘긴다.

### 3. Q5를 본문 핵심 figure로 승격

Q5가 abstract/motivation과 가장 강하게 연결되는 증거이므로, 단순 diagnostic처럼 다루지 말고 본문 핵심 메시지 figure로 강화한다.

## 한 줄 결론

본문 Q2-Q5는 유지해도 되지만, 현재 구성을 그대로 쓰기보다는 다음처럼 바꾸는 것이 더 설득력 있다.

- Q2는 간접 consistency 대신 직관적인 case/group evidence로 강화
- Q3는 stage necessity에 집중하고 세부 구조 sweep은 appendix로 축소
- Q4는 portable cue story를 dataset 해석과 함께 강조
- Q5는 behavior-aware routing의 semantic payoff를 보여주는 본문 핵심 figure로 사용

가장 중요한 변화는, 본문에서 “그럴 듯한 diagnostic”보다 “직접 읽히는 evidence”를 더 많이 보여주는 것이다.