# RouteRec / FMoE motivation 정리

이 문서는 `dataset_router_motivation.md`에서 정리한 raw-log 축과 현재 본문 실험 패턴을 바탕으로, **논문의 motivation / introduction / method / experiment / appendix를 어떤 서사로 엮을지**를 정리한 메모입니다. 목표는 다음 두 가지입니다.

1. motivation이 **결과를 보고 사후적으로 끼워 맞춘 것처럼 보이지 않게** 만들기
2. 그럼에도 **왜 router input이 hidden-only가 아니라 behavior-aligned signal이어야 하는지**를 자연스럽게 설명하기

---

## 0. 가장 먼저 정할 핵심 문장

이 논문이 메인으로 주장해야 하는 것은 아래입니다.

> **RouteRec is not mainly a claim that sequential recommendation universally requires radically different computation paths. It is a claim that, once MoE-based conditional computation is introduced, expert allocation should be guided by routing signals aligned with recurring behavioral axes in sessionized interaction logs, rather than by prediction-optimized hidden states alone.**

즉,

- **“왜 path를 나눠야 하냐”**를 너무 세게 주장하지 않는다.
- 대신 **“MoE를 쓰기로 했을 때, 무엇으로 expert partition을 정의할 것이냐”**를 메인 질문으로 둔다.
- hidden state가 틀렸다고 하는 게 아니라, **expert specialization axis와 hidden geometry가 반드시 잘 정렬되지는 않는다**고 말한다.

이 framing의 장점은 다음과 같다.

- hidden-only 대비 성능이 압도적으로 크지 않아도 괜찮다.
- interpretability가 아주 강하게 나오지 않아도 괜찮다.
- MoE literature와도 더 자연스럽게 연결된다.

---

## 1. headline 용어: heterogeneity보다 `behavioral routing demand`

### 추천 용어

- 메인 용어: **behavioral routing demand**
- 보조 용어: **recurring behavioral axes**
- 피하고 싶은 용어: 그냥 **heterogeneity** 하나로 퉁치는 표현

### 왜 이렇게 가는가

`heterogeneity`만 쓰면 너무 넓다. “데이터가 다양하다” 정도로 들릴 수 있다. 반면 `routing demand`라고 하면 질문이 바로

- 언제 shared-path보다 conditional routing이 의미가 있는가?
- 그리고 그 demand는 raw logs의 어떤 축에서 보이는가?

로 구체화된다.

### 논문에서의 역할 분리

- **behavioral axes**: routing demand가 어디서 생기는지 설명하는 개념적 축
- **behavioral routing demand**: 그 축들이 강할 때 RouteRec류 방법이 더 도움이 될 여지가 있다는 해석 프레임
- **behavioral cues**: 위 축들을 sample/session/prefix 수준에서 operationalize한 실제 router input

즉,

- dataset-level descriptor = 왜 routing이 필요해 보이는가
- sample-level cue = 지금 이 입력을 어떻게 routing할 것인가

로 분리하는 것이 핵심이다.

---

## 2. MoE literature를 어떻게 끌어올 것인가

MoE 논문들은 크게 두 부류로 나눠서 소개하는 것이 좋다.

### 2.1 Heterogeneity-aligned routing

이 부류는 **입력 공간의 이질성이 어디서 나타나는지**를 먼저 정하고, 그 축에 맞는 gate / selector / expert structure를 둔다.

#### (a) 초기 MoE: local subtask partition
- **Adaptive Mixtures of Local Experts**
- **Hierarchical Mixtures of Experts and the EM Algorithm**

이 계열의 핵심은, expert가 전체 입력 공간의 서로 다른 subset / subtask를 나눠 맡는다는 것이다. 즉 MoE의 출발점 자체가 “하나의 공유 함수로 모든 입력을 동일하게 처리하지 않아도 된다”는 local partitioning 관점이다.

#### (b) Task heterogeneity
- **MMoE**
- **PLE**

이 계열은 이질성의 축을 **task 관계**로 둔다.

- MMoE: task마다 다른 gate를 둬 expert mixture를 다르게 본다.
- PLE: task conflict / negative transfer를 shared-specific expert decomposition으로 푼다.

즉 “무엇을 router input으로 넣을까”를 샘플 내용만의 문제가 아니라, **task-aware control** 문제로 확장한다.

#### (c) Domain / scenario heterogeneity
- **AdaSparse**
- **PEPNet**

이 계열은 이질성의 축을 **domain, scenario, personalized prior**로 둔다.

- AdaSparse: domain-aware weighting으로 각 domain별 sparse structure를 학습한다.
- PEPNet: personalized prior information을 input으로 받아 embedding/parameter usage를 조절한다.

여기서 중요한 것은, hidden 안에 알아서 모든 걸 담기게 두지 않고 **domain-aware / personalized prior**를 gate 쪽에 직접 준다는 점이다.

#### (d) Modality / time heterogeneity in recommendation
- **HM4SR**
- **M3SRec**
- **FAME**

이 계열은 sequential recommendation과 더 가깝다.

- HM4SR는 multimodal SR에서 **modality relevance**와 **time-aware interest evolution**을 expert 구조에 반영한다.
- M3SRec는 multimodal representation learning을 expert mixture로 조직한다.
- FAME은 item/user preference의 **facet structure**를 중심으로 attention head와 MoE를 조직한다.

즉 recommendation에서도 이미 MoE는 “그냥 expert를 넣는 것”보다 **무슨 축으로 specialization할 것인가**의 문제로 가고 있다.

### 2.2 Router-mechanics-oriented MoE

이 부류는 router input보다 **sparse routing mechanics**가 중심이다.

- **Switch Transformer**: sparse routing의 scale / stability / efficiency
- **DSelect-k**: top-k gate의 비매끄러움을 개선하는 differentiable sparse gate
- **Expert Choice Routing**: token-to-expert assignment / load balance
- **V-MoE**: token/patch importance와 adaptive compute

이 논문들은 중요하지만, RouteRec의 main motivation으로 가져가면 안 된다. 이들은 주로

- sparse gate를 어떻게 더 안정적으로 학습할까?
- token-expert matching을 어떻게 더 잘할까?
- load balance / efficiency를 어떻게 확보할까?

를 묻는다. RouteRec이 필요한 것은 이쪽보다, **specialization axis와 routing signal의 alignment**를 다루는 첫 번째 계열이다.

### introduction에서의 요약 문장 추천

> Existing MoE work suggests that routing is most meaningful when it is aligned with the source of input heterogeneity—task relations in multi-task learning, domains in multi-domain recommendation, personalized priors in user-adaptive systems, or modality/time structure in multimodal recommendation. This motivates asking the same question in sessionized sequential recommendation: along which recurring behavioral axes should expert allocation specialize?

---

## 3. SeqRec에서는 어떤 축이 router-aware하게 중요해 보이나?

여기서 가장 중요한 원칙은 다음이다.

> **dataset-level 축은 motivation용 descriptor이지, router input 그 자체가 아니다.**

즉 introduction에서는

- raw logs를 보면 seqrec input이 어떤 축에서 반복적으로 달라지는지
- 그 축이 router design과 어떤 식으로 이어질 수 있는지

만 말하고,
actual router input은 method에서 sample-level cue로 소개한다.

### 3.1 추천하는 최종 축

`dataset_router_motivation.md` 기준으로, headline 축은 아래 네 개 + 보조 하나가 가장 좋다.

1. **Tempo / regime diversity**
2. **Transition ambiguity**
3. **Memory regime**
4. **Exposure regime**
5. **Context availability** (보조축)

### 3.2 왜 이 축이 좋은가

이 축들은 다음 조건을 만족한다.

- raw logs에서 직접 계산 가능하다.
- dataset-specific artifact보다 여러 데이터셋에서 반복적으로 보이는 패턴이다.
- method의 cue family로 자연스럽게 연결된다.
- 나중에 appendix에서 보조 분석을 붙여도 “결과 맞춤형 composite score”처럼 보일 위험이 상대적으로 작다.

### 3.3 각 축의 논문 내 역할

#### A. Tempo / regime diversity
의미:
- 세션의 pace, 길이, 형상이 얼마나 흔들리는가
- input이 항상 같은 temporal regime에 있지 않다는 뜻

motivation 역할:
- 같은 FFN/shared path만으로 처리하기보다, tempo-sensitive routing 여지가 있음을 설명

method 연결:
- **Tempo cue family**
- macro/mid/micro 전체에서 시간 간격, session form, pace trend를 보는 이유

#### B. Transition ambiguity
의미:
- 비슷한 local state에서 다음 행동이 얼마나 여러 방향으로 갈라지는가
- local multimodality / branchiness의 정도

motivation 역할:
- hidden-only router가 왜 불충분할 수 있는지 가장 직관적으로 설명하는 축
- same local content does not imply the same routing need

method 연결:
- **Focus cue family**
- category concentration / switching / intent concentration vs drift
- micro/mid routing necessity

#### C. Memory regime
의미:
- repetition, recurrence, carryover가 얼마나 강하고 또 얼마나 가변적인가

motivation 역할:
- 같은 sequence length라도 repeat-heavy regime과 exploratory regime이 다를 수 있음을 설명
- persistent preference vs local novelty를 다른 routing axis로 볼 수 있게 함

method 연결:
- **Memory cue family**
- repeat intensity, recurrence, carryover-like cue
- 특히 macro / mid explanation에 유리함

#### D. Exposure regime
의미:
- behavior가 head-heavy exposure 쪽에 더 묶이는지, 더 preference-driven한지

motivation 역할:
- popularity-driven browsing과 preference-driven consumption을 구별할 여지를 설명
- main axis로 너무 세게 밀 필요는 없지만, 독립 cue family를 정당화하는 데 유용

method 연결:
- **Exposure cue family**
- popularity level / spread / drift

#### E. Context availability (보조축)
의미:
- 위 heterogeneity가 있어도 router가 활용할 repeated-session context가 충분한가

motivation 역할:
- 왜 어떤 데이터셋에서는 macro routing headroom이 크고, 어떤 데이터셋에서는 작은지를 설명
- heterogeneity 자체보다는 **routing support / headroom condition**

method 연결:
- 별도 cue family라기보다 **macro/mid routing support**
- session-based setting과 cross-session history access를 강조할 근거

---

## 4. dataset table / score는 motivation에서 어디까지 쓸까?

### 절대 피해야 할 것

- intro에서 `SimpleScore`를 전면에 내세우기
- “우리가 만든 score와 결과가 잘 맞는다 → 그래서 motivation이 맞다”로 쓰기
- dataset-level score를 router input처럼 보이게 만들기

이 셋은 모두 사후 끼워맞춤처럼 보일 위험이 크다.

### 추천 사용 방식

#### Introduction / Motivation
- score는 안 보여준다.
- raw logs에서 recurring axis가 보인다는 정성적 요약만 쓴다.
- dataset name은 많아도 1~2개만 가볍게 예시로 언급한다.

#### Main experiments (Q1 discussion)
- “RouteRec gains are clearest where branching is strong and repeated-session context is sufficiently available” 정도의 한두 문장만 쓴다.
- 표나 correlation은 본문에 넣지 않는다.

#### Appendix / dataset analysis section
- dataset table
- directional comparison with gain / win rate
- simple score
- auxiliary candidate metrics 배제 이유

를 넣는다.

### 한 줄 원칙

> **본문은 axis 중심, appendix는 score 중심.**

---

## 5. 가장 자연스러운 storyline

사용자께서 적어둔 1~10 흐름을 논문용으로 다듬으면 아래 버전이 가장 자연스럽다.

### Step 1. SeqRec에 conditional computation을 넣어보고 싶었다
- sequential recommendation에서도 user/session behavior가 단일한 regime으로 보이지 않는다.
- 따라서 MoE는 자연스러운 후보처럼 보인다.

### Step 2. 그런데 MoE literature를 보면 핵심은 “expert를 쓰느냐”보다 “무엇으로 나누느냐”였다
- task-aware: MMoE, PLE
- domain-aware: AdaSparse
- personalized-prior-aware: PEPNet
- modality/time-aware: HM4SR

즉 MoE는 좋은 representation을 넘어서, **routing signal이 무엇이어야 하는가**가 중요하다.

### Step 3. 그래서 seqrec raw logs에서 반복적으로 나타나는 behavioral axis를 생각했다
- 너무 model-specific한 handcrafted feature list에서 출발하지 않고,
- raw logs 수준에서 recurring axis를 먼저 본다.

### Step 4. sessionized sequential logs에서는 네 가지 behavioral axis + 하나의 support axis가 자연스럽다
- tempo/regime diversity
- transition ambiguity
- memory regime
- exposure regime
- context availability

여기서는 숫자를 많이 보여줄 필요 없이, timestamp, session form, item category, repeat structure, popularity statistics 정도만 언급하면 충분하다.

### Step 5. 그래서 이 축들을 sample-level cue로 operationalize해서 router input으로 넣었다
- dataset-level axis → sample-level cue family
- Tempo / Focus / Memory / Exposure
- macro / mid / micro scope

### Step 6. 실험 결과는 이 framing과 대체로 일치했다
- dynamic and context-rich datasets: gain 큼
- context-scarce or strong shared-path-suitable datasets: gain 작음
- 다만 이건 main motivation이 아니라 **supporting empirical pattern**으로만 쓴다.

### 이 흐름의 장점
- 결과를 보고 점수를 만든 느낌이 약해진다.
- feature choice가 ad hoc처럼 보이지 않는다.
- MoE literature와 논리적으로 이어진다.

---

## 6. Introduction를 문단별로 어떻게 짤까

### Paragraph 1 — behavioral regimes are not uniform
핵심:
- sequential recommendation은 next-item prediction으로 보이지만, 실제 입력은 한 종류의 signal이 아니다.
- sessionized interaction logs에서는 pace, focus, repetition, exposure가 반복적으로 달라진다.

주의:
- 아직 RouteRec 얘기 너무 빨리 하지 않는다.
- “다 다르다”가 아니라 **recurring axes**라는 표현을 써서 구조적으로 보이게 한다.

### Paragraph 2 — prior seqrec mostly improves representation under shared computation
핵심:
- session-aware, time-aware, feature-aware methods는 주로 representation enrichment 쪽에 있었다.
- 즉 what the model encodes는 좋아졌지만, which computation path should be used는 명시적으로 다루지 않았다.

여기 들어갈 논문:
- SASRec / TiSASRec / GRU4Rec / DuoRec / BSARec / SIGMA
- feature-aware line으로 DIF-SR, FDSA

### Paragraph 3 — MoE literature asks a different question: what defines expert allocation?
핵심:
- MoE는 shared vs sparse capacity의 문제가 아니라, **what defines specialization**의 문제다.
- MMoE, PLE, AdaSparse, PEPNet, HM4SR는 각각 task/domain/personalized prior/modality-time structure를 gate 쪽에 반영했다.

이 문단의 역할:
- RouteRec이 뜬금없는 feature engineering paper가 아니라, MoE routing design question을 seqrec로 가져온 논문이라는 인상 주기

### Paragraph 4 — in sessionized seqrec, raw logs suggest recurring behavioral axes
핵심:
- sessionized logs를 보면 tempo/regime diversity, transition ambiguity, memory regime, exposure regime, and context availability가 반복적으로 나타난다.
- 이들은 router input 그 자체는 아니지만, router가 어떤 축을 보아야 하는지 알려주는 raw-log descriptors다.

주의:
- 점수, correlation, dataset table은 여기 넣지 않는다.
- “we inspect raw logs” 정도만 써도 충분하다.

### Paragraph 5 — RouteRec proposal
핵심:
- 위 축들을 sample-level lightweight cues로 operationalize한다.
- cues are control signals, not richer predictive side information.
- macro / mid / micro scope에서 cue를 추출하고, hierarchical sparse router로 expert allocation을 정한다.

### Paragraph 6 — result-oriented preview
핵심:
- stronger gains in dynamic / context-rich datasets
- smaller margins where repeated-session context is scarce or strong shared-path encoders are already sufficient

주의:
- 이 문단은 “proof”가 아니라 “what the experiments later show” 정도로 예고만 한다.

---

## 7. Method에서는 이 축을 어떻게 쓰면 좋은가

### 7.1 4.2 Behavioral Cue Construction 첫 문단 역할

여기서 딱 말하고 싶은 건 다음이다.

> We do not start from an arbitrary feature bank. We start from recurring sources of routing demand visible in raw session logs, and then instantiate them as lightweight sample-level cues.

이 한 문장이 있으면 좋다.

### 7.2 axis → cue family 대응 표는 method에서 매우 유용

간단한 축약표 예시:

| Raw-log axis | Operational cue family | Example signal |
| --- | --- | --- |
| Tempo / regime diversity | Tempo | gap stats, pace trend, valid-prefix ratio |
| Transition ambiguity | Focus | switch rate, category concentration, suffix entropy |
| Memory regime | Memory | repeat rate, recurrence, carryover |
| Exposure regime | Exposure | popularity level, drift, concentration |
| Context availability | routing support | history-window validity, repeated-session support |

이 표는 현재 `dataset_router_motivation.md`의 axis 정리와 지금 method의 네 family를 부드럽게 연결해준다.

### 7.3 sessionization을 어떻게 정당화할까

이 부분은 억지로 “우리 method 때문에 sessionized setting을 만들었다”처럼 보이면 안 된다.

추천 framing:
- 우리는 **sessionized sequential recommendation with accessible cross-session history**를 명시적 문제 설정으로 둔다.
- 이 setting에서 within-session local dynamics와 cross-session reusable context가 함께 observable해진다.
- 따라서 routing demand를 설명하는 축도 자연스럽게 session form, local branching, repeated-session context로 나타난다.

즉 sessionization은 gimmick이 아니라 **문제를 드러내는 관측 틀**로 써야 한다.

---

## 8. Experiment에서는 어떻게 가져갈까

### 8.1 본문 main result 해석

가장 좋은 문장은 이 정도다.

> RouteRec is most effective where local transition ambiguity is pronounced and repeated-session context is sufficiently available, while its margin narrows when routing headroom is limited by context scarcity or when strong shared-path encoders already match the dominant regime.

이 문장은
- KuaiRec / LastFM / Foursquare의 강한 이득
- Beauty / Retail Rocket / ML-1M의 더 제한적인 이득

을 모두 무리 없이 설명한다.

### 8.2 dataset별 해석

#### KuaiRec
- high branching
- strong context
- strong memory variability
- full-win dataset

해석:
- local multimodality와 reusable context가 모두 강해 routing demand/headroom이 큼

#### LastFM
- very strong context
- memory regime 강함
- full-win dataset

해석:
- repeated-session context와 memory-related routing이 특히 잘 맞는 사례

#### Foursquare
- high volatility and context
- gains exist but MRR는 덜 극적

해석:
- routing demand는 크지만, earliest-hit precision보다 broader recall 쪽에 더 기여

#### Beauty
- branching은 있어도 context scarcity 큼

해석:
- heterogeneity 자체보다 repeated-session support 부족이 headroom을 제한

#### ML-1M
- some context exists, but long stable sequences and strong shared-path suitability dominate

해석:
- routing demand가 전혀 없다기보다, additional headroom이 작음

### 8.3 Appendix에서만 보여줄 것

- Dataset Table
- Directional comparison with gain / win rate
- simple composite score
- dropped candidates (`duration_irregularity`, `cross_session_drift` 등)

이렇게 하면 본문은 깔끔하고, appendix는 충분히 설명적이다.

---

## 9. 실제로 쓸만한 introduction 초안 문장

### 버전 A: 가장 추천

> Rather than treating sequential inputs as heterogeneous in a single opaque sense, we find that sessionized interaction logs vary recurrently along a small set of behavioral axes: temporal regime diversity, ambiguity in local next-step transitions, repetition and carryover structure, and exposure regime. These axes are visible from ordinary logs even before any model-specific transformation.

> This changes the central MoE question in sequential recommendation. The issue is not only whether to use conditional computation, but which behavioral axis should define expert allocation. Existing MoE work often aligns routing with the source of heterogeneity—across tasks, domains, personalized priors, or modalities. We follow the same principle in sessionized recommendation, and design router cues as lightweight controls aligned with recurring behavioral axes rather than as richer predictive side information.

### 버전 B: 조금 더 보수적

> We do not begin from an arbitrary bank of routing features. Instead, we inspect raw session logs and identify a small set of recurring sources of routing demand: variability in session dynamics, branching in local transitions, repetition structure, and the availability of reusable cross-session context. RouteRec operationalizes these sources as lightweight behavioral cues for expert allocation.

---

## 10. 최종 추천: 무엇을 본문에 넣고 무엇을 appendix로 뺄까

### 본문에 넣을 것
- `behavioral routing demand`라는 용어
- recurring behavioral axes 4개 + context availability 1개
- MoE literature에서 task/domain/prior/modality-time aligned routing examples
- cue families are operationalizations of these axes
- stronger gains in dynamic/context-rich datasets라는 qualitative statement

### 본문에서 빼거나 약하게 둘 것
- simple composite score
- rank correlation 수치
- “우리 score가 gain과 맞는다”는 식의 직접적인 문장

### appendix에 넣을 것
- axis table
- dataset table
- directional correlation
- metric selection rationale

---

## 11. refs.bib에 추가할 만한 논문 목록

아래는 현재 refs.bib에 없거나, motivation 강화에 직접적으로 도움이 되는 논문들 위주로 추린 목록이다.

### A. MoE origin / expert partition
- Jacobs et al., 1991 — Adaptive Mixtures of Local Experts
- Jordan and Jacobs, 1994 — Hierarchical Mixtures of Experts and the EM Algorithm

### B. Heterogeneity-aware routing in recommendation / MTL
- Ma et al., 2018 — MMoE
- Tang et al., 2020 — PLE
- Yang et al., 2022 — AdaSparse
- Chang et al., 2023 — PEPNet

### C. Router mechanics / contrast papers
- Fedus et al., 2022 — Switch Transformers
- Hazimeh et al., 2021 — DSelect-k
- Zhou et al., 2022 — Expert Choice Routing
- Riquelme et al., 2021 — V-MoE

### D. SeqRec-adjacent MoE motivation papers
- Bian et al., 2023 — M3SRec
- Zhang et al., 2025 — HM4SR
- Liu et al., 2025 — FAME (already likely present)

---

## 12. BibTeX appendix

```bibtex
@article{jacobs1991adaptive,
  title={Adaptive Mixtures of Local Experts},
  author={Jacobs, Robert A. and Jordan, Michael I. and Nowlan, Steven J. and Hinton, Geoffrey E.},
  journal={Neural Computation},
  volume={3},
  number={1},
  pages={79--87},
  year={1991},
  doi={10.1162/neco.1991.3.1.79}
}

@article{jordan1994hierarchical,
  title={Hierarchical Mixtures of Experts and the EM Algorithm},
  author={Jordan, Michael I. and Jacobs, Robert A.},
  journal={Neural Computation},
  volume={6},
  number={2},
  pages={181--214},
  year={1994},
  doi={10.1162/neco.1994.6.2.181}
}

@inproceedings{ma2018mmoe,
  title={Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts},
  author={Ma, Jiaqi and Zhao, Zhe and Yi, Xinyang and Chen, Jilin and Hong, Lichan and Chi, Ed H.},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1930--1939},
  year={2018},
  doi={10.1145/3219819.3220007}
}

@inproceedings{tang2020ple,
  title={Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations},
  author={Tang, Hongyan and Liu, Junning and Zhao, Ming and Gong, Xudong and Jin, Zhaohui and Lei, Yu and Zhang, Min and Li, Xuan},
  booktitle={Proceedings of the 14th ACM Conference on Recommender Systems},
  pages={269--278},
  year={2020},
  doi={10.1145/3383313.3412236}
}

@inproceedings{yang2022adasparse,
  title={AdaSparse: Learning Adaptively Sparse Structures for Multi-Domain Click-Through Rate Prediction},
  author={Yang, Xuanhua and Peng, Xiaoyu and Wei, Penghui and Liu, Shaoguo and Wang, Liang and Zheng, Bo},
  booktitle={Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  pages={4635--4639},
  year={2022},
  doi={10.1145/3511808.3557541}
}

@inproceedings{chang2023pepnet,
  title={PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information},
  author={Chang, Jianxin and Zhang, Chenbin and Hui, Yiqun and Leng, Dewei and Niu, Yanan and Song, Yang and Gai, Kun},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={379--389},
  year={2023},
  doi={10.1145/3580305.3599884}
}

@article{fedus2022switch,
  title={Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={Journal of Machine Learning Research},
  volume={23},
  number={120},
  pages={1--39},
  year={2022},
  url={https://www.jmlr.org/papers/v23/21-0998.html}
}

@inproceedings{hazimeh2021dselectk,
  title={DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning},
  author={Hazimeh, Hussein and Zhao, Zhe and Chowdhery, Aakanksha and Sathiamoorthy, Maheswaran and Chen, Yihua and Mazumder, Rahul and Hong, Lichan and Chi, Ed H.},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={29335--29347},
  year={2021}
}

@inproceedings{zhou2022expertchoice,
  title={Mixture-of-Experts with Expert Choice Routing},
  author={Zhou, Yanqi and Lei, Tao and Liu, Hanxiao and Du, Nan and Huang, Yanping and Zhao, Vincent Y. and Dai, Andrew M. and Chen, Zhifeng and Le, Quoc V. and Laudon, James},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={7103--7114},
  year={2022}
}

@inproceedings{riquelme2021vmoe,
  title={Scaling Vision with Sparse Mixture of Experts},
  author={Riquelme, Carlos and Puigcerver, Joan and Mustafa, Basil and Neumann, Maxim and Jenatton, Rodolphe and Pinto, Andre Susano and Keysers, Daniel and Houlsby, Neil},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={8583--8595},
  year={2021}
}

@inproceedings{bian2023m3srec,
  title={Multi-modal Mixture of Experts Representation Learning for Sequential Recommendation},
  author={Bian, Shuqing and Pan, Xingyu and Zhao, Wayne Xin and Wang, Jinpeng and Wang, Chao and Wen, Ji-Rong},
  booktitle={Proceedings of the 32nd ACM International Conference on Information and Knowledge Management},
  pages={110--119},
  year={2023},
  doi={10.1145/3583780.3614978}
}

@inproceedings{zhang2025hm4sr,
  title={Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation},
  author={Zhang, Shengzhe and Chen, Liyi and Shen, Dazhong and Wang, Chao and Xiong, Hui},
  booktitle={Proceedings of the ACM Web Conference 2025},
  pages={3672--3682},
  year={2025},
  doi={10.1145/3696410.3714676}
}
```

> 참고: `FAME`, `MMoE`, `PLE`, `M3SRec`, `HM4SR` 중 일부는 현재 refs에 이미 있을 수 있으므로 중복 여부만 확인해서 merge하면 된다.

---

## 13. 최종 한 줄 정리

가장 좋은 서사는 아래다.

> **MoE를 seqrec에 쓰겠다고 했을 때 핵심은 “왜 expert를 쓰느냐”보다 “무슨 behavioral axis를 따라 expert allocation을 정할 것이냐”이다. Raw session logs를 보면 tempo/regime diversity, transition ambiguity, memory regime, exposure regime, and context availability가 반복적으로 나타난다. RouteRec은 이 축들을 sample-level lightweight cues로 operationalize하여 router input으로 사용한다.**

이렇게 쓰면,

- 결과를 보고 억지로 score를 만든 느낌이 줄고,
- feature choice가 ad hoc처럼 보이지 않으며,
- MoE literature와도 훨씬 자연스럽게 연결된다.
