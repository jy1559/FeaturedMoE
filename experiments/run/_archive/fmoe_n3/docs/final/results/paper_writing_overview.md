# FMoE Paper Writing Overview (Co-author Memo)

이 문서는 "논문 제출용 완성본"이 아니라, 우리가 writing할 때 바로 가져다 쓰는 **스토리 설계도 + 문장 프레임 + 실험 인사이트 지도**다.

- 목적: 정보를 많이 전달하는 문서가 아니라, **어떤 논리 순서로 독자를 설득할지**를 빠르게 고정하는 문서
- 원칙: verification 수치 우선, verification 없는 곳만 wide 보조
- 톤: 설명자 톤보다 공동저자 메모 톤

---

## 1) Narrative Spine (한 페이지)

### Claim Stack (강약 포함)

| 레벨 | Claim | 강도 | 지금 써도 되는 문장 강도 |
| --- | --- | --- | --- |
| Main | Feature-guided routing converts MoE gating from hidden-only black-box behavior into more inspectable and portable routing priors. | Strong | 강하게 써도 됨 |
| Support A | Compact/common raw-signal templates are sufficient; large handcrafted feature banks are not mandatory. | Medium-Strong | 강하게 쓰되 "complete replacement" 같은 표현은 피함 |
| Support B | Gains are better explained by stage semantics/composition than by simply adding routed blocks or generic depth. | Strong | 강하게 써도 됨 |
| Validation | Evaluation-time perturbation (all-zero/all-shuffle) degrades quality, indicating that aligned cues are actively used. | Strong | 강하게 써도 됨 |
| Downplay | Train-time corruption effects are mixed and should be framed as secondary/diagnostic evidence. | Guarded | 보수적으로 써야 함 |

### One-sentence Thesis (작업용)

- English: `FMoE is not about adding more features to representations; it is about using compact, behavior-derived cues to control where MoE computation flows.`
- 한국어: `FMoE의 핵심은 표현에 feature를 더 많이 넣는 것이 아니라, compact한 행동 단서를 이용해 MoE 계산 경로 자체를 제어하는 데 있다.`

**실제로 쓸 때 한 줄 팁:** 논문 전체에서 "feature enrichment"보다 "computation path control"을 반복 키워드로 잡으면 메시지가 훨씬 선명해진다.

**강도 메모:**
- 이건 세게 써도 됨: `stage semantics/composition matter beyond parameter growth`
- 이건 보수적으로: `train corruption always hurts`

---

## 2) Reader Hook 설계

### 독자가 초반 3문단에서 받아야 할 흐름

1. **Problem Tension**
- 핵심 질문: "MoE가 좋아졌다고 하는데, expert assignment가 실제로 무슨 행동 의미를 갖는지 설명 가능한가?"
- 독자 훅: 성능보다 먼저 "해석 가능한 계산 경로" 문제를 던진다.

2. **Gap Framing**
- 기존 feature-aware SR은 representation enrichment 중심이고, routing path control은 약하다.
- 기존 MoE SR은 routing이 hidden-only black-box인 경우가 많다.
- 즉, 두 라인이 만나지 못한 공백을 제시한다.

3. **Our Move + Why Interesting**
- raw signal(순서/시간/카테고리)에서 compact cue를 만들고,
- macro/mid/micro stage routing에 연결해 계산 경로를 의미적으로 분해한다.
- 여기서 "복잡 feature bank"가 아니라 "portable template"이라는 실용 포인트를 강조한다.

### Hook 문장 템플릿 (초안용)

- English: `If experts are supposed to specialize, we should be able to explain what they specialize for.`
- 한국어: `전문가(expert)가 특화된다면, 그 특화가 무엇을 위한 것인지 설명 가능해야 한다.`

- English: `Most feature-aware recommenders enrich representations, but leave the computation path largely unchanged.`
- 한국어: `대부분의 feature-aware 추천 모델은 표현은 풍부하게 만들지만, 계산 경로 자체는 크게 바꾸지 않는다.`

**실제로 쓸 때 한 줄 팁:** 첫 문단에서 성능 숫자를 바로 꺼내기보다, "왜 지금 이 문제를 다시 봐야 하는가"를 먼저 고정하면 intro 이탈률이 줄어든다.

**강도 메모:**
- 이건 세게: black-box routing interpretability gap
- 이건 보수: "기존 연구가 해석 불가능하다" 같은 절대화

---

## 3) Abstract Blueprint + v1

### Blueprint (문장 역할 태그)

| 문장 | 역할 | 넣어야 할 내용 |
| --- | --- | --- |
| S1 | Problem | hidden-only routing의 해석 공백 |
| S2 | Gap | feature-aware line이 path control을 직접 다루지 못함 |
| S3 | Idea | FMoE: compact cue-guided macro/mid/micro routing |
| S4 | Evidence A | compact/portable feature template 결과 |
| S5 | Evidence B | stage/composition 효과 + eval perturbation 하락 |
| S6 | So-what | inspectable + practical routing prior라는 메시지 |

### Abstract v1 (English)

Mixture-of-experts (MoE) recommenders can improve sequential prediction, but routing decisions are often learned as hidden-only black boxes, making expert assignment difficult to interpret and verify. Existing feature-aware recommenders usually enrich sequence representations, yet they rarely provide explicit control over the computation path itself. We propose FeaturedMoE (FMoE), which derives compact behavioral cues from common raw sequential signals and uses them as routing guidance across macro, mid, and micro stages. Our results suggest that strong routing behavior does not require a large handcrafted feature bank: compact templates and partial-feature settings remain competitive in verification runs. We further show that performance is better explained by stage semantics and composition than by simply adding routed blocks, and that evaluation-time cue perturbations (all-zero/all-shuffle) consistently reduce ranking quality. These findings position FMoE as a practical framework for making MoE routing in sequential recommendation more inspectable, portable, and behavior-aware.

### Abstract v1 (한글 번역)

MoE 기반 순차 추천은 성능을 개선할 수 있지만, 라우팅 결정이 hidden-only 블랙박스로 학습되는 경우가 많아 expert assignment의 의미를 해석하고 검증하기 어렵다. 기존 feature-aware 추천 모델은 주로 시퀀스 표현을 풍부하게 만드는 데 집중하며, 계산 경로 자체를 명시적으로 제어하는 데는 한계가 있다. 우리는 common raw sequential signal로부터 compact한 행동 단서를 만들고, 이를 macro/mid/micro 단계 라우팅에 직접 주입하는 FeaturedMoE(FMoE)를 제안한다. 검증 실험 결과, 강한 라우팅 성능은 대규모 handcrafted feature bank 없이도 유지될 수 있었고, compact template 및 partial-feature 설정에서도 경쟁력이 유지됐다. 또한 성능 차이는 단순한 routed block 추가보다 stage semantics와 composition에 더 크게 좌우되었으며, eval-time cue perturbation(all-zero/all-shuffle)은 일관된 성능 하락을 보였다. 이는 FMoE가 순차 추천 MoE 라우팅을 더 inspectable하고 portable하며 behavior-aware하게 만드는 실용적 프레임워크임을 시사한다.

**실제로 쓸 때 한 줄 팁:** abstract 마지막 문장은 "성능 우위"보다 "왜 이 프레임이 연구/실무에서 유용한지"를 닫아주는 문장으로 쓰는 게 인상에 남는다.

**강도 메모:**
- 이건 세게: eval-time all-zero/all-shuffle degradation
- 이건 보수: train-time corruption 일반화 결론

---

## 4) Introduction Blueprint + v1

### Introduction Blueprint (phase 명시 없이)

| 단락 | 목표 | 꼭 넣을 것 | 피할 것 |
| --- | --- | --- | --- |
| P1 | 문제 제기 | 이질적 행동 신호 vs shared computation | 바로 결과 자랑 |
| P2 | 관련축 정리 | backbone 개선, feature-aware representation 개선 | 인용 나열만 하는 문단 |
| P3 | gap 선명화 | routing path control의 공백 | 기존 연구 과도 폄하 |
| P4 | 핵심 아이디어 | raw signal -> compact cue -> stage routing | feature engineering 과잉 느낌 |
| P5 | 모델/검증 요약 | A1 주인공 + A2/A3 역할 분리 | phase naming |
| P6 | contribution 정리 | claim stack과 동일한 3~4개 기여 | 과강한 단정 |

### Introduction v1 (English)

Sequential recommendation is often treated as next-item prediction over interaction histories, but user behavior is rarely governed by a single homogeneous signal. Long-term preference, session-level intent, and short-range transitions frequently coexist, while many models still rely on largely shared computation. As a result, heterogeneous behavioral evidence is compressed into one latent stream, and it remains unclear when the model should route computation toward stable preference, current intent, or local transitions.

Prior work has improved sequential recommenders from two major directions. One line strengthens sequence backbones, while another line introduces auxiliary signals such as time and category information to enrich representations. Both directions are effective, but they mostly improve what is represented, not how computation is explicitly routed under different behavioral regimes.

MoE-based sequential models offer a natural promise of expert specialization, yet routing is still commonly learned as a hidden-only function of latent states. This can improve metrics, but it leaves expert assignment semantically opaque: we often cannot tell whether experts correspond to meaningful behavioral regimes or arbitrary latent partitions.

FMoE starts from a different design question: can common raw sequential signals provide explicit routing guidance? We derive compact behavioral cues from broadly available signals and organize them into complementary families, then connect them to macro, mid, and micro routing stages. The design goal is not to build a large handcrafted feature bank, but to provide a small, portable, and inspectable routing prior.

Our experiments are structured to test this design logic rather than only report aggregate gains. We evaluate whether compact cue templates remain effective under reduced feature settings, whether stage semantics and composition explain performance better than generic depth expansion, and whether aligned cues are actively used through evaluation-time perturbation tests. In the final architecture package, A1 serves as the main model, while A2 and A3 act as robustness and stress-control variants.

Overall, we frame FMoE as a practical approach to behavior-aware MoE routing in sequential recommendation: compact in feature assumptions, explicit in routing structure, and empirically testable through architecture-aware and alignment-aware analyses.

### Introduction v1 (한글 번역)

순차 추천은 보통 상호작용 이력 기반의 다음 아이템 예측 문제로 다뤄지지만, 실제 사용자 행동은 하나의 균질한 신호로 설명되지 않는다. 장기 선호, 세션 수준 의도, 단기 전이 성향이 동시에 존재하며, 많은 모델은 여전히 상당 부분 공유 계산(shared computation)에 의존한다. 그 결과 이질적 행동 증거가 하나의 잠재 표현으로 압축되고, 모델이 언제 안정적 선호에 의존하고 언제 현재 의도나 국소 전이에 의존해야 하는지 계산 경로 관점에서 불명확해진다.

기존 연구는 크게 두 방향에서 순차 추천을 개선해왔다. 하나는 시퀀스 백본을 강화하는 방향이고, 다른 하나는 시간·카테고리 같은 보조 신호를 활용해 표현을 풍부하게 만드는 방향이다. 두 방향 모두 효과적이지만, 주로 "무엇을 표현하는가"를 개선하지 "행동 레짐에 따라 계산을 어떻게 라우팅할 것인가"를 직접 제어하진 못한다.

MoE 기반 순차 추천은 expert specialization을 약속하는 매력적인 구조지만, 라우팅은 여전히 hidden-only 잠재 상태 함수로 학습되는 경우가 많다. 이런 방식은 지표를 개선할 수 있어도 expert assignment의 의미를 불투명하게 만든다. 즉, expert가 실제 행동 레짐을 반영하는지, 아니면 잠재 공간의 임의 분할인지 구분하기 어렵다.

FMoE는 다른 질문에서 출발한다. common raw sequential signal이 라우팅 자체를 명시적으로 안내할 수 있는가? 우리는 널리 이용 가능한 raw signal에서 compact한 행동 단서를 만들고, 이를 상보적 family로 구성한 뒤 macro/mid/micro 라우팅 단계에 연결한다. 목표는 큰 handcrafted feature bank를 구축하는 것이 아니라, 작고 portable하며 inspectable한 routing prior를 제공하는 것이다.

실험은 단순 평균 성능 보고가 아니라 이 설계 논리를 검증하도록 구성했다. 축약된 feature 설정에서도 compact template이 유지되는지, generic depth 확장보다 stage semantics와 composition이 성능을 더 잘 설명하는지, eval-time perturbation 테스트에서 정렬된 cue가 실제로 사용되는지를 확인한다. 최종 아키텍처 패키지에서는 A1을 메인 모델로 두고, A2/A3를 robustness 및 stress-control 변형으로 위치시킨다.

결과적으로 우리는 FMoE를 순차 추천에서 behavior-aware MoE routing을 위한 실용적 접근으로 제시한다. 즉, feature 가정은 compact하고, routing 구조는 명시적이며, architecture-aware/alignment-aware 분석으로 경험적으로 검증 가능하다는 점에 초점을 둔다.

**실제로 쓸 때 한 줄 팁:** Introduction 마지막 단락은 "우리가 뭘 했다"보다 "독자가 이걸 읽고 무엇을 믿게 되는가"로 닫으면 전달력이 올라간다.

**강도 메모:**
- 이건 세게: stage semantics/composition 중요성
- 이건 보수: "interpretability solved" 같은 완전 해결형 문장

---

## 5) Evidence-to-Claim Map

verification 우선으로 claim에 직접 붙일 수 있는 근거만 추렸다.

| Claim | 직접 근거 (추천 인용) | 해석 포인트 | 문장 강도 |
| --- | --- | --- | --- |
| Compact templates are enough | P10 verification(H1/H3): `FULL` valid 0.0811, test 0.1615 vs `TOP2_PER_GROUP` valid 0.0808, test 0.1619 | full feature 대비 compact setting이 test에서 유지/상회 | 중간 이상 |
| Portability has a boundary | P10 verification: `NO_CATEGORY_NO_TIMESTAMP` test 0.1614 (FULL 0.1615보다 근소 하락) | 신호를 크게 줄여도 기능은 유지되나 경계는 존재 | 보수 |
| Stage semantics matter beyond generic depth | P11 verification(H3): `MACRO_MID_MICRO` test 0.1616 vs `LAYER_ONLY_BASELINE` test 0.1593 | 단순 depth 증가만으로 동일 효과 설명 어려움 | 강하게 가능 |
| Composition/layout materially changes outcomes | P12 verification(H3): `ATTN_MICRO_BEFORE` test 0.1621 vs `ATTN_ONESHOT` test 0.1610 | stage 집합이 유사해도 배치/조합에 따라 성능 차이 | 강하게 가능 |
| Aligned cues are actively used at eval | P13 verification(H3): `FULL_DATA` test 0.1615 vs `EVAL_ALL_ZERO` 0.1597 vs `EVAL_ALL_SHUFFLE` 0.1552 | eval-time 교란에서 명확한 하락 | 강하게 가능 |
| Train corruption is mixed (downplay) | P13 verification(H3): `TRAIN_PERMUTE_FOCUS` test 0.1620, `TRAIN_PERMUTE_TEMPO` 0.1618 | train 교란은 핵심 주장보단 보조 진단으로 배치 | 보수 |

### 문장 레벨 운영 규칙

- 핵심 claim 문단에는 `verification mean`만 사용
- wide 수치는 `보조 문장`이나 `appendix-style commentary`에서만 보충
- P13은 `eval perturbation`을 main validation으로 쓰고, `train corruption`은 "mixed"로 짧게

**실제로 쓸 때 한 줄 팁:** 결과 섹션에서 "best score"를 앞세우기보다, "이 claim을 지지하는 최소 근거 2개"를 붙이는 방식이 설득력이 더 높다.

**강도 메모:**
- 이건 세게: eval perturbation degradation
- 이건 보수: 모든 corruption 일반화

---

## 6) Model/Result 정리 프레임 (A1/A2/A3)

### 역할 배치 (논문에서 보이는 캐릭터 설정)

| 모델 | 역할 | 논문 내 기능 | 쓰는 위치 |
| --- | --- | --- | --- |
| A1 | Protagonist (main) | 핵심 설계 아이디어를 가장 잘 반영하는 주인공 | Method, Main Results |
| A2 | Robustness Variant | A1 설계의 대체 조합에서 핵심 현상이 유지되는지 확인 | Ablation/Robustness |
| A3 | Stress/Control Variant | category/theme zero-fill 같은 stress 조건에서의 경계 확인 | Sanity/Limit Discussion |

### 결과 정리 템플릿 (작업용)

1. Main Table: A1 중심 핵심 축 결과(Compactness, Stage/Composition, Eval Perturbation)
2. Support Table: A2/A3 포함 robustness/control 비교
3. Diagnostic Figure: n_eff/entropy/top1 같은 route dynamics를 "주장 보조"로만 배치

### 쓸 때 추천 문장 프레임

- English: `A1 is treated as the architectural protagonist, while A2 and A3 are used to test robustness and boundary behavior under controlled deviations.`
- 한국어: `A1은 아키텍처 주인공으로 두고, A2/A3는 통제된 변형 하에서 강건성 및 경계 거동을 확인하는 대조군으로 사용한다.`

**실제로 쓸 때 한 줄 팁:** 모델을 "동등 후보 3개"로 제시하면 메시지가 흐려진다. A1을 주인공으로 못 박고 A2/A3를 질문형 대조군으로 두는 게 훨씬 읽기 쉽다.

**강도 메모:**
- 이건 세게: A1 중심 내러티브
- 이건 보수: A2/A3 우열 결론 단정

---

## Quick Quality Checklist (draft 점검용)

- [ ] Abstract 각 문장이 Claim Stack 항목에 1:1 매핑되는가?
- [ ] Introduction 앞 2단락만 읽어도 "왜 이 논문을 봐야 하는지"가 명확한가?
- [ ] 결과 인용이 직접증거 vs 보조증거로 분리되어 있는가?
- [ ] P13에서 강한 문장이 eval perturbation 근거와 직접 연결되는가?
- [ ] A1/A2/A3 역할이 중복 없이 분리되는가?

