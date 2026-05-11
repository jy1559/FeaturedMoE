# RouteRec Rebuttal 1개월 준비 계획

## 0. 목표 요약

이번 1개월은 **새 논문을 다시 만드는 기간**이 아니라, 현재 RouteRec 논문이 받을 수 있는 핵심 의심을 최소 비용으로 방어하는 기간으로 잡는 것이 좋다.

현재 논문은 이미 다음 요소를 갖고 있다.

- 6개 public dataset
- 9개 baseline
- routing-input ablation
- temporal scope / routing structure ablation
- cue-portability experiment
- routing-profile analysis
- dataset-level behavioral indicator appendix

하지만 reviewer가 찌를 가능성이 높은 부분은 아직 남아 있다.

1. **성능 이득이 extra capacity 때문 아닌가?**
2. **behavioral cue가 진짜 routing에 의미 있게 쓰였나?**
3. **hand-crafted feature engineering 아닌가?**
4. **hidden-state routing으로도 충분한 것 아닌가?**
5. **sessionization / cue construction / seen-target evaluation에서 leakage나 unfairness는 없나?**
6. **dataset별 gain이 들쭉날쭉한데, 이걸 어떻게 설명할 것인가?**

따라서 rebuttal 준비의 핵심은 다음 한 문장을 방어하는 것이다.

> RouteRec의 이득은 단순히 MoE capacity를 늘렸기 때문이 아니라, behaviorally aligned cue를 routing prior로 사용했기 때문에 나타난다. 또한 RouteRec은 모든 dataset에서 무조건 이기는 black-box model이 아니라, behavioral routing demand가 있는 setting에서 특히 유용한 conditional-computation framework이다.

---

## 1. 새 GitHub Repo 정리 계획

### 1.1 목표

기존 연구용 repo를 그대로 공개하면 너무 지저분해 보일 가능성이 높다.  
새 repo는 **paper artifact repo**로 정리하는 것이 좋다.

목표는 다음 세 가지다.

1. 논문 결과를 재현할 수 있다.
2. preprocessing / evaluation protocol이 명확하다.
3. RouteRec 구조를 처음 보는 사람도 코드 흐름을 이해할 수 있다.

모든 실험 흔적, 실패한 phase, wandb cache, 중간 산출물을 다 올릴 필요는 없다.  
**논문 결과 재현에 필요한 최소 단위**만 남기는 것이 좋다.

---

### 1.2 권장 Repository 구조

```text
RouteRec/
  README.md
  LICENSE
  requirements.txt 또는 environment.yml

  configs/
    datasets/
      kuairec.yaml
      foursquare.yaml
      lastfm.yaml
      retailrocket.yaml
      beauty.yaml
      ml1m.yaml
    models/
      routerec.yaml
      sasrec.yaml
      gru4rec.yaml
      ...
    experiments/
      main_table/
      ablations/
      diagnostics/

  scripts/
    preprocess/
      run_preprocess_all.sh
      preprocess_dataset.py
    train/
      run_routerec.sh
      run_baselines.sh
    eval/
      evaluate.sh
      aggregate_results.py
    reproduce/
      reproduce_main_table.sh
      reproduce_ablation.sh
      reproduce_diagnostics.sh

  src/
    data/
      dataset.py
      preprocessing.py
      sessionization.py
    models/
      routerec/
        model.py
        router.py
        experts.py
        cue_bank.py
      baselines/
    routing/
      cue_extraction.py
      hierarchical_router.py
      regularizers.py
    training/
      trainer.py
      losses.py
    evaluation/
      metrics.py
      evaluator.py
    utils/

  results/
    paper/
      main_table.csv
      full_metrics.csv
      ablation_routing_input.csv
      ablation_scope.csv
      ablation_router_structure.csv
      cue_portability.csv
      routing_profiles.csv
    README.md

  notebooks/
    analysis_routing_profiles.ipynb
    analysis_behavior_bins.ipynb

  docs/
    preprocessing.md
    hyperparameters.md
    artifact_notes.md
    leakage_checklist.md
```

---

### 1.3 README에 반드시 들어갈 내용

README는 길게 쓰기보다, 실행 흐름이 바로 보여야 한다.

```bash
# 1. Environment
conda env create -f environment.yml
conda activate routerec

# 2. Preprocess datasets
bash scripts/preprocess/run_preprocess_all.sh

# 3. Train RouteRec
bash scripts/train/run_routerec.sh configs/datasets/kuairec.yaml

# 4. Train baselines
bash scripts/train/run_baselines.sh configs/datasets/kuairec.yaml

# 5. Reproduce main table
bash scripts/reproduce/reproduce_main_table.sh

# 6. Reproduce ablations
bash scripts/reproduce/reproduce_ablation.sh
```

README 구성은 다음 정도면 충분하다.

```text
1. Overview
2. Installation
3. Dataset preparation
4. Preprocessing protocol
5. Training
6. Evaluation
7. Reproducing paper results
8. Expected output files
9. Citation
10. License
```

---

### 1.4 Config 정리 원칙

논문에 들어간 설정은 코드 안에 흩어두지 말고 config로 빼야 한다.

예시:

```yaml
model:
  name: routerec
  backbone: sasrec
  hidden_size: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.15

routing:
  scopes: [macro, mid, micro]
  cue_families: [tempo, focus, memory, exposure]
  num_groups: 4
  experts_per_group: 3
  topk_groups: 3
  topk_experts: 2
  use_hierarchical_router: true
  use_sparse_routing: true

regularization:
  use_route_consistency: true
  lambda_consistency: 0.0005
  use_z_loss: true
  lambda_z: 0.0001

cue:
  macro_history_sessions: 5
  micro_window_size: 5
  cue_dim_per_scope: 16
  missing_focus_policy: zero_fill
```

이렇게 해야 reviewer나 artifact evaluator가 봤을 때 “논문 설정이 어디 있는지” 바로 알 수 있다.

---

### 1.5 Preprocessing 문서화

`docs/preprocessing.md`는 특히 중요하다.  
RouteRec은 sessionization, cue construction, seen-target evaluation을 사용하기 때문에 leakage 의심을 받을 수 있다.

반드시 명시할 것:

- raw interaction 정렬 방식
- sessionization threshold
- iterative filtering rule
- train / validation / test split 방식
- seen-target evaluation 정의
- sampled subset을 쓴 dataset과 sampling seed
- item popularity가 train split에서만 계산된다는 점
- macro / mid / micro cue가 prediction target 이후 정보를 보지 않는다는 점
- category / timestamp / item ID 사용 방식
- valid/test에서 unseen item을 어떻게 처리했는지

---

### 1.6 공개하지 않아도 되는 것

다음은 새 artifact repo에 굳이 올리지 않는 것이 좋다.

- wandb run 전체
- tensorboard log 전체
- 모든 checkpoint
- 실패한 phase별 shell script
- 서버 경로가 박힌 config
- 개인 계정명 / 서버명 / 절대경로
- debugging notebook
- 중복 CSV
- 임시 분석 그림
- 모델 개발 중간 버전 전체

대신 논문에 필요한 결과 CSV와 재현 script만 정리한다.

---

## 2. Rebuttal 1개월 실험 우선순위

## P0. 반드시 해야 하는 방어 실험

---

### 2.1 Shuffled-cue / Random-cue Ablation

#### 목적

가장 중요한 방어 실험이다.

Reviewer는 다음처럼 의심할 수 있다.

> RouteRec이 좋은 이유가 behavioral cue 때문이 아니라, 단순히 MoE parameter가 늘었기 때문 아닌가?

이를 막기 위해 cue의 semantic alignment를 깨는 실험이 필요하다.

#### 비교군

- SASRec
- Hidden-router MoE
- RouteRec with true cues
- RouteRec with shuffled cues
- RouteRec with random cues
- 가능하면 RouteRec with family-shuffled cues

#### 기대 결과

- True cue RouteRec이 가장 좋아야 한다.
- Shuffled/random cue는 성능이 떨어져야 한다.
- Hidden-router MoE보다 true cue RouteRec이 더 안정적이어야 한다.

#### Rebuttal에서 쓸 수 있는 주장

> The improvement is not explained by additional MoE capacity alone. When the behavioral alignment of cues is broken through shuffling or random replacement, the gain is substantially weakened.

#### 추천 dataset

우선순위:

1. KuaiRec
2. LastFM
3. Foursquare

시간이 부족하면 KuaiRec + LastFM만 먼저 한다.

---

### 2.2 Capacity-matched Baseline

#### 목적

RouteRec의 이득이 parameter 수 증가 때문이라는 의심을 방어한다.

#### 비교군

- SASRec
- SASRec-wide  
  - RouteRec과 parameter 수가 비슷하도록 FFN width 증가
- Flat MoE with hidden router
- Flat MoE with random/cue-broken router
- RouteRec

#### 기대 결과

- SASRec-wide가 일부 좋아질 수는 있다.
- 하지만 RouteRec보다 낮거나, 최소한 RouteRec의 핵심 dataset에서 밀려야 한다.
- hidden/router-only MoE가 RouteRec만큼 안정적이지 않아야 한다.

#### Rebuttal에서 쓸 수 있는 주장

> Matching the parameter budget does not reproduce RouteRec's gain, suggesting that the benefit comes from behavior-guided conditional computation rather than model size alone.

---

### 2.3 Leakage / Protocol Audit Table

#### 목적

RouteRec은 feature/cue를 사용하므로 leakage 의심이 생길 수 있다.  
실험 하나보다 더 중요한 방어 자료가 될 수 있다.

#### 작성할 표

| Component | Source | Train-only? | Prefix-only? | Leakage risk |
|---|---|---:|---:|---|
| Item popularity | training interactions | Yes | N/A | No |
| Category cue | item metadata | Yes / fixed metadata | Yes | No |
| Tempo cue | timestamps | N/A | Yes | No |
| Memory cue | item IDs | train/past/prefix | Yes | No |
| Macro cue | past sessions | Yes | before current target | No |
| Mid cue | current session prefix | N/A | Yes | No |
| Micro cue | recent prefix | N/A | Yes | No |
| Validation/test target | held-out split | No train update | N/A | No |

#### Rebuttal에서 쓸 수 있는 주장

> All routing cues are deterministic summaries computed from training statistics and the observed prefix only. No cue uses the held-out target item or future interactions within the current prediction step.

---

### 2.4 Seed Variance / Robustness

#### 목적

Single-run best result라는 의심을 줄인다.

#### 현실적인 범위

전체 6 dataset × 10 model × 3 seeds는 너무 무겁다.  
선택적으로만 한다.

추천 dataset:

- KuaiRec
- LastFM
- ML-1M 또는 Foursquare

추천 model:

- SASRec
- best non-RouteRec baseline
- Hidden-router MoE
- RouteRec

#### 기대 결과

- RouteRec의 평균 성능이 유지된다.
- 표준편차가 너무 크지 않다.
- ML-1M처럼 RouteRec이 약한 case도 정직하게 보여줄 수 있다.

#### Rebuttal에서 쓸 수 있는 주장

> The main pattern remains stable across seeds on representative datasets.

---

## P1. Diagnostic 분석 실험

이 실험들은 rebuttal에도 좋지만, 특히 다음 제출에서 RouteRec을 **diagnostic routing framework**로 강화하는 데 중요하다.

---

### 2.5 Cue Score ↔ Routing Weight Correlation

#### 목적

RouteRec router가 실제로 behavioral cue에 반응하는지 확인한다.

#### 방법

각 test sample 또는 session에 대해 다음 상관을 계산한다.

- Tempo cue score ↔ Tempo family routing weight
- Focus cue score ↔ Focus family routing weight
- Memory cue score ↔ Memory family routing weight
- Exposure cue score ↔ Exposure family routing weight

scope별로 나누면 더 좋다.

- macro
- mid
- micro
- all-stage average

#### 기대 결과

완벽하게 모든 family에서 높을 필요는 없다.  
오히려 dataset마다 다른 profile이 나오는 것이 diagnostic lens에는 더 좋다.

예상:

- LastFM: Memory correlation이 강할 가능성
- Foursquare: Tempo / Focus가 강할 가능성
- KuaiRec: Tempo / Memory / Focus가 섞여 나올 가능성
- ML-1M: 전반적으로 약할 수 있음

#### Rebuttal에서 쓸 수 있는 주장

> The learned router does not assign experts arbitrarily; family-level routing weights are directionally aligned with the corresponding behavioral cue scores.

---

### 2.6 Behavior Bin별 Route Profile + Gain

#### 목적

단순히 routing profile이 다르다는 것을 넘어서, 성능 gain과 연결한다.

#### 방법

각 behavioral dimension에 대해 bin을 만든다.

- repeat low / mid / high
- tempo slow / mid / fast
- focus broad / mid / narrow
- popularity tail / mid / head
- session length short / mid / long

각 bin에서 다음을 계산한다.

1. 평균 cue score
2. 평균 family routing weight
3. RouteRec 성능
4. baseline 성능
5. RouteRec gain

#### 기대 결과

예:

| Bin | Memory cue | Memory route weight | RouteRec gain |
|---|---:|---:|---:|
| Low repeat | low | low | small |
| Mid repeat | mid | mid | medium |
| High repeat | high | high | large |

#### 핵심 주장

> Behavior cue → route activation → performance gain

이 chain을 보여주는 것이 중요하다.

---

### 2.7 Expert Family Masking

#### 목적

Routing profile이 단순 correlation이 아니라 기능적으로 의미 있는지 확인한다.

#### 방법

학습된 RouteRec을 유지하고, inference 때 특정 expert family를 막는다.

- Mask Tempo experts
- Mask Focus experts
- Mask Memory experts
- Mask Exposure experts

그 후 behavior bin별 성능 하락을 측정한다.

#### 기대 결과

| Masking | 가장 크게 떨어져야 하는 bin |
|---|---|
| Memory masking | repeat-heavy sessions |
| Tempo masking | fast / irregular sessions |
| Focus masking | narrow-focus 또는 high-switch sessions |
| Exposure masking | head / popularity-sensitive sessions |

#### Rebuttal 또는 다음 제출에서 쓸 수 있는 주장

> The selected expert families are not only correlated with behavioral cues; they are functionally useful for the corresponding behavioral regimes.

#### 현실적인 실행 범위

먼저 KuaiRec 하나에서 한다.  
잘 나오면 LastFM, Foursquare로 확장한다.

---

## P2. 다음 제출용 확장 실험

아래는 중요하지만 rebuttal 1개월 안에 무리해서 다 할 필요는 없다.

---

### 2.8 Counterfactual Cue Intervention

#### 아이디어

같은 hidden sequence를 유지하고, cue만 바꿔서 prediction이 어떻게 변하는지 본다.

#### Intervention 예시

- original cue
- repeat-heavy prototype cue
- fast-tempo prototype cue
- narrow-focus prototype cue
- popularity-heavy prototype cue

#### 볼 지표

- repeated item score 변화
- same-category item score 변화
- popular item score 변화
- recent-transition candidate score 변화
- prediction entropy 변화

#### 기대 주장

> Behavioral cues act as interpretable control signals over scoring behavior.

---

### 2.9 Router Transfer / Low-resource Adaptation

#### 아이디어

RouteRec의 router는 item identity가 아니라 normalized behavioral cue를 보므로, dataset 간 transfer 가능성이 있다.

#### 실험 설계

1. Dataset A에서 RouteRec 학습
2. router만 저장
3. Dataset B에서 backbone/expert는 새로 학습
4. router는 다음 방식으로 비교
   - scratch
   - transferred initialization
   - frozen transferred router
   - random initialization

#### 추천 pair

- KuaiRec → Foursquare
- Foursquare → Retail Rocket
- LastFM → ML-1M
- Beauty → Retail Rocket

#### full-data보다 low-resource가 중요

- train 10%
- train 30%
- train 50%
- train 100%

#### 기대 주장

> Behavioral routing priors may transfer across domains, especially when target-domain data is limited.

#### 주의

Frozen router는 잘 안 될 가능성이 높다.  
우선은 **transferred initialization + low-resource convergence**를 보는 것이 안전하다.

---

### 2.10 SRPFN / PFN 연결

이건 rebuttal용이 아니라 다음 연구 주제에 가깝다.

가능한 방향:

- RouteRec cue로 SRPFN support selection을 조절
- behavior-aware support selection
- behavior diagnostic token을 PFN 입력에 추가
- behavior-conditioned update-free inference

지금은 RouteRec rebuttal 방어선부터 정리하는 것이 우선이다.

---

## 3. 논문 프레이밍 보완

### 3.1 Motivation을 computation heterogeneity로 강화

현재 논문은 다음 정도로 읽힐 수 있다.

> 세션마다 behavior가 다르다. 그래서 MoE를 쓴다.

나쁘지는 않지만, reviewer는 이렇게 물을 수 있다.

> behavior가 다르면 hidden representation이 알아서 다르게 encode하면 되는 것 아닌가?

따라서 motivation을 한 단계 올리는 것이 좋다.

#### 핵심 문장

> The issue is not only that sessions have different representations, but that the same transformation is applied after those representations are formed.

한국어 해석:

> 문제는 세션 표현이 다르다는 것만이 아니라, 그렇게 만들어진 표현을 처리하는 변환 함수가 여전히 공유된다는 점이다.

이렇게 쓰면 RouteRec의 필요성이 더 선명해진다.

---

### 3.2 Cue는 representation feature가 아니라 routing prior

RouteRec이 feature engineering처럼 보이면 약하다.  
cue의 역할을 다음처럼 명확히 해야 한다.

#### 기존 느낌

> behavioral cues are useful features.

#### 강화된 느낌

> behavioral cues are low-dimensional priors over computation paths.

즉 cue는 prediction representation을 풍부하게 만드는 feature가 아니라,  
**어떤 expert computation path를 쓸지 결정하는 control signal**이다.

#### 추천 문장

> RouteRec does not use behavioral cues as richer item representations. Instead, it uses them as compact routing priors that determine how the hidden sequence representation should be transformed.

---

### 3.3 Universal SOTA claim을 피하기

ML-1M 같은 dataset에서 RouteRec이 항상 best는 아니다.  
따라서 “항상 이긴다” 식의 주장은 위험하다.

더 좋은 framing:

> RouteRec helps when the data exhibits behavioral routing demand; when this demand is weak, the advantage becomes smaller.

이렇게 가면 dataset별 gain 차이를 논리 안으로 넣을 수 있다.

---

### 3.4 Behavioral Routing Demand 정의

다음 제출에서는 이 개념을 main text에 올리는 것이 좋다.

#### 정의

> Behavioral routing demand is the degree to which a dataset contains recurring, observable, and performance-relevant differences in session behavior that may benefit from different computation paths.

세 조건으로 풀 수 있다.

1. **Recurring**  
   특정 behavior regime이 충분히 반복해서 나타남

2. **Observable**  
   timestamp, item ID, category, popularity 등으로 측정 가능함

3. **Performance-relevant**  
   그 regime에서 baseline failure 또는 RouteRec gain 차이가 있음

이 개념이 있으면 RouteRec은 단순히 “새 MoE 모델”이 아니라,  
**behavior-dependent computation이 필요한지 진단하는 framework**로 보일 수 있다.

---

## 4. 1개월 일정표

---

### Week 1. Repo freeze + protocol defense

#### 목표

코드와 결과 산출물을 artifact 형태로 정리한다.

#### 할 일

- 새 repo 생성
- paper-track 코드만 이동
- config 정리
- preprocessing script 정리
- result CSV 정리
- README 작성
- `docs/preprocessing.md` 작성
- `docs/leakage_checklist.md` 작성
- main RouteRec run 1~2개 재현 확인

#### 산출물

- clean README
- `reproduce_main_table.sh`
- `docs/preprocessing.md`
- leakage checklist
- paper result CSV

---

### Week 2. 핵심 방어 실험

#### 목표

extra capacity / broken cue / hidden router 관련 의심을 방어한다.

#### 할 일

1. shuffled cue / random cue ablation
2. capacity-matched SASRec-wide
3. hidden-router MoE
4. hidden+cue router
5. 가능하면 cue fusion baseline

#### 추천 dataset

- KuaiRec
- LastFM
- Foursquare

시간 부족 시:

- KuaiRec
- LastFM

#### 산출물

- routing validity table
- capacity control table
- 1개 요약 figure

---

### Week 3. Diagnostic 분석

#### 목표

RouteRec이 실제로 behavior-guided routing을 하는지 분석한다.

#### 할 일

1. cue score ↔ routing weight correlation
2. behavior bin별 route profile
3. behavior bin별 RouteRec gain
4. expert family masking

#### 추천 순서

1. correlation
2. bin profile
3. gain
4. masking

#### 산출물

- correlation heatmap
- bin-wise route/gain plot
- family masking table

---

### Week 4. Rebuttal package 작성

#### 목표

실험 결과를 reviewer concern별로 압축한다.

#### 할 일

- reviewer concern별 답변 템플릿 작성
- 추가 결과 표/그림 3~4개 선택
- appendix에 넣을 수 있는 설명 작성
- artifact repo README 마무리
- 실패한 실험 내부 메모 정리

#### Rebuttal 답변 템플릿

1. **Extra capacity concern**  
   → capacity-matched baseline + shuffled-cue control

2. **Feature engineering concern**  
   → cue is routing prior, not representation feature

3. **Hidden router concern**  
   → hidden router less stable / less behavior-aligned

4. **Leakage concern**  
   → all cues computed from train statistics and observed prefix only

5. **Dataset-dependent gain concern**  
   → RouteRec helps when behavioral routing demand exists

---

## 5. 결과 패턴별 대응 전략

### Case A. Shuffled cue가 크게 떨어짐

가장 좋은 경우다.

주장:

> behaviorally aligned cue가 RouteRec 성능에 필수적이다.

다음 제출에서는 cue-as-routing-prior를 강하게 밀 수 있다.

---

### Case B. Capacity-matched baseline도 비슷하게 좋아짐

위험하다.

대응:

- 성능 claim을 약화
- diagnostic / interpretability 중심으로 이동
- route profile / masking / behavior bin gain을 더 강조

---

### Case C. Cue-route correlation은 나오는데 masking은 약함

해석 가능성은 있지만 functional specialization은 약하다는 뜻이다.

대응:

- “behavior-aligned routing”까지만 주장
- “functionally specialized experts” 주장은 조심
- masking은 appendix 또는 negative analysis로 처리

---

### Case D. ML-1M에서 계속 약함

문제는 아니다.

대응:

- ML-1M은 routing demand가 약한 negative case로 사용
- “RouteRec is not universally better; it helps when behavior regimes are heterogeneous and observable”라고 설명

---

### Case E. Transfer가 안 나옴

rebuttal에는 넣지 않는다.

대응:

- transfer는 future work로 남김
- diagnostic lens를 main으로 유지

---

## 6. 최종 우선순위

실행 우선순위는 다음과 같다.

1. **새 repo 정리 + preprocessing/protocol 문서화**
2. **shuffled/random cue ablation**
3. **capacity-matched SASRec-wide / hidden MoE control**
4. **leakage audit table**
5. **3-seed robustness on selected datasets**
6. **cue score ↔ route weight correlation**
7. **behavior bin별 route/gain 연결**
8. **expert family masking**
9. **dataset-level routing demand 설명 정리**
10. **counterfactual / transfer는 다음 제출로 보류**

---

## 7. 최종 정리

이번 rebuttal 준비의 목표는 RouteRec이 완벽하다는 것을 증명하는 것이 아니다.

현실적인 목표는 다음이다.

> RouteRec의 이득은 단순 capacity 때문이 아니라 behavior-aligned routing에서 나오며, cue construction과 evaluation protocol은 leakage 없이 공정하게 설계되었고, dataset별 gain 차이는 behavioral routing demand 관점에서 설명 가능하다.

이 정도를 방어하면, 설령 이번 제출에서 결과가 좋지 않더라도 다음 제출에서 논문을 훨씬 강하게 다시 쓸 수 있다.

다음 버전의 더 강한 방향은 다음이다.

> RouteRec = behavior-guided recommender  
> 에서  
> RouteRec = behavior-dependent computation을 진단하는 routing framework  
> 로 전환한다.

즉 성능 향상만 주장하는 모델이 아니라,

- 어떤 behavior regime에서 shared computation이 부족한지
- 어떤 cue family가 어떤 expert path를 활성화하는지
- 어떤 subset에서 routed computation이 실제 성능 gain으로 이어지는지
- routing prior가 dataset을 넘어 재사용될 수 있는지

를 보여주는 방향이 더 인상적이다.
