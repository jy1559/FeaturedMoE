# Phase 13 — Feature Sanity / Category-Zero / Corruption Checks

## 1. Phase 목표

이 phase의 핵심 질문은 다음이다.

> **FMoE의 성능 향상이 정말 feature hint 때문인가? 아니면 feature branch 파라미터가 늘어난 부수 효과인가?**

즉 이 phase는

- feature를 inference 시 망가뜨리면 성능이 떨어지는지,
- 학습 데이터 자체에서 feature-sequence alignment를 깨뜨리면 무너지는지,
- family 단위 corruption에도 민감한지,
- category-derived feature를 0 처리한 데이터 버전에서도 framework가 유지되는지

를 확인하는 sanity check phase다.

논문에서는 이 phase를 통해 다음을 보여주고 싶다.

1. **feature는 실제로 routing hint로 쓰인다**
2. **분포만 비슷한 feature가 아니라, aligned feature가 중요하다**
3. **category signal이 없어도 framework는 유지될 수 있다**

---

## 2. 이 phase에서 주장하고 싶은 것

### main claim

> **The gain of FeaturedMoE comes from aligned feature-guided routing, not from simply adding an auxiliary feature branch with more parameters.**

### 이 phase가 잘 나오면 하고 싶은 말

- eval shuffle / zero에서 drop이 크면:
  - “trained model이 inference 시 feature를 실제로 참고한다”
- train corruption에서 성능이 크게 떨어지면:
  - “분포가 아니라 alignment가 중요하다”
- family별 corruption sensitivity 차이가 보이면:
  - “어떤 family가 실제 routing hint로 중요한지”를 분석 가능
- category-zero data에서도 framework가 유지되면:
  - “category가 필수는 아니고, 있으면 더 좋다”

### 기대와 다르게 나와도 쓸 수 있는 말

- eval perturbation drop은 작지만 train corruption drop은 크면:
  - “inference robustness는 있지만 학습 시 aligned feature가 중요하다”
- 일부 family corruption에만 민감하면:
  - “gain은 sparse subset of features에서 온다”
- category-zero에서 크게 하락하면:
  - “category-derived behavioral abstraction이 현재 framework의 중요한 축이다”

---

## 3. 이 phase의 철학

이 phase는 **잘 되기를 바라는 phase가 아니라, 일부러 망가뜨려서 해석 가능성을 확보하는 phase**다.

즉,
- 성능이 낮아지는 것이 오히려 성공적인 결과일 수 있고,
- clean setting과 corrupted setting의 차이가 클수록
  “feature hint를 실제로 쓰고 있다”는 설득력이 커진다.

따라서 여기서는 best setting 탐색보다,
**망가뜨렸을 때 왜 무너지는지**를 잘 설명하는 것이 중요하다.

---

## 4. Phase 구성 요약

총 **24개 세팅**.

- 4.1 Category-zero data condition: **2개**
- 4.2 Eval-time perturbation: **6개**
- 4.3 Train-time corruption: **6개**
- 4.4 Semantic mismatch / role mismatch: **3개**
- 4.5 8배수 보강: **7개**

---

## 4.1 Category-zero data condition (2개)

### 목적

이 축은 “category-derived signal이 없는 데이터 조건”을 value perturb 방식으로 시뮬레이션한다.

중요한 점은,
이건 Phase 10의 `NO_CATEGORY`와 비슷해 보이지만 목적이 다르다.

- Phase 10 `NO_CATEGORY`: model spec 차원에서 category-related feature를 제거하는 **availability ablation**
- Phase 13 `CATEGORY_ZERO_DATA`: 구조는 유지한 채 category/theme 컬럼 값만 0으로 만들어, **category 신호가 사라진 조건**을 흉내내는 sanity / robustness check

구현 정책(고정):
- `stage_feature_drop_keywords`를 쓰지 않는다 (구조 제거 금지)
- `feature_perturb_mode=zero`, `feature_perturb_apply=both`
- `feature_perturb_keywords=["cat","theme"]`로 타겟 컬럼만 0 처리

### 세팅 목록

- `P13-00_FULL_DATA`
- `P13-01_CATEGORY_ZERO_DATA`

### 이 축에서 보고 싶은 것

- 성능이 어느 정도 유지되면:
  - “category가 없어도 framework는 유지 가능”
- 많이 하락하면:
  - “category-derived behavior abstraction이 중요한 역할을 한다”

### 해석 포인트

이 실험은 “category 없어도 잘 된다”를 무조건 보여주는 용도라기보다,
**category signal이 사라졌을 때 어느 정도까지 버티는지**를 보는 실험이다.

즉 적당히 떨어져도 충분히 의미 있다.

---

## 4.2 Eval-time perturbation (6개)

### 목적

이 축은 이미 학습된 모델이 inference 시 feature alignment를 실제로 사용하는지 확인한다.

즉,
- 학습은 정상적으로 했는데,
- 평가 시 feature를 0으로 만들거나 섞었을 때 성능이 떨어지는가?

를 본다.

이건 가장 직접적인 “feature usage” 증거다.

### 세팅 목록

- `P13-02_EVAL_ALL_ZERO`
- `P13-03_EVAL_ALL_SHUFFLE`
- `P13-04_EVAL_SHUFFLE_TEMPO`
- `P13-05_EVAL_SHUFFLE_FOCUS`
- `P13-06_EVAL_SHUFFLE_MEMORY`
- `P13-07_EVAL_SHUFFLE_EXPOSURE`

### 각 세팅이 의미하는 것

- `EVAL_ALL_ZERO`
  - feature를 통째로 0으로 바꿈
- `EVAL_ALL_SHUFFLE`
  - feature distribution은 유지하되 sample alignment를 깨뜨림
- `EVAL_SHUFFLE_*`
  - 특정 family만 섞어서 어느 family가 중요한지 확인

### 이 축이 잘 나왔을 때 해석

- `ALL_ZERO`, `ALL_SHUFFLE`에서 drop 큼:
  - “학습된 모델이 feature를 실제로 사용한다”
- family-specific shuffle에서 drop 차이 발생:
  - “어떤 family가 routing hint로 더 중요했는지”를 직접 보여줌

### 다른 결과가 나왔을 때 해석

- eval perturbation drop이 작으면:
  - “학습 시 feature가 중요한 역할을 했지만, inference 시 hidden representation에 일부 흡수되었을 가능성”

---

## 4.3 Train-time corruption (6개)

### 목적

이 축은 “파라미터만 늘어서 좋아진 것 아니냐”는 반박에 가장 강하게 대응한다.

핵심은,
- feature branch 구조는 그대로 두되,
- feature와 sequence의 alignment만 깨뜨려서 학습시키는 것.

즉,
**분포는 비슷하지만 정렬이 틀린 feature**를 주었을 때 성능이 무너지면,
이 모델은 parameter 수가 아니라 aligned feature signal을 학습하고 있다고 볼 수 있다.

### 세팅 목록

- `P13-08_TRAIN_GLOBAL_PERMUTE_ALL`
- `P13-09_TRAIN_BATCH_PERMUTE_ALL`
- `P13-10_TRAIN_PERMUTE_TEMPO`
- `P13-11_TRAIN_PERMUTE_FOCUS`
- `P13-12_TRAIN_PERMUTE_MEMORY`
- `P13-13_TRAIN_PERMUTE_EXPOSURE`

### 각 세팅이 의미하는 것

- `GLOBAL_PERMUTE_ALL`
  - 전체 sample 단위로 feature를 섞음
  - 가장 강한 alignment 파괴
- `BATCH_PERMUTE_ALL`
  - batch/session local 수준에서 섞음
  - 더 약한 corruption
- family별 permutation
  - 특정 family만 misaligned하게 둠

### 이 축이 잘 나왔을 때 해석

- clean train > corrupted train:
  - “aligned feature signal이 학습에 실제로 필요했다”
- family별 corruption sensitivity 차이:
  - “어떤 family가 more causal-like hint인지”를 보여줌
- global이 batch보다 더 나쁘면:
  - “misalignment 강도에 따라 성능이 단계적으로 무너진다”는 서사 가능

### 다른 결과가 나왔을 때 해석

- corruption drop이 예상보다 작으면:
  - “모델이 hidden-only path로도 상당 부분 보정 가능하다”
- 일부 family만 크게 민감하면:
  - “정말 중요한 family는 sparse subset이다”

---

## 4.4 Semantic mismatch / role mismatch (3개)

### 목적

이 축은 zero / shuffle보다 한 단계 더 세밀하다.

즉,
- feature가 완전히 사라진 것도 아니고,
- distribution도 어느 정도 유지되지만,
- **semantic role만 틀리게 준 경우**
모델이 얼마나 민감한지 보는 실험이다.

이 실험은 “feature가 단순 숫자 입력이 아니라 의미 있는 hint다”라는 말을 더 강하게 만들어 준다.

### 세팅 목록

- `P13-14_FEATURE_ROLE_SWAP`
- `P13-15_STAGE_MISMATCH_ASSIGN`
- `P13-16_POSITION_SHIFT_FEATURE`

### 각 세팅이 의미하는 것

- `FEATURE_ROLE_SWAP`
  - 기본 페어를 `Tempo↔Exposure`, `Focus↔Memory`로 고정
  - distribution은 어느 정도 남지만 의미가 뒤틀림

- `STAGE_MISMATCH_ASSIGN`
  - macro용 feature를 mid stage에 주거나, mid용을 micro에 주는 식
  - feature와 stage semantics 사이 정렬을 깨뜨림

- `POSITION_SHIFT_FEATURE`
  - sequence 내에서 feature를 한 칸씩 밀거나 shift
  - temporal alignment를 깨뜨림

### 이 축이 잘 나왔을 때 해석

- semantic mismatch에서도 성능 하락:
  - “feature는 단순 side input이 아니라 의미와 정렬을 가진 routing hint”
- `POSITION_SHIFT_FEATURE`가 특히 나쁘면:
  - “temporal alignment가 실제로 중요하다”
- `STAGE_MISMATCH_ASSIGN`이 나쁘면:
  - “feature와 stage semantics의 대응이 중요하다”

---

## 4.5 8배수 보강 (7개)

GPU 8개 병렬 실행 매트릭스를 맞추기 위한 sanity 보강 세팅이다.

### 세팅 목록

- `P13-17_EVAL_ZERO_TEMPO`
- `P13-18_EVAL_ZERO_FOCUS`
- `P13-19_EVAL_ZERO_MEMORY`
- `P13-20_EVAL_ZERO_EXPOSURE`
- `P13-21_TRAIN_POSITION_SHIFT_PLUS1`
- `P13-22_TRAIN_POSITION_SHIFT_PLUS2`
- `P13-23_TRAIN_POSITION_SHIFT_PLUS3`

### 보강 의도

- eval zero family 분해(4개):
  - family별 정보 제거 민감도를 더 정밀하게 분리
- train position shift 강도 스윕(3개):
  - temporal misalignment 강도(shift=1/2/3)별 하락 곡선 확인

---

## 5. 이 phase에서 꼭 보고 싶은 결과 패턴

### 가장 이상적인 패턴

1. `FULL_DATA` > `CATEGORY_ZERO_DATA`
2. eval zero / shuffle에서 성능 drop 존재
3. train corruption에서 더 큰 성능 drop 존재
4. semantic mismatch에서도 성능 저하 발생

이 패턴이면 논문에서 다음을 매우 강하게 말할 수 있다.

> **FeaturedMoE benefits from aligned feature-guided routing; merely adding a feature branch is not enough if the feature-to-sequence or feature-to-stage alignment is broken.**

### 차선의 패턴

1. eval perturbation drop은 작음
2. train corruption / semantic mismatch에서만 유의미한 drop

이 경우에도 충분히 좋은 결과다.
이때는
- inference 시에는 hidden representation이 일부 보정하지만,
- 학습 시 aligned feature가 중요했다
로 정리하면 된다.

---

## 6. 추천 분석 포인트

### main table

- clean vs corrupted overall metric
- clean vs corrupted cold-item metric

### figures

1. **eval perturbation drop plot**
2. **train corruption drop plot**
3. **family별 corruption sensitivity heatmap**
4. **category-zero vs full comparison**

### supplementary

- corruption 전후 routing entropy / top1 concentration 변화
- corruption 전후 route consistency 변화
- family별 corruption impact ranking

---

## 7. 구현 메모

### eval-time perturbation

현재 모델의 feature ablation / shuffle 기능을 최대한 재사용한다.

### train-time corruption

preprocessing 또는 dataloader 레벨에서
- feature permutation
- family-specific permutation
- position shift
를 적용한 dataset version을 만드는 것이 가장 깔끔하다.

### semantic mismatch

- family mapping 자체를 바꾸거나,
- stage별 feature assignment를 의도적으로 어긋나게 하는 override가 필요할 수 있다.

즉 이 부분은 config만으로 안 되면 작은 adapter 층을 추가해도 된다.

---

## 8. 최종적으로 논문에 남기고 싶은 문장

가장 이상적인 결과가 나왔을 때:

> **The gain of FeaturedMoE comes from aligned feature-guided routing rather than from simply increasing model capacity with an auxiliary feature branch.**

대안 결과가 나왔을 때:

> **The sanity checks show that FeaturedMoE is particularly sensitive to misaligned feature supervision during training, indicating that the model learns meaningful feature-to-routing correspondences rather than merely exploiting additional parameters.**
