# FMoE N3 Router/Wrapper + Diagnostics 정리안 (Codex 전달용)

## 문서 목적

이 문서는 다음 3가지를 한 번에 정리한다.

1. **모델 구조 측면**
   - router / wrapper를 어떻게 정리하고, 무엇을 추가/수정할지
   - stage별 wrapper 분리와 bias 사용 정책을 어떻게 가져갈지
2. **diag / logging 측면**
   - 너무 많은 로그를 남기지 않고, **논문/분석에 실제로 쓸 핵심 지표만** 남기도록 재구성
   - 특히 wrapper 내부 primitive 동작을 제대로 추적하도록 구조화
   - session/token 분리, generic naming, normalized metric 중심으로 정리
3. **다음 phase 실험 측면**
   - 모든 경우의 수를 다 돌리지 않고, wrapper 중심 비교군을 설계
   - bias / source / stage별 wrapper 분리 실험을 순서 있게 수행

이 문서의 목표는 **코드 구조 정리 + diag 정리 + 다음 실험 설계**를 Codex가 바로 작업할 수 있게 만드는 것이다.

---

# 1. Router / Wrapper 구조 정리

## 1.1 Primitive router는 당장 더 늘리지 않는다

현재 primitive set:

- `a_joint`: flat expert logits
- `b_group`: group logits
- `c_shared`: shared intra logits
- `d_cond`: group-conditional intra logits
- `e_scalar`: group scalar logits

판단:
- 지금은 primitive 종류가 부족한 상태가 아니다.
- 부족한 것은 **어떻게 조합해 최종 routing을 만들지(wrapper)**, 그리고 **그 내부 동작을 어떻게 분석할지(diag)** 쪽이다.
- 따라서 **새 primitive 추가보다 wrapper 정리/추가가 우선**이다.

## 1.2 Wrapper는 1개 추가하고, unused primitive는 계산하지 않도록 정리

### 유지할 wrapper
- `w1_flat`: `A`
- `w2_a_plus_d`: `A + alpha_D * D`
- `w3_bxc`: `B x C`
- `w4_bxd`: `B x D`
- `w5_exd`: `E x D`

### 추가할 wrapper
- **`w6_bxd_plus_a`** (신규)
  - 의미: `structured base + flat correction`
  - 권장 형태:
    - `z_struct = log p(B x D)`
    - `z_final = alpha_struct * z_struct + alpha_A * z_A`
  - 기본 구현은 `alpha_struct=1.0`, `alpha_A=1.0`로 시작
  - 필요하면 이후 `alpha_A`만 config 노출

### 해석
- `w4_bxd`는 가장 정석적인 hierarchical/factorized baseline
- `w6_bxd_plus_a`는 factorized routing 위에 flat router가 residual correction을 하는 형태
- 논문 스토리상 `w6`가 매우 중요할 가능성이 높다:
  - “구조적 routing + 예외 보정”
  - “interpretable factorization + flexible correction”

## 1.3 현재 W2는 바로 삭제하지 않는다

- `w2_a_plus_d`는 지금 구조상 다소 애매하지만,
  - flat + conditional residual 이라는 독립적인 ablation 의미는 있다.
- 따라서 **즉시 삭제/교체하지 말고 유지**
- 다만 우선순위는 `w4`, `w6`, `w1`보다 낮다.

## 1.4 Wrapper마다 필요한 primitive만 계산하도록 변경

현재는 wrapper 종류와 무관하게 A/B/C/D/E를 전부 계산하는 구조인데, 이건 정리 필요.

필수 변경:
- wrapper별 `required_primitives` 정의
- 초기화 시 필요한 primitive만 instantiate
- forward 시 필요한 primitive만 compute
- diag도 사용된 primitive만 기록

예시:
- `w1_flat` -> `["a_joint"]`
- `w4_bxd` -> `["b_group", "d_cond"]`
- `w5_exd` -> `["e_scalar", "d_cond"]`
- `w6_bxd_plus_a` -> `["a_joint", "b_group", "d_cond"]`

이렇게 해야
- 계산 낭비 감소
- diag 구조가 더 깔끔해짐
- wrapper 비교 해석이 쉬워짐

---

# 2. Stage별 Wrapper / Bias 사용 정책

## 2.1 Stage마다 wrapper를 다르게 쓰는 것을 허용/권장

원칙:
- macro / mid / micro는 시간 스케일과 granularity가 다르므로,
- 같은 wrapper를 강제할 필요는 없다.
- **stage-specific wrapper design**을 허용하는 것이 자연스럽다.

### 권장 직관
- `macro`: factorized 성격이 잘 맞을 가능성이 높음
  - 우선 후보: `w4_bxd`, `w5_exd`
- `mid`: factorized 또는 hybrid
  - 우선 후보: `w4_bxd`, `w6_bxd_plus_a`
- `micro`: flat 또는 hybrid
  - 우선 후보: `w1_flat`, `w6_bxd_plus_a`

## 2.2 기본 추천 세팅 후보

### Candidate S1 (보수적)
- macro = `w4_bxd`
- mid = `w4_bxd`
- micro = `w1_flat`

### Candidate S2 (조금 더 강한 구조)
- macro = `w4_bxd`
- mid = `w6_bxd_plus_a`
- micro = `w1_flat`

### Candidate S3 (hybrid 강화)
- macro = `w4_bxd`
- mid = `w6_bxd_plus_a`
- micro = `w6_bxd_plus_a`

이 3개 정도면 stage-specific wrapper 가설을 보기 충분하다.

## 2.3 Bias는 wrapper 본체와 분리해서 취급

현재 bias는 wrapper 내부가 아니라,
**wrapper가 만든 final raw logits에 post-addition** 되는 구조다.

### feature_group_bias
- group feature magnitude 기반 prior를 만들고
- expert 단위로 expand한 뒤
- `raw_logits`에 더하는 방식

### rule_bias
- rule router가 만든 logits를
- `raw_logits`에 더하는 방식

둘 다 **softmax/top-k 전 logit bias**이다.

## 2.4 Bias 사용 원칙

### 원칙 A. Core wrapper 비교 실험에서는 bias OFF
wrapper 자체 비교를 할 때는 아래를 기본값으로 둔다.

- `feature_group_bias_lambda = 0`
- `rule_bias_scale = 0`

이유:
- wrapper 효과와 bias 효과가 섞이지 않게 하기 위함
- 논문 스토리상도 더 깔끔함

### 원칙 B. Bias는 “추가 augmentation” 실험으로 분리
bias는 wrapper 자체가 아니라 다음처럼 별도 phase에서 실험.

- best wrapper + no bias
- best wrapper + feature_group_bias
- best wrapper + rule_bias
- best wrapper + both

즉 bias는 **routing augmentation / prior injection**으로 취급한다.

## 2.5 Primitive source 실험은 제한적으로만 수행

source 실험은 전부 다 돌리지 말고, 아래 정도만 본다.

### 기본값
- A/B/C: `both`
- D/E: `feature`

### 추가 비교 후보
- D/E: `both`
- A/B: `feature`
- A only: `hidden`, B/D/E는 기본 유지

source 실험은 wrapper 핵심 비교가 끝난 뒤, 소수 후보에만 적용한다.

---

# 3. Diagnostics / Logging 정리안 (핵심만)

## 목표

diag는 “많이 저장”이 아니라 **논문과 분석에 실제로 쓸 것만 남기는 것**이 목표다.

원칙:
- raw count 위주 저장 금지
- **normalized metric 중심**
- session/token 분리
- generic naming
- wrapper/primitive/final을 같은 schema에서 다룰 수 있게 설계
- used primitive만 기록

---

## 3.1 로그 층 구조

diag는 3개 층만 남긴다.

### Tier A. Final effective routing
최종적으로 실제 사용된 routing 결과

### Tier B. Internal routing
wrapper 안의 primitive / wrapper-level routing 동작

### Tier C. Feature alignment + visualization payload
feature와 routing의 관계, 논문용 PCA/scatter export

이 3개만 유지한다.

---

## 3.2 공통 메타 필드 (모든 stage/node 공통)

각 로그 단위에 아래 메타는 공통 포함:

- `stage_name`: macro / mid / micro
- `aggregation_level`: session / token
- `node_kind`: final / wrapper / primitive
- `node_name`
  - 예: `final.expert`
  - `wrapper.group`
  - `wrapper.intra`
  - `primitive.a_joint`
  - `primitive.b_group`
  - `primitive.d_cond`
- `route_space`: expert / group / intra
- `support_size`
- `wrapper_name`
- `top_k_final`
- `source_type` (primitive only)
- `temperature` (primitive only)
- `is_bias_feature_group`
- `is_bias_rule`

주의:
- **session stage는 session aggregation과 token aggregation을 모두 만들 수 있지만, 서로 다른 로그로 저장**
- 서로 섞지 않는다.

---

## 3.3 Tier A: Final effective routing (핵심 지표만)

이 층은 **논문 표/핵심 요약용**이다.

## A-1. final expert routing
`node_name = final.expert`

저장 지표:
- `entropy_norm`
  - `entropy / log(E)`
- `n_eff_norm`
  - `n_eff / E`
- `top1_monopoly_norm`
  - `(top1_max_frac - 1/E) / (1 - 1/E)`
- `jitter_adj_norm`
  - 인접 토큰 top1 변화율
- `knn_consistency_score`
  - `exp(-JS)` 기반 normalized score 사용
  - 기존 final expert KNN consistency 유지하되 naming만 generic하게 정리

## A-2. final group routing
`node_name = final.group`

저장 지표:
- `entropy_norm`
  - `entropy / log(G)`
- `n_eff_norm`
  - `n_eff / G`
- `top1_monopoly_norm`
  - `(top1_max_frac - 1/G) / (1 - 1/G)`
- `knn_consistency_score`

## A-3. final intra routing
group별로 분리 저장
`node_name = final.intra.<group_name>`

저장 지표:
- `entropy_norm`
- `n_eff_norm`
- `top1_monopoly_norm`
- `knn_consistency_score`

주의:
- intra는 group마다 support size가 다를 수 있으므로, 항상 group별 독립 node로 기록
- 비교 시 support_size 기준 정규화값만 사용

---

## 3.4 Tier B: Internal routing (wrapper/primitive 내부 추적)

이 층이 이번 정리의 핵심이다.

목표:
- primitive가 실제로 어떤 분포를 냈는지
- wrapper가 어떤 intermediate group/intra 구조를 만들었는지
- 최종 route와 얼마나 일치/불일치하는지

단, 너무 많은 지표를 남기지 않는다.

## B-1. Primitive 공통 저장 지표
used primitive만 기록

예:
- `primitive.a_joint`
- `primitive.b_group`
- `primitive.c_shared`
- `primitive.d_cond`
- `primitive.e_scalar`

저장 지표:
- `entropy_norm`
- `n_eff_norm`
- `top1_monopoly_norm`
- `confidence_mean`
  - 평균 top1 prob
- `agreement_to_final`
  - final 같은 space로 비교 가능하면 `exp(-JS)` score로 저장
- `knn_consistency_score`
  - primitive input space 기준으로 KNN을 만들고,
  - primitive own output distribution의 consistency를 측정

설명:
- `agreement_to_final`은 internal node가 최종 route를 얼마나 설명하는지 보는 핵심 지표
- `knn_consistency_score`는 feature/hidden input이 유사한 샘플끼리 primitive가 비슷한 route를 내는지 보는 핵심 지표

## B-2. Wrapper-level intermediate node
factorized wrapper가 있을 때만 기록

### `wrapper.group`
저장 지표:
- `entropy_norm`
- `n_eff_norm`
- `top1_monopoly_norm`
- `agreement_to_final`
  - final group과의 agreement score

### `wrapper.intra.<group_name>`
저장 지표:
- `entropy_norm`
- `n_eff_norm`
- `top1_monopoly_norm`
- `agreement_to_final`
  - final intra.<group_name>와의 agreement score

## B-3. 꼭 넣을 wrapper distortion 지표
factorized wrapper에만 공통 저장

- `group_prepost_agreement`
  - wrapper.group vs final.group
- `intra_prepost_agreement_mean`
  - wrapper.intra vs final.intra의 group별 agreement 평균

이 2개는 매우 중요하다.
이 값이 낮으면:
- wrapper가 구조적으로 factorization을 만들었더라도
- 최종 top-k / correction / bias 때문에 실제 최종 route는 다른 형태가 되었다는 뜻이다.

즉 논문에서 wrapper 구조를 해석할 때 반드시 필요하다.

---

## 3.5 Tier C: Feature alignment + 논문용 visualization payload

이 층은 수치 요약 + 시각화 export로 나눈다.

## C-1. 최소 수치 요약 지표
너무 많지 않게 아래 3개만 남긴다.

### feature-family-specific consistency
`node_name = feature_align.<group_name>`

저장 지표:
- `consistency_to_final_group`
- `consistency_to_final_intra`
- `consistency_to_primitive`
  - 해당 group 관련 primitive가 있으면 연결
  - 예: D/E가 있으면 그쪽 consistency

설명:
- 특정 feature group으로 KNN을 만들었을 때,
- routing도 비슷한지 보는 지표
- feature가 실제로 routing specialization에 기여하는지 보는 최소 요약

### group bin response summary
각 group feature에 대해 quantile bin 기반 summary를 만든다.

단, 통계 전부를 저장하지 말고 아래 2개만 저장:
- `bin_monotonicity_to_group`
- `bin_monotonicity_to_confidence`

즉,
- feature group score가 커질수록 해당 group 선택이 증가하는가
- feature group score가 커질수록 confidence가 증가하는가

이건 Spearman 기반 normalized score로 저장

## C-2. PCA / scatter export (강조)

이 부분은 **논문용 그림**을 위해 꼭 남긴다.

중요:
- 이것은 scalar metric이 아니라 **sampled payload export**다.
- stage/session/token을 잘 분리하고, wrapper 구조 비교에 공통적으로 쓸 수 있게 generic schema로 저장한다.

### export 원칙
- stage별 별도 저장
- session stage와 token stage 분리
- wrapper 구조와 무관하게 동일 schema
- used primitive가 있으면 primitive별 색칠도 가능하게 meta 포함
- 너무 많은 샘플 저장 금지
- reservoir sampling 또는 capped sampling 사용

### 저장 대상 (권장 3종)
#### (1) `viz.feature_pca`
입력:
- stage feature 또는 encoded feature

저장 필드:
- `pc1`, `pc2`
- `stage_name`
- `aggregation_level`
- `wrapper_name`
- `final_top1_group`
- `final_top1_expert`
- `final_confidence`
- `session_length` (가능하면)
- `group_feature_scores` (요약값만)

목적:
- “feature 공간에서 final route가 어떻게 갈리는지”를 공통 비교

#### (2) `viz.router_input_pca`
입력:
- primitive가 실제 본 input
- stage-level primitive는 stage input
- group-level primitive는 group input

저장 필드:
- `pc1`, `pc2`
- `stage_name`
- `aggregation_level`
- `node_name`
- `top1_label`
- `confidence`
- `wrapper_name`

목적:
- primitive/router input 공간에서 routing이 어떻게 갈리는지 비교

#### (3) `viz.group_feature_pca`
입력:
- group feature projection

저장 필드:
- `pc1`, `pc2`
- `stage_name`
- `aggregation_level`
- `group_name`
- `final_top1_group`
- `final_top1_expert`
- `primitive_top1` (가능하면 D/E)
- `confidence`

목적:
- group-specific feature 공간과 routing 결과를 연결해서 논문 그림으로 사용

### 반드시 지킬 점
- PCA 좌표만 저장하지 말고, **색칠에 필요한 label/meta를 같이 저장**
- 그래야 나중에 wrapper 구조별 비교 그림을 쉽게 뽑을 수 있음
- session/token stage는 절대 한 plot로 섞지 않음

---

# 4. Generic Naming / Aggregation 규칙

## 4.1 Naming은 generic하게 유지

권장 형식:

- `final.expert`
- `final.group`
- `final.intra.<group_name>`
- `wrapper.group`
- `wrapper.intra.<group_name>`
- `primitive.a_joint`
- `primitive.b_group`
- `primitive.c_shared`
- `primitive.d_cond`
- `primitive.e_scalar`
- `feature_align.<group_name>`
- `viz.feature_pca`
- `viz.router_input_pca`
- `viz.group_feature_pca`

wrapper가 바뀌어도 같은 schema 안에서 비교 가능해야 한다.

## 4.2 Aggregation 규칙

- session stage는 `aggregation_level=session`과 `aggregation_level=token` 분리
- micro/token stage는 기본 `aggregation_level=token`
- support size가 달라지는 경우:
  - entropy는 `entropy_norm`
  - n_eff는 `n_eff_norm`
  - monopoly는 `top1_monopoly_norm`
- raw usage count / raw entropy는 기본 diag 저장에서 제외

즉 비교는 항상 normalized metric 기준으로 한다.

---

# 5. 다음 Phase 실험 계획

## 목표

- 모든 경우의 수를 다 실험하지 않는다.
- 기존 실험에서 괜찮았던 backbone / training / regularization 설정은 고정한다. (regularization은 모두 없는 채로 고정)
- 이번 phase에서는 **router/wrapper/bias/source/stage-specific design만 비교**한다.

즉, router-related factor만 통제된 실험으로 본다.

---

## 5.1 고정할 것

아래는 이전 best 또는 stable 세팅으로 고정:

- backbone
- optimizer / lr space / wd / dropout
- expert hidden/depth
- batch size
- training length / early stopping
- non-router regularizer
- evaluation protocol
- dataset split
- feature engineering / preprocessing

주의:
- 이번 phase에서는 router 구조 및 bias 관련 설정만 바꾸고,
- 나머지 조건은 가급적 건드리지 않는다.

---

## 5.2 Phase A — Wrapper core 비교 (bias 모두 OFF)

bias는 모두 꺼둔다.

- `feature_group_bias_lambda = 0`
- `rule_bias_scale = 0`

비교 후보:
1. `all_w1`
3. `all_w2`
2. `all_w3`
2. `all_w4`
2. `all_w5`
3. `all_w6`
4. `stage_split_s1`
   - macro=w4, mid=w4, micro=w1
5. `stage_split_s2`
   - macro=w4, mid=w6, micro=w1
5. `stage_split_s3`
   - macro=w6, mid=w1, micro=w1

여기서 우선 4~5개 정도만 본다.

### 목적
- flat vs factorized vs hybrid
- uniform wrapper family vs stage-specific wrapper family

### 추천 실행
- 1차: 각 후보 1 seed
- 2차: 상위 2~3개 후보만 3 seeds

이렇게 해야 “운 좋게 한 번 뜬 설정”을 줄일 수 있다.

---

## 5.3 Phase B — Bias augmentation 비교

Phase A best wrapper family를 고정하고 bias만 비교.

비교 후보:
1. no bias
2. feature_group_bias only
3. rule_bias only
4. both bias

### 목적
- wrapper 구조와 무관한 augmentation으로 bias가 유의미한지 확인
- core routing 효과와 bias 효과 분리

### 주의
- Phase A 이전에는 bias 실험하지 않는다.

---

## 5.4 Phase C — Source 비교 (소수 후보만)

best 1~2개 wrapper family에서만 source 비교.

권장 비교:
1. 기본값
   - A/B/C=`both`, D/E=`feature`
2. D/E=`both`
3. A/B=`feature`, D/E=`feature`
4. A=`hidden`, B/D/E 기본

### 목적
- group-feature router가 hidden까지 받아야 하는지
- stage router가 feature only여도 충분한지
- hidden vs feature 역할 분리 가능성 확인

### 주의
- source 실험은 wrapper core 비교 후에만 한다.
- 처음부터 source 조합 전체를 탐색하지 않는다.

---

## 5.5 Phase D — top-k

Phase A/B/C 세팅에서 top-k를 조금씩 바꿔보기. global 기준 50%만 사용하거나, 25%로 상당히 적게 볼 수도 있음
혹은 group별로 1개 expert만 써서 group마다 하나만 할당해서 사용하는 느낌. group specialist 느낌도 가능
primitive router마다 top-k 설정 가능하므로 wrapper에 따라서 들어가는 구조가 달라짐

예시:
- global top-k: 최종 weight에서 상위 N개만 남기고, 그 아래는 아예 0(1e-9)로 고정, 나머지 활성 expert의 weight는 다시 합 1로 normalize
- global group top-k: 최종 weight에서 그룹별로 N개만 남기고, 그 아래는 아예 0(1e-9)로 고정, 나머지 활성 expert의 weight는 다시 합 1로 normalize
- primitive router에서 A같이 모든 expert를 할당하는 router에서 라우터 단위로 top-k 활용
- group 위주 router에서 group안에서 일부만 사용하도록
router/wrapper 종류에 따라, 직관적으로 생각했을 때 specialization 위해서 어떻게 볼 수 있을 지 생각해서 combo 생성

### 목적
- 일부 expert만 선택해서 활성화 하는 것이 실제로 도움이 되는지 확인
- global/group마다 다르게 두거나 primitive router마다 활용할 때 어떻게 달라지는 지 확인

---

## 5.6 실험 로그 해석 우선순위

실험 결과를 볼 때는 아래 순서로 본다.

### 1순위: 성능
- HR / NDCG / MRR 등 main metric
- best valid MRR@20이 가장 중요, 그 다음 test MRR@20, best valid HR@10, test HR@10 순

### 2순위: final routing 핵심 지표
- `final.expert.n_eff_norm`
- `final.group.n_eff_norm`
- `final.expert.knn_consistency_score`
- `final.group.knn_consistency_score`
- `final.expert.top1_monopoly_norm`

### 3순위: internal routing 해석 지표
- `wrapper.group_prepost_agreement`
- `wrapper.intra_prepost_agreement_mean`
- `primitive.*.agreement_to_final`
- `primitive.*.knn_consistency_score`

### 4순위: feature alignment / visualization
- `feature_align.*`
- PCA/scatter export

즉, diag는 많이 보지 말고 위 순서로 좁혀서 본다.

---

# 6. Codex 구현 우선순위

## Step 1. Router/Wrapper 구조 정리
- `w6_bxd_plus_a` 추가
- wrapper별 required primitive만 계산하도록 리팩토링
- bias on/off가 명확히 분리되도록 config/aux 정리

## Step 2. Internal routing diag 추가
- primitive / wrapper / final을 generic node schema로 기록
- used primitive만 저장
- normalized metric만 저장

## Step 3. Final 핵심 diag 정리
- 기존 산발적 metric을 위 schema로 통합
- session/token aggregation 분리
- raw count 저장 축소

## Step 4. Visualization payload export 추가
- feature PCA
- router input PCA
- group feature PCA
- reservoir/capped sampling 적용
- label/meta 포함 저장
- 지금의 logging 폴더에 데이터 json/md로 저장하고, PCA는 feature마다 분리해서 값으로 두고, 그에 따른 expert weight나 loss/metric csv나 json으로 깔끔하게 저장하도록.

## Step 5. Phase A 실험 config 세트 작성
- wrapper core 비교용 4~5개 config
- bias OFF 기본
- 나머지 세팅 고정

---

# 7. 최종 요약

이번 phase 핵심은 아래 3가지다.

1. **wrapper 구조 비교를 깔끔하게 한다**
   - primitive 추가보다 wrapper 추가/정리가 우선
   - 특히 `w6_bxd_plus_a`를 넣고 비교
2. **diag를 “핵심만” 남긴다**
   - final / internal / feature+viz 의 3층만 유지
   - normalized metric 중심
   - session/token 분리
   - generic naming
3. **실험 순서를 단순화한다**
   - Phase A: wrapper core 비교 (bias OFF)
   - Phase B: bias augmentation
   - Phase C: source 비교
   - Phase D: stage-specific refinement

이 문서 기준으로 구현/정리하면,
- 모델 구조
- diag 구조
- 다음 실험 계획
이 한 번에 정리될 수 있다.
