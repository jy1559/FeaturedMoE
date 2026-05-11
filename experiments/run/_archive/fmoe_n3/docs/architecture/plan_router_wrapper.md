# N3 Router Wrapper 아키텍처 정리 (KOR)

## 1. 문서 목적

이 문서는 N3 라우팅 구조를 다음 두 계층으로 분리해 설명한다.

- Primitive Router (a~e): "어떤 종류의 분포를 예측하는가"
- Wrapper Router (w1~w5): "primitive 출력을 어떻게 조합해 최종 expert gate를 만드는가"

핵심 목표는 수식 의미와 코드 구현을 1:1로 맞추고, 실험 조합을 쉽게 바꾸는 것이다.

## 2. 고정 Notation (기준 예시)

- 그룹 수: `G=4` (Tempo, Focus, Memory, Exposure)
- 그룹당 feature 수: `F_g=4`
- stage feature 총합: `F=16`
- 그룹당 expert 수: `C=3` (기본값, override 허용)
- 전체 expert 수: `E=G*C=12`

### 입력 텐서 기호

- `h`: hidden 기반 입력 (`B x S x D_h`)
- `x_stage`: stage feature 기반 입력 (`B x S x D_f`)
- `x_group[g]`: g번째 그룹 feature 기반 입력 (`B x S x D_f_group`)

코드에서는 `x_stage`를 stage 전체 feature에서 만들고, `x_group[g]`는 group별 raw feature를 projection해서 만든다.

### 확률 기호

- `p(g,e)`: 그룹-전문가 joint 분포
- `p(g)`: inter-group 분포
- `p(e|g)`: 그룹 조건부 intra 분포
- `p(e)`: 그룹 공유 intra 분포 (길이 `C`, 모든 그룹에 복제 사용)

## 3. Primitive Router 5종

각 primitive는 독립적으로 아래 파라미터를 가진다.

- `source`: `hidden | feature | both`
- `temperature`: primitive 내부 softmax 온도
- `top_k`: primitive 내부 sparsity 제어

### A: StageJointExpertRouter (`A_JOINT`)

- 출력: `z_A in R^E`
- 의미: 전체 expert 12개를 한 번에 직접 예측하는 flat logits

### B: StageGroupRouter (`B_GROUP`)

- 출력: `z_B in R^G`
- 의미: 그룹 분포 logits (`p(g)`용)

### C: StageSharedIntraRouter (`C_SHARED`)

- 출력: `z_C in R^C`
- 의미: 그룹 구분 없이 공유되는 intra 분포 logits (`p(e)`용)

### D: GroupConditionalIntraRouter (`D_COND`)

- 출력: `z_D in R^{G x C}`
- 의미: 그룹별 조건부 intra logits (`p(e|g)`)

### E: GroupScalarRouter (`E_SCALAR`)

- 출력: `z_E in R^G`
- 의미: group input에서 얻은 scalar를 모아 group logits 생성 (`softmax`로 `p(g)`)

## 4. Wrapper Router 5종

wrapper는 primitive 출력을 받아 최종 expert logits(`R^E`)를 만든다.

### W1: FlatJointWrapper (`w1_flat`)

- 수식: `z = z_A`
- 의미: standard(flat) baseline

### W2: FlatPlusGroupIntraResidualWrapper (`w2_a_plus_d`)

- 수식: `z_{g,e} = z_A(g,e) + alpha_D * z_D(g,e)`
- 의미: flat logits에 group-conditional residual을 더함

### W3: GroupSharedIntraProductWrapper (`w3_bxc`)

- 수식: `p(g,e)=p_B(g)*p_C(e)`
- 구현: `p_B=softmax(z_B)`, `p_C=softmax(z_C)`, 이후 outer product

### W4: GroupConditionalProductWrapper (`w4_bxd`)

- 수식: `p(g,e)=p_B(g)*p_D(e|g)`
- 의미: 계층형(HIR 의미) 분해

### W5: ScalarGroupConditionalProductWrapper (`w5_exd`)

- 수식: `p(g)=softmax(z_E)`, `p(g,e)=p(g)*p_D(e|g)`
- 의미: group 확률을 E primitive에서 얻는 변형 계층형

### 확률형 wrapper의 logit 변환

`w3~w5`는 확률공간에서 조합한 뒤 최종 expert logits로 바꿔 downstream과 통일한다.

- 기본 아이디어: `log p(g,e)`
- 구현 주의: 0 확률 슬롯은 `-1e9`로 마스킹 유지
  - 이유: primitive top-k로 만든 sparsity가 `eps` 때문에 되살아나는 문제 방지
  - 결과: 예를 들어 `W5 + D(top_k=1)`이면 그룹당 1개 expert 활성 패턴을 유지 가능

## 5. Router 입력 파라미터 상세

### 5.1 `source` (`hidden | feature | both`)

source는 primitive별로 독립 설정한다.

- `hidden`: hidden만 사용
- `feature`: feature만 사용
- `both`: hidden과 feature를 마지막 차원 concat

현재 구현 규칙은 단순 concat 유지:

```python
# stage_modules.py::_compose_input_from_source
if source == "both":
    router_input = torch.cat([hidden, feature], dim=-1)
```

### 5.2 `temperature`

- primitive 내부 logits를 `logits / temperature`로 스케일링
- 값이 클수록 분포가 완만해지고, 작을수록 sharpen
- primitive마다 다르게 줄 수 있음

### 5.3 `top_k`

- primitive softmax 전에 top-k 마스킹
- `<=0`이면 비활성(전체 유지)
- `k >= 슬롯 수`면 사실상 비활성
- 예: `d_cond.top_k=1`이면 각 group에서 intra 1개만 활성

### 5.4 Stage 최종 top-k와의 관계

- primitive top-k: wrapper 내부 조합 전에 적용
- stage 최종 top-k(`moe_top_k`/schedule): wrapper 결과 expert logits에 적용
- 즉, 두 단계로 sparsity를 제어할 수 있다.

## 6. Stage 입력 구성 (코드 관점)

실제 stage에서 primitive에 넣는 텐서는 다음처럼 구성된다.

- stage-level primitive(A/B/C):
  - hidden 후보: `hidden`
  - feature 후보: `encoded_feat`
- group-level primitive(D/E):
  - hidden 후보: `hidden_group = hidden.unsqueeze(-2).expand(..., G, ...)`
  - feature 후보: `group_feature_context`
    - group raw feature를 group별 projection으로 임베딩해서 생성

따라서 D/E에서도 `source`를 `hidden|feature|both`로 독립적으로 줄 수 있다.

## 7. Config/API 전환

구식 키는 active path에서 제거됐다.

- `stage_router_type`
- `stage_factored_group_router_source`
- `stage_factored_group_logit_scale`
- `stage_factored_intra_logit_scale`
- `stage_factored_combine_mode`

신규 키:

- `stage_router_wrapper`
  - stage별 wrapper 선택: `w1_flat|w2_a_plus_d|w3_bxc|w4_bxd|w5_exd|w6_bxd_plus_a`
- `stage_router_primitives`
  - stage별 primitive 설정 묶음
  - 예: `a_joint/b_group/c_shared/d_cond/e_scalar/wrapper(alpha_d,alpha_struct,alpha_a)`

예시 (W5 + D top1):

```yaml
stage_router_wrapper:
  mid: w5_exd
stage_router_primitives:
  mid:
    d_cond: {source: feature, temperature: 1.0, top_k: 1}
    e_scalar: {source: feature, temperature: 1.0, top_k: 0}
```

## 8. Logging / Diagnostics 정리

`router_aux`에 다음 정보가 stage별로 기록된다.

- `final_expert_logits`, `final_expert_probs`
- `group_probs`, `intra_probs`
- `wrapper_alias`, `wrapper_name`
- `primitive_outputs` (각 primitive의 logits/probs/source/temperature/top_k)
- `wrapper_internal` (wrapper 계수/내부 파라미터)

P8 저장 경로는 run-centric으로 고정:

- `.../logging/fmoe_n3/<dataset>/P8/<run_id>/diag/meta.json`
- `.../diag/tier_a_final/final_metrics.csv`
- `.../diag/tier_b_internal/internal_metrics.csv`
- `.../diag/tier_c_viz/*`
- `.../diag/raw/*` (best/test/raw trace)

호환용 key:

- `factored_group_logits`
  - wrapper가 group logits를 직접 내는 경우(B/E 기반) 채움
  - 기존 `factored_group_balance` 및 일부 진단 경로와의 호환 유지

### knn / group_knn / intra_group_knn

기존 핵심 consistency 지표는 최종 gate(`final_expert_probs`) 기준 흐름을 유지한다.

- `route_consistency_knn`: expert 전체 분포의 KNN-기반 JS
- `route_consistency_group_knn`: group 집계 분포 기준 JS
- `route_consistency_intra_group_knn`: 그룹별 intra 분포 기준 JS

즉 분석 스크립트의 핵심 의미는 바꾸지 않고, 라우터 내부 분해 정보만 보강한다.

## 9. Wrapper별 기대 역할

- `w1_flat`: 강한 flat baseline
- `w2_a_plus_d`: flat + intra residual 개선 검증
- `w3_bxc`: shared intra 가정의 타당성 검증
- `w4_bxd`: 정석 계층형 baseline
- `w5_exd`: group scalar 추정 방식의 대안 검증
- `w6_bxd_plus_a`: structured(BxD) + flat(A) residual correction

## 10. 코드 레벨 매핑 (핵심 진입점)

아래 파일/함수 기준으로 읽으면 구조가 빠르게 보인다.

- `experiments/models/FeaturedMoE_N3/router_wrapper.py`
  - Primitive 클래스: `StageJointExpertRouter`, `StageGroupRouter`, `StageSharedIntraRouter`, `GroupConditionalIntraRouter`, `GroupScalarRouter`
  - Wrapper 클래스: `FlatJointWrapper`, `FlatPlusGroupIntraResidualWrapper`, `GroupSharedIntraProductWrapper`, `GroupConditionalProductWrapper`, `ScalarGroupConditionalProductWrapper`, `GroupConditionalResidualJointWrapper`
  - helper: `_primitive_payload`, `_joint_prob_to_logit`, `normalize_wrapper_name`, `build_wrapper_module`, `required_primitives_for_wrapper`

- `experiments/models/FeaturedMoE_N3/stage_modules.py`
  - `StageRuntimeConfigN3`: 신규 router schema 필드 정의
  - `_build_primitive_specs`: primitive별 source/temperature/top_k 파싱
  - `_compose_input_from_source`: `both`일 때 concat 규칙
  - `_compute_router_outputs`: primitive 실행 -> wrapper 조합 -> final gate/aux 기록

- `experiments/models/FeaturedMoE_N3/stage_executor.py`
  - stage별 `stage_router_wrapper`, `stage_router_primitives`를 `N3StageBlock`으로 전달

- `experiments/models/FeaturedMoE_N3/featured_moe_n3.py`
  - `_parse_stage_nested_map`: stage별 primitive 설정 파싱
  - 모델 init에서 신규 schema 읽고 executor로 주입
  - loss 계산 시 `gate_weights`(최종 gate) 기준으로 balance/aux 적용

---

필요하면 다음 단계로 wrapper별 추천 초기 설정(temperature/top_k/source) 프리셋도 이 문서에 추가할 수 있다.
