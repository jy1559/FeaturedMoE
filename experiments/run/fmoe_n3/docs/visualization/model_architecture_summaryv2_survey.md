# FMoE_N3 다음 실험 요약 지시문

목표:
현재 코드베이스와 `model_architecture_summary.md`를 바탕으로, FMoE_N3의 다음 실험을
1) residual 재설계
2) dense vs sparse(top-k) routing 비교
3) standard vs factored / hidden vs both vs feature 비교
중심으로 정리하라.

핵심 배경:
- 현재 문서 기준 기본 anchor는 `moe + both + gated_bias` 쪽이다.
- `standard`와 `factored`는 둘 다 생존 가치가 있다.
- `aux/reg`는 강하게 넣기보다 약하게 넣는 편이 낫다.
- 가장 비어 있는 핵심 축은 `residual 설계`다.
- KuaiRec은 구조 차이 탐색용, lastfm은 일반화 확인용으로 본다.

## 1. 참고 논문에서 가져올 메시지

### Switch Transformer
- sparse MoE를 top-1 routing으로 단순화
- 핵심 장점은 정확도 자체보다 조건부 계산 효율
- 같은 자원에서 최대 7x pretraining speedup 보고
=> top-k는 필수 전제가 아니라 “효율/전문화” 검증 축으로 본다.

### ST-MoE
- sparse expert의 핵심 문제를 training instability로 봄
- sparse MoE를 더 안정적으로 쓰기 위한 design guide
=> sparse를 쓰려면 weak stabilization(z-loss류)와 routing diagnostics가 중요하다.

### DeepSpeed-MoE
- 공식 구현에서 residual MoE를 직접 지원
- dense fallback branch + routed expert branch를 함께 두는 방향
=> FMoE에서도 `identity residual + shared FFN + alpha * MoE`가 1순위 구조다.

### DeepSeekMoE
- expert segmentation + shared experts 분리
- 공통 지식은 shared experts가 맡고, routed experts는 specialization에 집중
=> FMoE에서도 shared/common branch와 specialized branch를 분리하는 것이 좋다.

### FAME (SeqRec)
- attention head/facet 단위로 MoE를 넣어 preference disentanglement
=> head-wise/facet-wise MoE는 후보 아이디어지만 지금은 후순위다.

### HyMoERec (SeqRec)
- shared + specialized branch + adaptive fusion
=> residual 재설계와 stage-wise alpha 실험의 직접 참고 사례다.

## 2. 현재 실험 우선순위

### 1순위: Residual redesign
기본 추천식:
- `u_s = LN(h_s)`
- `y_shared = SharedFFN_s(u_s)`
- `y_moe = MoE_s(u_s)`
- `out_s = h_s + y_shared + alpha_s(t) * y_moe`

규칙:
- 첫 라운드는 shared scale(beta)=1 고정
- `alpha_s(t)`만 조절
- `alpha_s(t) = warmup(t) * sigmoid(a_s)`
- `a_s`는 stage별 learnable scalar
- macro는 작게, micro는 크게 시작

실험:
- `R0 = h + MoE`
- `R1 = h + SharedFFN`
- `R2 = h + SharedFFN + 0.3*MoE`
- `R3 = h + SharedFFN + 0.5*MoE`
- `R4 = h + SharedFFN + alpha*MoE`
- `R5 = h + SharedFFN + alpha_stage*MoE`
- `R6 = h + SharedFFN + warmup(alpha_stage)*MoE`

### 2순위: Router family / source 정리
Residual winner를 고정한 뒤:
- `standard + gated_bias`
vs
- `factored + group_gated_bias`

그 다음 winner에 대해서만:
- `hidden`
- `both`
- `feature`

원칙:
- 기본값은 `both`
- `feature-only`는 제한적으로만 검증

### 3순위: Weak aux/reg
winner 구조에 대해서만:
- no z-loss
- weak z-loss
- weak z-loss + weak balance

원칙:
- 강한 aux/reg는 피할 것

## 3. Dense vs Sparse(top-k) 축 추가

핵심:
- top-k는 지금 단계의 필수 요소가 아니라 후반부 추가 검증 축이다.
- 먼저 dense 구조를 정리하고, 그 다음 sparse를 넣는다.

### A. Dense
- 모든 expert를 계산하고 gate로 가중합
- 안정적 baseline

### B. Global top-k
- 전체 expert 중 상위 k개만 선택
- 추천 실험:
  - `dense`
  - `global_top1`
  - `global_top2`

### C. Group-wise top-k
- factored/group 구조가 있을 때만 사용
- 각 group에서 상위 k개 선택
- 추천 실험:
  - `group_top1`
  - 필요 시 `group_top2`

원칙:
- 처음엔 count 기반만 (`k=1,2`)
- ratio 기반은 후순위
- 처음 sparse 실험에서는 capacity/drop token은 생략하고
  “선택 + renorm”만 구현해서 sparsity 자체 효과를 본다.

## 4. 실험 순서

1. Anchor 확정
- `moe + both + gated_bias` 계열 strongest config 하나 고정

2. Residual only ablation
- `R0~R6`

3. Router family 비교
- `standard+gated_bias` vs `factored+group_gated_bias`

4. Router source 비교
- `hidden / both / feature`

5. Weak aux/reg 비교
- z-loss 중심

6. Dense vs Sparse 비교
- `dense`
- `global_top1`
- `global_top2`
- factored winner면 `group_top1`
- 필요 시 `group_top2`

7. Parameter study
- len30
- family mask
- encoder mode
- macro window
- expert scale

8. Cross-dataset verification
- KuaiRec winner 2~3개만 lastfm으로 이식

## 5. 로깅 필수 항목

기존:
- `top1_max_frac`
- `n_eff`
- `jitter`

추가:
- stage별 expert utilization histogram
- stage별 routing entropy
- alpha_stage trajectory
- sparse일 경우 active expert count
- factored/group일 경우 group utilization histogram
- shared branch norm vs alpha*moe norm

## 6. 지금 하지 말 것

- feature-only 대규모 재탐색
- 강한 aux/reg 조합 반복
- per-head/facet MoE 조기 도입
- segmented expert 구조 대폭 확장
- sparse + capacity/drop/overflow를 한 번에 도입

## 7. 최종 질문

이번 라운드에서 답하고 싶은 질문은 아래 순서다.

1. shared FFN + MoE residual 분리가 실제로 유효한가?
2. standard와 factored 중 어느 쪽이 더 낫나?
3. hidden / both / feature 중 기본 source는 무엇인가?
4. weak z-loss가 안정성에 도움이 되는가?
5. dense routing이 충분한가?
6. sparse가 유효하다면 global top-k가 나은가, group-wise top-k가 나은가?