# FeaturedMoE_N Roadmap Next 2026-03-12

## 1. Current Read

- `FeaturedMoE_N`는 아직 `SASRec` 계열 baseline을 넘는 상태가 아니다.
- 현재 Kuai 기준으로:
  - baseline P0 top은 `best MRR@20 ~= 0.0234`, `test MRR@20 ~= 0.0223`
  - `fmoe_n` S00 P0 top은 `best MRR@20 ~= 0.0178`, `test MRR@20 ~= 0.0167`
  - `fmoe_n` S01 ARCH1 top cluster도 대체로 `best 0.0180~0.0183`, `test 0.0163~0.0170` 수준이다
- 즉 지금은 "조금 더 tuning하면 이긴다"보다, `backbone / MoE / router` 중 어디가 병목인지 먼저 쪼개는 단계에 가깝다.

## 2. What Prior Tracks Actually Say

### 2.1 `fmoe_v2`

- 강한 기준선은 여전히 `simple serial`이었다.
- ML1 top은 `serial + L7`에서 `MRR@20 ~= 0.0982`까지 갔다.
- RR도 `serial + L18/L15/L7` 계열이 `0.260+`를 찍었다.
- 해석:
  - 복잡한 구조를 열기 전에, 강한 serial anchor 자체는 늘 다시 확인할 가치가 있다.

### 2.2 `fmoe_rule`

- `rule_soft` hybrid는 분명히 먹힌 적이 있다.
- ML1에서 `serial + L7 + mid/micro=rule_soft`가 `MRR@20 ~= 0.0988`로 매우 강했다.
- 해석:
  - 지금 `N`에서 hybrid가 약하다고 해서 `rule-based` 자체가 별로라고 결론 내리면 안 된다.
  - 현재 약한 것은 `rule 자체`보다 `N backbone / layout / bridge style`과의 조합 문제일 가능성이 크다.

### 2.3 `fmoe_v4_distillation`

- `rule_hybrid_soft`가 ML1에서 `0.0980`으로 top이었다.
- plain도 나쁘지 않았고, hybrid가 항상 압도한 것은 아니었다.
- 해석:
  - `plain`과 `hybrid`를 둘 다 살아 있는 가설로 둬야 한다.
  - 한쪽 몰빵은 좋지 않다.

### 2.4 `fmoe_v3`

- router structure 변경으로 약간의 개선은 있었지만, 혁신적이지는 않았다.
- 해석:
  - router를 아예 손대지 말자는 뜻은 아니다.
  - 다만 지금 시점에 `hierarchical` 같은 큰 구조부터 열면 변수가 너무 많아진다.

### 2.5 `fmoe_hgr` / `fmoe_hgr_v4`

- `group-heavy` 계열은 대체로 classic v2 / rule / v4_distill anchor보다 약했다.
- ML1 기준 상위권이 대체로 `0.095~0.096`대였다.
- 해석:
  - `group_local`이나 grouped router는 볼 가치가 있지만, 첫 대형 구조 변경으로 가기에는 리스크가 크다.

## 3. Current `fmoe_n` Signal

현재 `S01` 쪽에서 읽히는 건 대략 이렇다.

- light layout만 많이 본다고 성능이 올라오지는 않았다.
- `ARCH1`에서 상대적으로 안정적이었던 건:
  - `A05`
  - `A06`
  - `A11`
  - `A13`
- 이들은 대체로 `lr ~= 3.0e-4 ~ 3.7e-4` 근처에서 나왔다.
- 반면 `A18` 같은 경우는 valid는 나쁘지 않았지만 test에서 더 크게 빠졌다.
- special logging 기준으로도 상위 combo들이 공통적으로:
  - `<=5 pop`은 overall보다 상대적으로 덜 나쁘거나 오히려 강한 편
  - `3-5 session`도 크게 망하지는 않음
  - `new_user`는 전반적으로 약함
- 즉 현재 문제는 특정 tail slice만의 붕괴라기보다, 전체적인 backbone-quality 부족 쪽에 더 가깝다.

## 4. Main Interpretation

지금 가장 중요한 건 아래 세 가지를 분리하는 것이다.

1. `N backbone` 자체가 약한가?
2. backbone은 괜찮은데 `MoE`가 오히려 해를 끼치고 있는가?
3. MoE는 살릴 수 있는데 `router`가 현재 layout과 안 맞는가?

현재는 1, 2, 3이 섞여 있어서 다음 tuning이 계속 애매해질 가능성이 높다.

그래서 다음 단계는 "조금 더 layout sweep"보다 `diagnostic control`을 강하게 넣는 쪽이 낫다.

## 5. Recommended Priority Order

### 5.1 Priority A: Diagnostic Controls First

이게 최우선이다.

#### A1. `MoE-off` control

- 같은 `FeaturedMoE_N` executor / feature path / layout 체인을 유지한다.
- 대신 각 `moe_block`을 아래 같은 control로 바꿔 본다.
  - dense FFN
  - simple nonlinear residual block
  - 거의 identity에 가까운 thin adapter
- 목적:
  - "중간의 SASRec-like attention layer가 실제로 backbone으로 일하고 있는가?"
  - "지금 gap은 router 문제가 아니라 MoE 자체 때문인가?"
- 기대 해석:
  - `MoE-off >= current N`이면 지금 MoE가 득보다 실이 크다.
  - `MoE-off << current N`이면 backbone보다 MoE가 실제로 일을 하고 있는 것이다.

#### A2. Pure-attention / transformer-like control

- `feature bank`와 `N`의 입력 표현은 유지하되, stage tail을 pure attention 쪽으로 둔다.
- 목적:
  - 현재 `N`의 input/feature path가 적어도 transformer backbone 위에서는 경쟁력이 있는지 확인
  - `SASRec` 대비 어디서 손해가 나는지 분리

#### A3. Full rule-based branch

- 지금은 주로 `plain`, `bias`, `hybrid(mid/micro)`를 보고 있다.
- 다음에는 `full rule-based`도 분기해야 한다.
- 우선순위:
  - all eligible stage `rule_soft`
  - 최소한 `mid/micro` 완전 rule
  - 필요하면 `macro learned + mid/micro full rule`도 유지
- 이유:
  - `fmoe_rule`이 강했던 만큼, 현재 N에서 hybrid가 약하다고 rule 계열을 버리면 안 된다.

#### A4. Strong classic anchor inside `N`

- `N` 안에서도 `classic serial` anchor를 다시 세워야 한다.
- 의미:
  - 더 가벼운 layout만 보는 게 아니라
  - `L7/L16/L18` 같은 강한 old-family skeleton을 `N` 문맥 안에서 다시 보는 것
- 목적:
  - "N의 feature path는 좋은데 new light layout만 약한가?"
  - "아니면 N 전체가 약한가?"

### 5.2 Priority B: Additive Loss / Aux / Regularizer

control에서 backbone viability가 확인되면 그 다음은 이쪽이다.

우선순위:

1. `balance_loss_lambda` 재정리
2. `router z-loss`
3. early-only `gate entropy regularization`
4. `rule agreement penalty`
5. `group coverage regularization`

해석은 이렇게 잡는 편이 낫다.

- `z-loss`:
  - router logits scale이 커져 expert collapse가 반복될 때
- early-only entropy:
  - 초반 collapse 이후 복구가 안 될 때
- rule agreement:
  - hybrid / bias가 slice는 좋은데 overall이 흔들릴 때
- coverage:
  - 특정 expert/group만 과도하게 쓰일 때

중요한 점:

- 지금 단계에서 `InfoNCE`, sampled softmax, `gBCE` main-loss replacement까지 가는 건 너무 멀다.
- 먼저 `CE + additive regularizer` 범위 안에서 정리하는 게 낫다.

### 5.3 Priority C: Router Changes, But Narrowly

router를 더 바꿔야 한다는 방향성 자체는 맞다.
다만 범위를 좁혀야 한다.

추천 순서:

1. `group_local-lite`
2. stage-wise grouped prior / grouped bias
3. stage-wise grouped learned router
4. `hierarchical` router는 더 뒤

이유:

- `fmoe_v3`는 router redesign이 아주 약간 먹혔지만 압도적이진 않았다.
- `fmoe_hgr*`는 group-heavy 쪽이 크게 성공적이지 않았다.
- 그래서 지금은 `meaning-preserving`에 가까운 small router change부터 보는 게 맞다.

## 6. What I Would Not Do First

아래는 당장 1순위로 두지 않는 편이 좋다.

- light layout만 더 많이 추가하기
- `hierarchical`부터 크게 여는 것
- teacher/distillation 복귀
- expert family heterogeneity부터 여는 것
- main loss 교체

이유는 단순하다.

- 지금은 `backbone / MoE / router` 분해가 먼저다.
- 이걸 안 하고 변수를 더 늘리면 attribution이 더 어려워진다.

## 7. Suggested Next Experiment Families

이번 문서는 `.sh`까지 만들지 않지만, 다음 세 family를 기준으로 생각하면 된다.

### Family 1. Backbone / No-MoE Control

- 같은 layout에서 `MoE -> dense FFN`
- 같은 layout에서 `MoE -> nonlinear bridge only`
- same feature path + pure attention tail
- classic serial heavy anchor in `N`

목적:

- "지금 못 나오는 이유가 MoE인가, backbone인가?"를 먼저 자른다.

### Family 2. Rule Truth Table

- `plain`
- `bias`
- `hybrid(mid only)`
- `hybrid(mid+micro)`
- `full rule`

단, layout은 많이 늘리지 말고 1~2개 strong anchor에서만 본다.

목적:

- "rule family가 진짜 죽었는가, 아니면 current layout과만 안 맞는가?"를 확인

### Family 3. Aux / Regularizer

- `balance_loss_lambda`
- `z-loss`
- early-only entropy
- rule agreement

단, 이 family는 Family 1에서 backbone viability가 확인된 뒤에 여는 게 낫다.

## 8. Concrete Decision Rule

다음 단계에서 아래처럼 해석하면 된다.

### Case A. `MoE-off`가 현재 `N`보다 좋다

- 현재 priority는 router가 아니다.
- 먼저 `MoE block` 자체나 stage composition을 다시 봐야 한다.
- 이 경우 loss/aux보다 `MoE design simplification`이 먼저다.

### Case B. `MoE-off`는 약하지만 `plain`만 괜찮다

- router complexity가 과한 것이다.
- `plain + small bias` 쪽으로 재정리하고, hybrid/full-rule은 strong anchor에서만 본다.

### Case C. `full rule`이나 `hybrid`가 strong anchor에서 다시 살아난다

- 현재 S01 약세는 `rule` 문제가 아니라 `layout/bridge mismatch`로 본다.
- 이 경우 P2는 rule-aware aux로 가는 게 맞다.

### Case D. control도 약하고 MoE도 약하다

- `N` backbone 자체를 재설계해야 한다.
- 즉 "feature tail on top of stronger SASRec backbone" 방향으로 더 가까이 가야 한다.

## 9. Bottom Line

- 지금은 `FeaturedMoE_N`에 뭔가를 많이 더하는 것보다, `control 실험`을 더 공격적으로 넣는 게 맞다.
- 특히 아래 둘은 거의 필수다.
  - `MoE-off`
  - `full rule-based`
- 그리고 다음 loss/aux는 여전히 볼 만하다.
  - `router z-loss`
  - early-only entropy
  - `rule agreement`
  - `group coverage`
- 다만 그 전에 먼저:
  - backbone이 되는지
  - MoE가 해를 끼치는지
  - hybrid가 현재 layout과만 안 맞는지
  를 분리해야 한다.

