# FeaturedMoE_N Roadmap Next

## 1. Principle

- `FeaturedMoE_N`은 당분간 `simple_flat` 메인라인으로 간다.
- 우선순위는 `loss/aux -> logging summary -> router structure -> expert 다양화` 순서다.
- `P0/P0.5`에서는 새 loss를 넣지 않는다.


## 2. P1: Existing Knob Re-check Only

구조 변경 없이 현재 이미 있는 knob만 본다.

우선순위:

1. `balance_loss_lambda` 재조정
2. `rule_bias_scale` 미세조정
3. `plain vs hybrid vs bias` 재현성 확인
4. `linear vs sinusoidal_selected` 재확인
5. `expert_scale`와 `moe_top_k`의 좁은 재검증

P1에서 보는 것:

- overall `MRR@20`
- `special_metrics.json` slice stability
- Kuai / lastfm 간 방향성 일치 여부

P1에서 안 하는 것:

- 새 aux 추가
- router 구조 변경
- teacher / distillation 복귀


## 3. P2 Candidates: Additive Loss / Aux

지금 넣지는 않지만, 넣을만한 후보를 미리 정리한다.

### 3.1 `router z-loss`

- 왜 지금 제외:
  - 현재 baseline 자체 ranking이 아직 정리되지 않았다.
  - router logit scale 문제인지, 단순 LR / balance issue인지 먼저 분리해야 한다.
- 승급 조건:
  - `plain/hybrid/bias` 모두 expert usage가 한두 expert로 과집중
  - same layout에서 LR tuning으로도 usage collapse가 반복
- 예상 리스크:
  - 과도하면 gate confidence를 눌러서 overall이 같이 떨어질 수 있음

### 3.2 early-only `gate entropy regularization`

- 왜 지금 제외:
  - 짧은 `P0` 예산에서는 entropy warmup과 LR 효과가 섞인다.
- 승급 조건:
  - 초반 expert collapse 후 후반 복구가 거의 안 보일 때
  - `special_metrics`에서 특정 slice만 극단적으로 약해질 때
- 예상 리스크:
  - entropy를 오래 주면 routing sharpness가 부족해질 수 있음

### 3.3 `rule agreement penalty`

- 왜 지금 제외:
  - `plain/hybrid/bias` 세 family의 baseline 방향을 먼저 확인해야 한다.
- 승급 조건:
  - hybrid / bias가 slice는 좋지만 overall이 흔들릴 때
  - learned logits와 rule prior가 계속 반대로 가는 패턴이 보일 때
- 예상 리스크:
  - prior를 너무 강하게 잠가 learned router 이득을 죽일 수 있음

### 3.4 `group coverage regularization`

- 왜 지금 제외:
  - 현재 router가 `simple_flat`이므로 먼저 plain collapse 여부부터 봐야 한다.
- 승급 조건:
  - 특정 group/expert family가 거의 쓰이지 않고, long-tail slice가 같이 약할 때
- 예상 리스크:
  - coverage 자체를 맞추려다 필요한 specialization까지 희석될 수 있음


## 4. Logging Roadmap

- 기본 단위는 epoch가 아니라 run-group summary다.
- `special_metrics.json`에서 아래를 phase별로 묶는다.
  - overall
  - popularity bucket
  - short-session bucket
  - new-user bucket
- 나중에 router 통계를 추가하면 같은 JSON에 `router_summary`를 붙이는 방향으로 간다.
- 후보 컬럼:
  - `stage_entropy`
  - `top1_share`
  - `active_expert_count`
  - `group_usage`


## 5. Router Structure: Deferred Until After Loss/Aux

우선순위:

1. `group_local`
2. `hierarchical`
3. expert family 다양화

이 순서로 미루는 이유:

- 기존 결과상 복잡한 router를 먼저 열수록 tuning noise가 커졌다.
- 지금은 `simple_flat`에서 loss/aux로 얼마나 개선되는지 먼저 보는 편이 싸다.
- `group_local`이 첫 구조 후보인 이유는 의미 보존 대비 구현비가 가장 낮기 때문이다.


## 6. Explicit Defer List

당장 제외:

- `InfoNCE` main loss replacement
- sampled softmax
- gBCE replacement
- full teacher/distillation 재확장
- hierarchical router first
- expert family heterogeneity first

제외 이유:

- baseline ranking과 slice story가 먼저 정리되지 않았다.
- 현재 단계에서 변수 수만 늘리고 attribution은 더 어려워진다.


## 7. Decision Rule After P0/P0.5

- `P1`로 넘어가는 조건은 P0/P0.5에서 anchor combo가 정리되는 것이다.
- anchor 판정 기준:
  - overall `MRR@20`
  - Kuai / lastfm 방향성
  - `special_metrics`에서 과한 붕괴가 없는지
- anchor가 정리되면 `P1`은 loss/aux 없이 existing knob 재정리부터 시작한다.
