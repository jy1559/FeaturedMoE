# Phase 5 계획: 로깅 체계 개편 + 안정적 specialization 유도

## 0. 목적과 방향

Phase 4 결론을 기준으로 Phase 5의 우선순위를 다음처럼 재정의한다.

- 핵심 문제는 단순 imbalance가 아니라 routing jitter
- 균등 분배(load balancing) 자체가 목표가 아님
- 적당한 specialization은 성능에 유리할 수 있음
- 단, monopoly(한 expert 독식)로 무너지면 안 됨
- feature ablation은 현재 산출물 구조/품질이 불안정하므로 로깅 체계부터 정리 필요

따라서 Phase 5는 아래 3축으로 진행한다.

1. 결과 저장 구조를 run 중심으로 재설계
2. 로깅을 stable semantic specialization 관점으로 확장
3. aux/reg를 "specialization 강화 + anti-monopoly + jitter 억제" 중심으로 재실험

---

## 1. 문서/실험 축 정리

- 기존 이질 expert 중심 계획은 별도 문서로 분리
- 파일명: experiments/run/fmoe_n3/docs/heteroExpert_plan.md
- 본 문서(phase5_plan.md)는 현재 모델 구조를 크게 바꾸지 않고,
  라우팅 품질/해석 가능성을 높이는 실험에 집중

---

## 2. 결과 저장 구조 개편안

## 2.1 목표

지금의 normal/special/diag/sidecar 분산 구조를 완전히 버리기보다,
"run 단일 폴더"를 중심으로 통합 조회가 가능하도록 정리한다.

요구사항:
- phase/dataset 별 분리
- run마다 고유 폴더(앞에 날짜시간)
- 성능/특수지표/diag/ablation을 한 run 아래에서 즉시 찾기 가능
- 기존 스크립트/집계 로직과의 호환성 유지

## 2.2 권장 디렉토리

신규 루트:
- experiments/run/artifacts/results/logging

구조:

- experiments/run/artifacts/results/logging/fmoe_n3/{dataset}/{phase}/{run_id}/

run_id 규칙(권장):
- 0317_153200_r_shared_moe_warmup_c1
- 형식: MMDD_HHMMSS + 짧은 실험 slug
- 동일 초(second) 충돌 방지를 위해 마지막에 2~3자리 suffix 허용

run 폴더 내부(권장):

- run_meta.json
  - phase, dataset, model, seed, combo, config hash, git commit, 시작/종료 시각
- metrics_summary.json
  - best valid/test 핵심 지표(MRR@20, HR@10 등)
- metrics_epoch.csv
  - epoch 단위 학습/검증 추이
- special_metrics.json
  - session 길이, item popularity, cold/new-user 등 slice 결과
- router_diag.json
  - usage/top1/n_eff/cv/entropy/jitter/consistency/purity 요약
- router_diag_epoch.csv
  - epoch별 라우팅 지표 추이
- feature_ablation.json
  - 실제 수행된 경우만 생성, 미수행 시 status 필드 명시
- artifacts/
  - 필요 시 상세 dump, 예: route heatmap 원본, 샘플별 진단
- links.json
  - 기존 normal/special/diag/sidecar 원본 경로 참조(하위 호환)

예시:
- experiments/run/artifacts/results/logging/fmoe_n3/KuaiRecLargeStrictPosV2_0.2/P5/0317_153200_p5_m3_c2/

## 2.3 기존 구조와 호환 방식

기존 경로를 즉시 제거하지 않고 병행한다.

- 기존 출력(normal/special/diag/sidecar) 유지
- 각 run 종료 시 logging 폴더로 "통합 인덱스" 생성
- links.json에 원본 파일 절대/상대 경로를 기록

이 방식의 장점:
- 기존 파서/집계 스크립트 재사용 가능
- 신규 분석은 logging 트리에서 한 번에 처리 가능
- 점진 마이그레이션 가능

## 2.4 저장 성공 검증 체크

매 run 종료 후 자동 체크 항목:

1. run_meta.json 존재
2. metrics_summary.json 존재 및 핵심 4개 지표 존재
3. special_metrics.json 존재
4. router_diag.json 존재
5. feature_ablation status 명시(success/skipped/failed)
6. links.json이 기존 원본 경로를 가리키는지 검증

검증 실패 시:
- run_meta.json의 save_status = partial/fail 로 기록
- 콘솔 warning + sidecar summary에 경고 표시

---

## 3. 로깅 확장 설계 (Phase 5 핵심)

## 3.1 공통 원칙

- balanced 여부보다 specialization 품질을 측정
- 지표는 stage별(macro/mid/micro)로 분리 기록
- 평균값 1개로 끝내지 말고 분포(평균/표준편차/중앙값/P90) 저장
- 추세가 중요한 지표는 epoch 단위로 저장
- 추가 로깅(고비용 진단)은 매 epoch가 아니라 best valid 체크포인트 기준 1회만 수행

## 3.2 로깅 실행 시점 규칙 (중요)

Phase 5에서는 로깅을 2단계로 분리한다.

1) 경량 상시 로깅 (학습 중)
- metrics_epoch.csv
- 기본 route 통계(간단 usage/entropy/jitter 스칼라)

2) 고비용 추가 로깅 (run당 1회)
- 시점: 학습 종료 후 best valid pth 로드 직후
- 대상 split: valid + test
- 산출물: router_diag.json, special_metrics.json 확장본, feature-route consistency, purity, bucket heatmap 집계

즉, "추가 로깅은 run에서 best run의 best valid pth로 한 번만" 수행한다.

## 3.3 추가 로깅 항목

### A. Expert usage / assignment

필수:
- expert usage mass
- top-1 assignment fraction
- usage CV
- top-1 max fraction
- n_eff

의미:
- specialization 강도, monopoly 위험, 실질 활성 expert 수 파악

### B. Routing entropy

필수:
- entropy 평균/표준편차/중앙값
- stage별 entropy

의미:
- 과도한 평탄화(averaging) vs 과도한 샤프닝(collapse) 감시

### C. Session jitter / smoothness

필수:
- 인접 step 라우팅 변화량 평균
- stage별 jitter(macro/mid/micro)
- short/long session 분할 jitter

의미:
- Phase 4에서 가장 강한 음의 신호였던 jitter를 직접 관리

### D. Feature-route consistency

필수:
- feature 유사 샘플(kNN) 간 route JS divergence 평균
- stage별 consistency score

권장:
- 계산 비용 이슈 시 mini-batch 샘플링 근사치 사용

의미:
- semantic specialization(비슷한 feature면 비슷한 route) 정량화

### E. Feature bucket -> expert 경향

필수 집계:
- 대표 feature bucket별 expert 사용 heatmap용 값

권장 feature:
- macro: session gap, popularity 평균, category entropy, new user 여부
- mid: repeat ratio, switch ratio, session progress
- micro: interval std, short-term switch rate, window validity

의미:
- 어떤 regime를 어떤 expert가 담당하는지 직관적으로 확인

### F. Expert purity summary

필수:
- expert별 담당 샘플의 feature 통계 요약
- 예: repeat ratio 평균, new user 비율, cold item 비율 등

의미:
- expert 역할 설명 가능성 확보

### G. Slice 성능 + 라우팅 동반 지표

slice(최소 세트):
- new user vs old user
- cold item vs warm item
- short session vs long session

확장 세트:
- high repeat vs high switch
- low validity micro vs high validity micro

동시 기록:
- MRR/HR + jitter + usage CV + top1 max

의미:
- 성능 개선이 어느 regime에서 발생했는지, 왜 발생했는지 연결 가능

---

## 4. Aux/Reg 실험 계획 (specialization 중심)

## 4.1 실험 철학

- load balancing 강제는 주 전략이 아님
- 목표: stable semantic specialization
- 보조 목표: anti-monopoly, jitter 감소

## 4.2 방법 축 (5개)

Method 0. Baseline
- 기존 best 설정(또는 현재 기본 aux/reg) 고정

Method 1. Smoothness regularization
- session 내 인접 step route 변화 페널티
- 가중치 권장: macro > mid >> micro

Method 2. Feature-consistency regularization
- feature 근접 샘플의 route 분포 유사도 유도
- JS/KL 기반 거리 최소화

Method 3. Sharp-but-not-monopoly
- sample 단위 entropy를 약하게 낮춰 specialization 유도
- expert 독식 구간(top1 max 임계 초과)만 페널티

Method 4. Soft prior
- feature regime별 선호 expert prior를 logit bias로 약하게 주입
- 라우터가 prior를 참고하되 override 가능하게 설계

## 4.3 코드 레벨 설계 (aux/reg 추가 방식)

적용 포인트는 크게 2곳이다.

1) 모델 forward 쪽
- 파일: experiments/models/FeaturedMoE_N3/stage_executor.py, experiments/models/FeaturedMoE_N3/featured_moe_n3.py
- 역할: stage별 router prob/logit 및 aux 통계(route prob, entropy, top1, session index)를 aux dict로 반환

2) 학습 루프 쪽
- 파일: experiments/recbole_train.py
- 역할: 기존 main loss에 method별 aux/reg 항을 합산

권장 인터페이스:

```python
# model forward 결과
output = {
  "loss_main": loss_main,
  "router_aux": {
    "macro": {"prob": p_macro, "logits": z_macro, "top1": t_macro, "session_id": sid},
    "mid":   {"prob": p_mid,   "logits": z_mid,   "top1": t_mid,   "session_id": sid},
    "micro": {"prob": p_micro, "logits": z_micro, "top1": t_micro, "session_id": sid},
  }
}
```

```python
# trainer 쪽 (개념)
loss = loss_main
if method == "smoothness":
  loss += lam_s * L_smooth(router_aux)
elif method == "consistency":
  loss += lam_c * L_consistency(router_aux, feature_embed)
elif method == "sharp_mono":
  loss += lam_e * L_entropy(router_aux) + lam_m * L_monopoly(router_aux)
elif method == "soft_prior":
  loss += lam_p * L_prior(router_aux, feature_prior)
```

method별 수식(권장):

1) Smoothness

$$
L_{smooth} = \sum_{s \in \{macro,mid,micro\}} w_s \cdot
\mathbb{E}_{t>1}\left[\|p_{s,t}-p_{s,t-1}\|_1\right]
$$

- 권장 stage weight: $w_{macro}=1.0, w_{mid}=0.5, w_{micro}=0.1$

2) Feature-consistency

$$
L_{cons} = \mathbb{E}_{(i,j)\in\mathcal{N}_k} \left[ JS(p_i, p_j) \right]
$$

- $\mathcal{N}_k$: feature embedding 기준 kNN pair
- 계산량 절감을 위해 미니배치 내부 샘플링 pair 사용

3) Sharp-but-not-monopoly

$$
L_{sharp} = \mathbb{E}[H(p)] \quad (H \downarrow)
$$

$$
L_{mono} = \sum_e \max(0, u_e - \tau)^2
$$

- $u_e$: expert e의 top1 usage fraction
- $\tau$: monopoly 임계(예: 전문가 수 K일 때 $\tau=\min(0.45, 2.5/K)$)

4) Soft prior

$$
  ilde{z}=z + \alpha b(feature), \quad p=softmax(\tilde{z})
$$

$$
L_{prior}=KL\left(p \;||\; softmax(b(feature))\right)
$$

- $\alpha$는 prior bias scale

최종 loss:

$$
L = L_{main} + \lambda_1 L_{smooth} + \lambda_2 L_{cons} +
\lambda_3 L_{sharp} + \lambda_4 L_{mono} + \lambda_5 L_{prior}
$$

단, method별로 필요한 항만 활성화한다.

## 4.4 Combo 설계 (8개, baseline 중심 변형 + outlier 포함)

고정 2x2x2 완전요인 대신, baseline 1개에서 실전적으로 흔히 바꾸는 축을 섞어 8개를 구성한다.

Baseline anchor:
- C0: 현재 best 설정(Phase 4 base 재현)

주요 변형군:
- C1: low temperature only
- C2: fewer experts only
- C3: shared fallback on only
- C4: router hidden dim 축소 (예: d_router_hidden 64 -> 32)
- C5: router hidden dim 확장 (예: 64 -> 128)
- C6: stage MLP layer depth +1 (mid/micro만)

Outlier:
- C7: 공격적 설정(예: low temp + fewer experts + fallback off + dim 32)

의도:
- C1~C3은 기존 축과 비교 가능성 확보
- C4~C6은 dim/구조 변화 효과 확인
- C7은 경계조건에서 method 강건성 확인

## 4.5 전체 매트릭스

- 5 methods x 8 combos = 40 runs

장점:
- method 효과와 router 조건 효과를 분리 분석 가능
- 어느 조건에서 어떤 reg가 잘 듣는지 확인 가능

## 4.6 튜닝 폭 제한

조합 폭발 방지 원칙:

- Baseline: 튜닝 없음
- Smoothness: lambda_s 1개만 개방
- Consistency: lambda_c 1개만 개방
- Sharp+Monopoly: lambda_e 1개 + tau 규칙 고정
- Soft prior: alpha 또는 lambda_p 중 1개만 개방

권장 시작값:
- lambda_s=0.02, lambda_c=0.02, lambda_e=0.005, lambda_m=0.02, lambda_p=0.01

---

## 5. 실행 순서

Step 1. Logging sanity run (1~2개)
- 저장 구조/지표 계산 정상 여부 확인
- best valid pth 로드 후 1회 추가로깅 동작 확인
- feature_ablation status 기록 확인

Step 2. Baseline method로 8 combos 선실행
- 조합별 원래 라우팅 성향(기저 jitter, monopoly 취약성) 파악

Step 3. Full 40 runs
- 5 methods x 8 combos 실행

Step 4. 상위 후보 seed 재검증
- 상위 3~5 조합에 대해 seed 2~3개 추가

Step 5. Phase 5 결과 문서화
- 성능 + 라우팅 품질 + 해석 가능성 기준으로 최종 추천

---

## 6. 성공 기준 (Go/No-Go)

정량 기준(권장):

1. test MRR@20
- 기준: Phase 4 base(0.1622) 대비 동급 이상, 가능하면 +0.0003 이상

2. jitter
- 기준: baseline 대비 유의미 감소(특히 macro/mid)

3. specialization 품질
- usage CV는 0으로 수렴하지 않음(적당한 불균형 허용)
- top1 max fraction이 collapse 임계 초과하지 않음

4. semantic alignment
- feature-route consistency 개선
- expert purity 해석 가능(역할 분화가 설명됨)

5. slice robustness
- cold/new-user/short-session에서 성능 또는 안정성 개선

Go 조건:
- 성능 동급 이상 + jitter/consistency 중 최소 1개 명확 개선

No-Go 조건:
- 성능 하락 + monopoly 증가 + consistency 개선 없음

---

## 7. feature ablation 정비안

현재 문제:
- 미수행/실패 여부가 결과상 불명확
- 비교 단위가 통일되지 않음

정비:

- feature_ablation.json에 반드시 status 필드
  - success | skipped | failed
- success일 때만 delta 지표 기록
  - delta_valid_mrr20, delta_test_mrr20 등
- 비교 기준 명시
  - baseline_run_id
- 최소 ablation 세트 고정
  - hidden-only
  - feature-only
  - full

이렇게 하면 "안 된 실험"과 "효과 없는 실험"을 구분 가능

---

## 8. 코드베이스 적용 포인트 (구현 가이드)

아래 파일/모듈 영역에서 반영하는 것이 자연스럽다.

- experiments/recbole_train.py
  - run_id 생성(MMDD_HHMMSS slug), logging 루트 경로 결정
  - best model 로드 직후 1회 추가로깅 호출
  - method별 aux/reg loss 조합 적용
- experiments/hydra_utils.py
  - logging/fmoe_n3/{dataset}/{phase}/{run_id} 템플릿 주입
- experiments/models/FeaturedMoE/run_logger.py
  - run_dir 경로를 model/dataset/phase/run_id 구조로 확장
  - 경량 상시 로그 + best-valid 1회 상세 로그 파일 분리

- experiments/run/artifacts/results/fmoe_n3/diag 관련 저장부
  - router_diag.json, router_diag_epoch.csv 생성 확장
- special metrics 생성부
  - slice 확장(new-user/cold 등) 및 동반 라우팅지표 저장
- sidecar 번들 생성부
  - links.json, run_meta.json 참조 연결

주의:
- 기존 normal/special/diag/sidecar 파이프라인은 유지
- 우선은 logging 통합 레이어만 추가하고, 이후 점진 통합

---

## 9. 리스크와 완화

리스크 1. 로깅 과다로 I/O 증가
- 완화: per-step 저장 금지, epoch 단위/요약 통계 중심 저장

리스크 2. consistency 계산 비용
- 완화: 샘플링 기반 근사(kNN on subset)

리스크 3. aux/reg가 과도해 성능 저하
- 완화: 각 method 튜닝 변수 1개만 개방, baseline 대비 early stop

리스크 4. 결과 해석 복잡도 증가
- 완화: 공통 스코어카드(성능/안정성/전문화/해석성) 4축으로 단순화

---

## 10. 최종 산출물

Phase 5 종료 시 최소 산출물:

1. experiments/run/fmoe_n3/docs/phase5_results.md
- 40 runs 요약표 + 상위 후보 비교

2. experiments/run/fmoe_n3/docs/phase5_logging_spec.md
- 실제 반영된 로깅 스키마 최종본

3. logging 트리 샘플 1개
- run 폴더 하나를 골든 샘플로 지정
- 팀 내 후속 실험 템플릿으로 재사용

핵심 목표 문장:
- Phase 5는 balanced routing을 만들기 위한 단계가 아니라,
  성능으로 이어지는 stable semantic specialization을 관측/유도/검증하는 단계다.
