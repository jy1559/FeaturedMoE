# P2/P3/P6 확장 분석 요약 (KuaiRecLargeStrictPosV2_0.2)

## 1) Phase2 input 중심 (붕괴 제외)
- 필터: best_valid_mrr20 >= 0.075
- 표본 수: 39
- 주요 상관(Spearman):
  - group_prior_align_lambda: +0.2196
  - factored_group_balance_lambda: -0.2874
  - balance_loss_lambda, z_loss_lambda: 거의 0 근처
- 해석:
  - P2 구간에서는 group prior 정렬 항을 키우면 valid MRR이 소폭 개선되는 경향이 보임.
  - 반대로 factored group balance를 강하게 주면 valid MRR이 하락하는 구간이 관측됨.
  - balance/z는 이 구간 탐색 범위에서 성능 분산을 크게 설명하지 못함.

## 2) Phase3 구조 비교 (요청 3종)
- standard + all gated (P3S1): avg valid 0.08106, avg test 0.16148
- factored + group gated (P3S2): avg valid 0.08102, avg test 0.16194
- feature-source routing (P3S3): avg valid 0.08090, avg test 0.16210

- feature-source routing 의미:
  - 라우터 입력을 hidden 단독에서 확장해 feature source 신호(주입된 feature family 정보)에 직접 반응하도록 만든 설정.
  - 즉, 라우팅 결정이 "표현 벡터 크기"보다 "어떤 feature family가 활성인지"를 더 직접적으로 반영하도록 유도하는 방식.

## 3) Phase6 router2x2 (X1/X2, 4조합)
- 비교축: router type (standard vs factored) x injection (gated_bias vs group_gated_bias)
- 최고 valid:
  - P6_RXI_X1_STA_GAT: valid 0.0812, test 0.1620
- X1 vs X2 경향:
  - X1 계열이 X2 대비 valid/test가 전반적으로 높음.
  - X2는 macro_n_eff_scope_norm이 크게 낮아지고(top-k 조건 포함), top1_over_uniform이 커지며 라우팅 집중이 강해짐.

## 4) Phase6 specialization 입력 영향 (spec_ablation)
- 입력군: route_smoothness, route_consistency, route_sharpness, route_monopoly, route_prior
- 주요 결과:
  - route_prior_lambda -> test_mrr20: Spearman -0.7027 (강한 음의 상관)
  - route_consistency_lambda -> macro_knn_js: Spearman -0.7707
  - route_smoothness_lambda -> test_mrr20: Spearman -0.5927
  - sharp/monopoly는 표본 수가 4로 작아 해석 신뢰도 제한

- 해석:
  - spec 입력을 강하게 주면(특히 prior/consistency 조합) 라우팅 유사도 계열 지표는 안정화될 수 있으나,
    test MRR 저하가 함께 발생하는 구간이 존재.
  - 즉, specialization regularization은 "분화 품질"과 "최종 추천 성능" 간 trade-off를 동반.

## 5) SASRec baseline 맥락
- 본 문서는 FMoE 내부 구조 비교에 초점.
- 최종 성능 주장에는 baseline 요약(특히 SASRec 대비 delta)을 반드시 병기 권장.
