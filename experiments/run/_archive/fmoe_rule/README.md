# FMoE Rule Router Ablation Entrypoints

`run/fmoe_rule`는 rule-based router ablation 전용 트랙입니다.

## 스크립트
- `train_single.sh`: 단일 실험 실행 (B0/B1/R0/R1)
- `tune_hparam.sh`: rule ablation 고정 combo에서 LR/WD quick tuning
- `pipeline_ml1_rr_rule.sh`: ML1M -> RR 순서로 ablation 매트릭스 실행
- `rr_rule_quick_tune.sh`: RR 전용 quick rule probe. ML1 rule-hybrid 상위권과 RR v2 상위 layout/dim 교집합만 빠르게 확인

## Ablation Arms
- `B0`: learned router (hidden+feature)
- `B1`: learned router (feature-only; hidden off)
- `R0`: rule_soft router (macro+mid+micro all rule)
- `R1`: mixed router (macro learned, mid/micro rule)

## 출력 경로
- Logs: `experiments/run/artifacts/logs/fmoe_rule/*`
- Results: `experiments/run/artifacts/results/fmoe_rule/*.json`

## 예시
```bash
bash experiments/run/fmoe_rule/train_single.sh \
  --dataset movielens1m --ablation R0 --layout-id 0 --execution serial --gpu 0 --dry-run

bash experiments/run/fmoe_rule/pipeline_ml1_rr_rule.sh \
  --datasets movielens1m,retail_rocket --gpus 0,1 --arms B0,B1,R0,R1 --dry-run

bash experiments/run/fmoe_rule/rr_rule_quick_tune.sh \
  --datasets retail_rocket --gpus 0,1,2,3 --dry-run
```

## RR Quick Rule Probe
- 기본 프로필 `rr_rule8`
- 중심 가설:
  - ML1 rule 상위권은 `R1` hybrid, `lr ~ 3e-4..1.2e-3`, `wd ~ 0..1.4e-4`
  - RR v2 상위권은 `L16/L15/L18 + moderate dim`, `lr ~ 3e-4..8e-4`
- 따라서 기본 probe는:
  - `R1`을 메인으로 `L16/L15/L18`와 `128/F24/H160/R64`, `160/F16/H160/R80`, `160/F24/H192/R96` 주변만 확인
  - `R0` pure-rule은 `L16` anchor 두 개만 sentinel로 확인
