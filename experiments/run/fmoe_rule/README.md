# FMoE Rule Router Ablation Entrypoints

`run/fmoe_rule`는 rule-based router ablation 전용 트랙입니다.

## 스크립트
- `train_single.sh`: 단일 실험 실행 (B0/B1/R0/R1)
- `pipeline_ml1_rr_rule.sh`: ML1M -> RR 순서로 ablation 매트릭스 실행

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
```
