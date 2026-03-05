# FeaturedMoE

3-stage hierarchical Mixture-of-Experts sequential recommender.

RecBole `SequentialRecommender` 기반으로,
- item sequence representation (Transformer)
- engineered session features (Macro/Mid/Micro)
를 분리 처리한 뒤 MoE로 결합해 next-item을 예측합니다.

---

## 문서

- `quick_guide.md`: 구조 개요 + 고영향 하이퍼파라미터 요약
- `deep_dive.md`: 코드 단위 동작 순서 + schedule/loss 상세 + 실패 패턴

---

## 빠른 실행

```bash
cd experiments

# 1) metric-only (권장 기본)
python recbole_train.py model=featured_moe dataset=movielens1m \
  feature_mode=full fmoe_debug_logging=false epochs=5

# 2) debug-full (expert/bucket/perf 로깅 포함)
python recbole_train.py model=featured_moe dataset=movielens1m \
  feature_mode=full fmoe_debug_logging=true epochs=5

# 3) sparse gating (top-k)
python recbole_train.py model=featured_moe dataset=amazon_beauty \
  feature_mode=full moe_top_k=2 fmoe_debug_logging=true epochs=5

```

---

## 출력 디렉토리

### `fmoe_debug_logging=false`

```text
outputs/FMoE/<run_name>/
  config.json
  epoch_metrics.csv
  summary.json
```

### `fmoe_debug_logging=true`

```text
outputs/FMoE/<run_name>/
  config.json
  epoch_metrics.csv
  expert_weights.csv
  expert_performance.csv
  feature_bias.csv
  summary.json
```

---

## 기본 해석 원칙

- FeaturedMoE 해석은 hard routing보다 **weight 기반 해석**을 우선합니다.
- `feature_bias.csv`는 ratio bucket(`R0-20`, `R20-40`, ...) 기준입니다.
- `count=0` bucket row는 저장되지 않습니다.

---

## 로그 시각화

```bash
cd experiments
python outputs/FMoE/visualize_fmoe_logs.py --latest
```

또는 특정 run:

```bash
python outputs/FMoE/visualize_fmoe_logs.py --run-name <run_name>
```

시각화 결과는 `outputs/FMoE/<run_name>/viz/`에 저장됩니다.
기본 동작은 `viz/` 내 기존 PNG/CSV를 정리한 뒤 새 결과만 저장합니다.
(`--no-clean` 옵션으로 유지 가능)

기본 산출물(통합형):
- `training_overview.png`
- `routing_dashboard.png`
- `feature_expert_dashboard.png`
- `bucket_performance_dashboard.png`
- `feature_expert_summary.csv`
- `feature_bucket_summary.csv`

---

## FeaturedMoE 전용 HyperOpt

```bash
cd experiments
bash run/fmoe/tune_hparam.sh --dataset movielens1m --layout_id 7 --schedule off --gpu 0
```

축별 실행 예시:

```bash
bash run/fmoe/tune_layout.sh --dataset movielens1m --parent_result run/artifacts/results/fmoe/<p1>.json --gpu 0
bash run/fmoe/tune_schedule.sh --dataset movielens1m --parent_result run/artifacts/results/fmoe/<p3>.json --mode alpha --gpu 0
bash run/fmoe/pipeline_ml1_rr.sh --datasets movielens1m,retail_rocket --gpus 0,1
```
