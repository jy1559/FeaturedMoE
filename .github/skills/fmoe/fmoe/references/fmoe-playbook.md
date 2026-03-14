# FMoE Playbook

## Goal
- `FMoE`를 주 트랙으로 운영해 SeqRec 성능을 높인다.
- 기본 루프를 `movielens1m -> retail_rocket` 순서로 고정한다.
- 지표는 `MRR@20` 단일값을 사용한다.

## Default Workflow

### 1) Dry-run
```bash
bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --dry-run
```

### 2) Main run (ML1M -> RetailRocket)
```bash
bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --datasets movielens1m,retail_rocket --gpus 0,1 --seed-base 42
```

### 3) Collect and summarize
```bash
python3 .codex/skills/fmoe/scripts/collect_results.py --repo-root /workspace/jy1559/FMoE --datasets movielens1m,retail_rocket --metric mrr@20
```

### 4) Generate next plan
```bash
python3 .codex/skills/fmoe/scripts/recommend_next.py --summary /workspace/jy1559/FMoE/experiments/run/artifacts/results/summary.csv --mode fmoe-first --topn 3
```

## P0-P4 Phase Intent
- `P0`: 고정 설정 재현성 점검(짧은 seed sweep).
- `P1`: 주요 하이퍼파라미터 탐색.
- `P2`: layout 탐색.
- `P3`: P2 결과를 고정하고 hparam 재정밀 탐색.
- `P4`: schedule 축(alpha/temp/topk/combined) 분리 탐색.

## Layout and Schedule Tuning Order
1. ML1M에서 layout anchor를 먼저 확정한다.
2. RetailRocket에서 동일 layout을 먼저 적용한다.
3. 필요 시 데이터셋별 layout을 재탐색한다.
4. schedule은 off 기준선을 둔 다음 on 프리셋을 비교한다.

## Dataset Expansion Order
- 1차: `movielens1m`, `retail_rocket`
- 2차: `amazon_beauty`
- 3차: `foursquare`
- 4차: `kuairec0.3`
- 5차: `lastfm0.3`

확장 데이터셋 부트스트랩 예시:
```bash
bash /workspace/jy1559/FMoE/experiments/run/fmoe/tune_hparam.sh \
  --dataset amazon_beauty --gpu 0 --layout-id 0 --schedule-preset off \
  --max-evals 20 --phase P1EXT --dry-run
```

## Policy
- baseline 대규모 재튜닝을 기본 루프에서 제외한다.
- 개선 채택은 단일 best score로 판단한다.
- 결과 보고는 수치 요약표와 가설 중심 해석을 함께 제공한다.
