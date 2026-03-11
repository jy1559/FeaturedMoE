# EXPERIMENT_PLAN_v2

실험 실행은 이 문서에서 **계획/템플릿만** 제공합니다. 실제 실행은 별도 세션에서 진행합니다.

## 공통 기록 규칙
- 결과 저장: `experiments/run/artifacts/results/fmoe_v2_hir`
- 로그 저장: `experiments/run/artifacts/logs/fmoe_v2_hir`
- 타임라인: `experiments/run/artifacts/timeline/events.jsonl`
- Run phase 태그: `P0/P1/P2/P3/P4`
- 중단 기준:
  - 3회 연속 OOM
  - trial의 50% 이상이 결과 JSON 없이 실패
  - best `mrr@20` 개선폭이 0.002 미만으로 2 phase 연속 정체

## Phase 0: Smoke Checklist
목표: 경로/스키마/shape 이상 여부 확인
- `serial` 1회, `parallel` 1회 dry-run 커맨드 확인
- layout parser 에러 케이스(누락/음수) 사전 점검
- `--dry-run`에서 `track=fmoe_v2_hir`와 artifacts 경로 확인

예시(실행 템플릿):
```bash
bash experiments/run/fmoe_v2_hir/train_single.sh --dataset movielens1m --layout-id 0 --execution serial --gpu 0 --dry-run
bash experiments/run/fmoe_v2_hir/train_single.sh --dataset movielens1m --layout-id 2 --execution parallel --gpu 0 --dry-run
```

## Phase 1: ML1M mode/layout screening
목표: 실행모드 + layout 조합 1차 스크리닝
- 방식: **넓고 얕게** (`GPU 수 x GPU당 조합 수` 만큼 다건 실행)
- 각 실행(run)에서 고정:
  - `fmoe_v2_layout_id`, `fmoe_stage_execution_mode`
  - dimension/batch/기타 구조 파라미터
- 각 실행(run)에서 탐색:
  - `learning_rate`, `weight_decay` (discrete list)
  - 기본 예시: `lr=[1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4]`
  - 기본 예시: `wd=[0,1e-6,1e-5,1e-4,1e-3]`
- serial layout 후보는 stage `pass_layers` 1~5 포함하도록 구성

예시:
```bash
bash experiments/run/fmoe_v2_hir/p1_wide_shallow.sh \
  --datasets movielens1m,retail_rocket \
  --gpus 0,1 \
  --combos-per-gpu 3 \
  --max-evals 12 \
  --dry-run
```

## Phase 2: ML1M refinement (layout/execution 고정)
목표: P1 best 주변 정밀화
- 권장 trial: 16~24
- 축: `learning_rate`, `weight_decay`, `hidden_dropout_prob`, `balance_loss_lambda`
- 필요 시 parallel merge 축(`fmoe_v2_parallel_stage_gate_*`) 분리

예시:
```bash
bash experiments/run/fmoe_v2_hir/tune_layout.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v2_hir/<p1>.json --phase P2 --gpu 0 --dry-run
bash experiments/run/fmoe_v2_hir/tune_hparam.sh --dataset movielens1m --layout-id <best_layout> --execution <best_exec> --parent-result experiments/run/artifacts/results/fmoe_v2_hir/<p2>.json --phase P3 --gpu 0 --dry-run
```

## Phase 3: RetailRocket transfer
목표: ML1M best 설정 전이 및 안정화
- 권장 trial: 16~30
- 전이 우선: layout/execution 고정 + optimizer만 재튜닝
- 배치 크기와 dropout을 우선 미세조정

예시:
```bash
bash experiments/run/fmoe_v2_hir/tune_hparam.sh --dataset retail_rocket --layout-id <best_layout> --execution <best_exec> --max-evals 24 --phase P1 --gpu 1 --dry-run
```

## Phase 4: Schedule axis
목표: 성능 정체 구간에서 alpha/temp/top-k schedule만 분리 탐색
- 권장 trial: axis별 8~20
- axis: `alpha`, `temp`, `topk`, `combined`
- `combined`는 마지막에만 수행

예시:
```bash
bash experiments/run/fmoe_v2_hir/tune_schedule.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v2_hir/<p3>.json --mode alpha --phase P4 --gpu 0 --dry-run
bash experiments/run/fmoe_v2_hir/tune_schedule.sh --dataset movielens1m --parent-result experiments/run/artifacts/results/fmoe_v2_hir/<p3>.json --mode combined --phase P4 --gpu 0 --dry-run
```
