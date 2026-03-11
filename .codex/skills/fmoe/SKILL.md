---
name: fmoe
description: FMoE-first sequential recommendation experiment operation and improvement advisory skill for /workspace/jy1559/FMoE. Use when asked to run or tune FeaturedMoE experiments, transfer from MovieLens1M to RetailRocket, summarize MRR@20 from hyperopt JSON or logs, diagnose OOM or missing result files, compare FeaturedMoE_HiR, or propose next architecture, schedule, and layout trials in Korean or English.
---

# FMoE

## Overview
순차 추천 실험을 FMoE 중심으로 운영하고, 결과를 집계해 다음 실험안을 제안한다. 기본 루프는 ML1M 앵커 최적화 후 RetailRocket 전이이며, MRR@20 단일 best score를 채택 기준으로 사용한다.

## Quick Start
1. Dry-run으로 실행 파이프라인 확인.
```bash
bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --dry-run
```
2. FMoE 메인 트랙 실행(ML1M -> RetailRocket).
```bash
bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --datasets movielens1m,retail_rocket --gpus 0,1 --seed-base 42
```
3. 결과 요약 생성(JSON 우선, 로그 fallback).
```bash
python3 .codex/skills/fmoe/scripts/collect_results.py --repo-root /workspace/jy1559/FMoE --datasets movielens1m,retail_rocket --metric mrr@20
```
4. 다음 3개 실험안 생성.
```bash
python3 .codex/skills/fmoe/scripts/recommend_next.py --summary /workspace/jy1559/FMoE/experiments/run/hyperopt_results/summary.csv --mode fmoe-first --topn 3
```

## Environment / Override Notes
- 테스트나 Hydra compose 검증은 먼저 `conda activate FMoE` 환경에서 실행한다.
- FMoE config는 root alias와 grouped config가 섞여 있으므로 Hydra override를 조심한다.
- runtime `rule_router.variant` 같은 root alias는 `rule_router.variant=teacher_gls`처럼 직접 override한다.
- search space의 flat dotted key는 `++search.rule_router.variant=[...]`로 넣지 말고 `++search={rule_router.variant:[teacher_gls]}`처럼 dict merge로 넣는다.

## Fixed Policy
- FMoE를 1차 트랙으로 사용하고, HiR과 신규 아키텍처는 2차 비교/확장 트랙으로 다룬다.
- 기본 데이터셋 순서를 `movielens1m -> retail_rocket -> amazon_beauty -> foursquare -> kuairec0.3 -> lastfm0.3`로 둔다.
- 기본 지표를 `MRR@20`으로 고정한다.
- 채택 규칙은 재현성 제약 없는 단일 best score로 둔다.
- baseline 대규모 재튜닝을 기본 워크플로우에서 제외한다.

## Logging Scope (FMoE Only)
- FMoE 로그는 사용자가 세션을 명시적으로 시작한 경우에만 기록한다.
- 시작/종료 명령은 아래 2개로 고정한다.
```bash
make session_start TITLE="<session-title>"
make session_end
```
- active 세션 상태에서는 사용자의 후속 채팅 요청을 Codex가 자동 턴 기록한다.
- VSCode/일반 디버깅/비FMoE 질문에는 자동 로그를 생성하지 않는다.

## Track Overview Auto-Update
- `experiments/run/common/track_experiment_report.py`로 트랙 단위 요약을 유지한다.
- 실행 스크립트(`fmoe`, `fmoe_hir`, `fmoe_v2`)는 run 종료 후 아래 파일을 자동 갱신한다.
  - `experiments/run/artifacts/logs/<track>/experiment_overview.md`
  - `experiments/run/artifacts/logs/<track>/experiment_overview.csv`
- `fmoe_v2` 스크립트는 `--exp-name`, `--exp-desc`, `--exp-focus`를 tracker로 전달해
  개별 런 나열 대신 **실험 단위 요약(설명/비교변수/best 설정/로그 경로)**로 집계한다.
- 포함 규칙:
  - OOM run은 포함.
  - `success` + 유효 `MRR@20` run 포함.
  - OOM이 아닌 에러 run, `MRR@20` 미생성 run은 제외.

## Operating Workflow
1. Step 1: Dry-run 검증 수행.
2. Step 2: ML1M에서 FMoE 단계별 탐색(P0~P4) 수행.
3. Step 3: RetailRocket으로 전이 실행.
4. Step 4: `collect_results.py`로 summary를 작성.
5. Step 5: `recommend_next.py`로 우선순위 3개 실험안을 제시.

## Track Selection
- `fmoe-main`: `experiments/run/fmoe/pipeline_ml1_rr.sh`를 사용해 메인 파이프라인을 실행.
  - 대상 데이터셋은 `movielens1m`, `retail_rocket`로 제한한다.
- `hir-compare`: `experiments/run/fmoe_hir/run_4phase_hir.sh`를 데이터셋별로 실행.
- `arch-probe`: 신규 변형 검증용 템플릿 명령만 출력(비파괴 모드).

## References
- 저장소/실험 경로 맵: `references/repo-map.md`
- FMoE 단계별 운영 지침: `references/fmoe-playbook.md`
- HiR 및 아키텍처 확장 체크리스트: `references/hir-and-arch-extension.md`
- 장애 대응 가이드: `references/troubleshooting.md`
