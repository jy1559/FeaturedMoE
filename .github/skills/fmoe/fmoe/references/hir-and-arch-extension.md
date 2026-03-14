# HiR and Architecture Extension

## HiR Track Role
- HiR는 2차 비교 트랙으로 운용한다.
- 주 목적은 FMoE 기준선 대비 routing 계층화 효과를 확인하는 것이다.
- 기본 실행은 `run_4phase_hir.sh`를 사용한다.

## HiR Quick Command
```bash
bash .codex/skills/fmoe/scripts/launch_track.sh --track hir-compare --datasets movielens1m,retail_rocket --gpus 0,1 --seed-base 42 --dry-run
```

## HiR Interpretation Rules
- `serial`과 `parallel` stage merge를 분리 비교한다.
- `off`와 `temp_mild` schedule을 같은 budget에서 비교한다.
- FMoE 대비 성능이 낮아도 라우팅 entropy/안정성 신호가 있으면 보조 실험을 유지한다.

## New Architecture Template

신규 변형(HiR 외) 추가 시 다음 순서를 따른다.

1. 모델 코드 추가:
- 위치: `/workspace/jy1559/FMoE/experiments/models/<NewModel>`
- 입력/출력 인터페이스를 RecBole 모델 규약과 맞춘다.

2. Config 추가:
- `/workspace/jy1559/FMoE/experiments/configs/model/<new_model>.yaml`
- 튜닝용이면 `<new_model>_tune.yaml`을 분리한다.

3. 등록/패치 확인:
- `/workspace/jy1559/FMoE/experiments/recbole_patch.py`에서 모델 등록 경로를 점검한다.

4. Run script 연결:
- `experiments/run/<new_track>/`에 최소 `train_single.sh`, `tune_hparam.sh`를 둔다.
- `--run-group`, `--run-axis`, `--run-phase`를 명시해 결과 집계를 일관화한다.

5. 결과 포맷 일치:
- `hyperopt_tune.py` 출력 JSON에서 `dataset`, `model`, `best_mrr@20`, `run_group`, `run_axis`, `run_phase` 필드를 유지한다.

## Extension Checklist
- [ ] `model=<new_model>`로 `recbole_train.py` 단일 실행 성공.
- [ ] `hyperopt_tune.py --max-evals 1` 스모크 성공.
- [ ] `run_group` 폴더 또는 루트에서 JSON 생성 확인.
- [ ] `collect_results.py`가 새 트랙 결과를 읽고 `summary.csv`에 반영.
- [ ] `recommend_next.py`에서 새 트랙을 분류 가능하도록 모델명 규칙 반영.

