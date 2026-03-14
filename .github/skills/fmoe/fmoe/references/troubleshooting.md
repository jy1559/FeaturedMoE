# Troubleshooting

## 1) CUDA OOM
증상:
- trial이 연속 실패하고 `Best MRR@20 = 0.0000` 또는 `0/N trials OK`가 발생한다.

대응:
1. `train_batch_size`와 `eval_batch_size`를 줄인다.
2. `max_evals`를 줄여 스모크로 먼저 검증한다.
3. `--dry-run`으로 실제 커맨드와 배치 설정을 확인한다.
4. 문제가 반복되면 `schedule=off` 기준선으로 되돌려 원인 축을 분리한다.

## 2) Result JSON 미생성 또는 누락
증상:
- 로그는 있는데 `hyperopt_results/fmoe`에 JSON이 보이지 않는다.

원인:
- 과거 실행이 `hyperopt_results` 루트에 저장되었거나 `run_group`이 비어 있었을 수 있다.

대응:
1. 다음 위치를 모두 확인한다.
   - `/workspace/jy1559/FMoE/experiments/run/artifacts/results/*.json`
   - `/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe/*.json`
   - `/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hir/*.json`
2. `collect_results.py`를 사용해 루트 + 하위 그룹을 동시 스캔한다.
3. JSON이 없으면 로그 fallback 결과를 사용한다.

## 3) run_group/run_axis 불일치
증상:
- 결과 파일은 있으나 phase/axis 분류가 깨져 summary가 혼합된다.

대응:
1. run script에서 `--run-group`, `--run-axis`, `--run-phase` 전달 여부를 확인한다.
2. 로그 경로(`experiments/run/artifacts/logs/<group>/<axis>/<phase>`)와 JSON 필드가 맞는지 비교한다.
3. 새 트랙 추가 시 기존 그룹명(`fmoe`, `fmoe_hir`, `baseline`) 정책을 따르거나 분류 규칙을 문서화한다.

## 4) 데이터셋 이름 케이스 이슈
증상:
- `KuaiRec0.3`와 `kuairec0.3`가 다른 데이터셋으로 집계된다.

대응:
1. 스크립트에서 dataset을 소문자 canonical 이름으로 정규화한다.
2. 보고서에서 원본 이름은 별도 컬럼으로 보관하고 집계 키는 canonical 값을 사용한다.

## 5) 실행 파이프라인 점검 체크
- [ ] `launch_track.sh --track fmoe-main --dry-run` 성공.
- [ ] `collect_results.py --help` 출력 정상.
- [ ] `recommend_next.py --help` 출력 정상.
- [ ] `summary.csv`, `summary.md`, `best_by_dataset.json` 생성 확인.
- [ ] `next_plan.md`, `next_plan.json` 생성 확인.
