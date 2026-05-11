# FMoE_N4 Ablation Study Plan

## 1. 목적

이 문서는 나중에 `experiments/run/fmoe_n4` 아래에 ablation용 `.py` / `.sh` 런처를 실제로 만들 때, 추가 프롬프트 없이 그대로 구현에 들어갈 수 있도록 쓰는 구현 명세서다.

이번 계획에서 의도적으로 고정하지 않는 것은 두 가지다.

- 최종 기준 아키텍처
- 최종 기준 하이퍼파라미터

이 둘은 실험 완료 후 별도로 지정될 예정이므로, 본 문서에서는 **모든 ablation을 "나중에 선택될 base result json에 대한 delta override"** 로 정의한다. 따라서 이후 구현 시점에는 base를 하드코딩하지 말고, 반드시 base run에서 다음 항목을 읽어와야 한다.

- `fixed_values`
- best `learning_rate`
- `feature_mode`
- `layer_layout`
- `stage_router_wrapper`
- `stage_router_source`
- `stage_router_granularity`
- `stage_feature_injection`
- `stage_compute_mode`
- `train_batch_size`
- `eval_batch_size`

## 2. 코드베이스 기준점

아래 파일들이 실제 구현의 직접적인 템플릿이다.

- `experiments/run/fmoe_n4/stage1_a12_broad_templates.py`
  - `fmoe_n4` 트랙에서 manifest 작성, row 생성, command 조립, `launch_wide_rows` 호출까지 이미 구현되어 있다.
  - 새 ablation `.py`는 이 파일의 전체 runner 구조를 그대로 복제하는 것이 가장 빠르다.
- `experiments/run/fmoe_n4/run_staged_tuning.py`
  - `fmoe_n4`의 shell entrypoint가 어떤 인자를 넘기는지 보여준다.
- `experiments/run/fmoe_n3/ablation/run_ablation_12.py`
- `experiments/run/fmoe_n3/ablation/ablation_2_12.py`
- `experiments/run/fmoe_n3/ablation/ablation_3_12.py`
  - setting matrix를 만드는 방식, `setting_id` / `setting_key` / `setting_group` naming, summary/manifest 작성 방식이 이미 검증되어 있다.
- `writing/results/00_results_guide.md`
  - 어떤 실험이 논문 어느 패널과 연결되는지 기준이 된다.
- `writing/ACM_template/sample-sigconf.tex`
  - main text와 appendix가 요구하는 ablation 메시지를 직접 보여준다.

## 3. 나중에 구현할 파일 목록

아래 파일명을 그대로 쓰는 것을 권장한다.

| 파일 | 역할 | 권장 axis | 권장 phase |
| --- | --- | --- | --- |
| `experiments/run/fmoe_n4/ablation_common.py` | base result 해석, 공통 parser, 공통 manifest/row helper | 공통 | 공통 |
| `experiments/run/fmoe_n4/ablation_routing_control.py` | shared FFN / hidden / both / feature 라우팅 비교 | `ablation_kuairec_routing_control_v1` | `P4A` |
| `experiments/run/fmoe_n4/ablation_routing_control.sh` | 위 python entrypoint shell wrapper | 동일 | 동일 |
| `experiments/run/fmoe_n4/ablation_stage_structure.py` | stage count / stage removal / wrapper / order 비교 | `ablation_kuairec_stage_structure_v1` | `P4B` |
| `experiments/run/fmoe_n4/ablation_stage_structure.sh` | 위 python entrypoint shell wrapper | 동일 | 동일 |
| `experiments/run/fmoe_n4/ablation_cue_family.py` | cue family drop / subset / sequence-only 비교 | `ablation_kuairec_cue_family_v1` | `P4C` |
| `experiments/run/fmoe_n4/ablation_cue_family.sh` | 위 python entrypoint shell wrapper | 동일 | 동일 |
| `experiments/run/fmoe_n4/ablation_objective_variants.py` | consistency / z-loss / balance ablation | `ablation_kuairec_objective_v1` | `P4D` |
| `experiments/run/fmoe_n4/ablation_objective_variants.sh` | 위 python entrypoint shell wrapper | 동일 | 동일 |
| `experiments/run/fmoe_n4/ablation_portability_followup.py` | Beauty / Retail follow-up | `ablation_portability_followup_v1` | `P4E` |
| `experiments/run/fmoe_n4/ablation_portability_followup.sh` | 위 python entrypoint shell wrapper | 동일 | 동일 |

## 4. 공통 구현 원칙

### 4.1 base result를 반드시 외부 입력으로 받기

모든 launcher는 아래 인자를 공통으로 받아야 한다.

- `--base-result-json`
- `--datasets`
- `--gpus`
- `--seeds`
- `--only-setting`
- `--manifest-out`
- `--dry-run`
- `--smoke-test`
- `--smoke-max-runs`
- `--resume-from-logs`

환경변수 fallback도 같이 둔다.

- `N4_BASE_RESULT_JSON`
- `N4_DATASETS`
- `N4_GPUS`
- `N4_SEEDS`

`--base-result-json`가 없으면 launcher를 바로 실패시켜야 한다. 이번 문서 범위에서는 기준 아키텍처/하이퍼파라미터를 하드코딩하지 않기 때문이다.

### 4.2 공통 helper로 뺄 함수

`ablation_common.py`에 아래 함수들을 두는 것을 권장한다.

- `resolve_base_spec(base_result_json: str) -> dict`
  - base result json에서 `fixed_values`, best `learning_rate`, batch size, 주요 stage override를 읽는다.
- `build_lr_choices(base_lr: float, mode: str) -> list[float]`
  - `screen5`: `base_lr x [0.5, 0.75, 1.0, 1.25, 1.5]`
  - `tight3`: `base_lr x [0.85, 1.0, 1.15]`
- `clone_base_overrides(base_spec: dict) -> dict`
  - base override를 deepcopy해서 setting delta를 얹기 좋게 만든다.
- `apply_delta_overrides(base_overrides: dict, delta: dict) -> dict`
  - setting별 변경점만 patch한다.
- `common_arg_parser(description: str) -> argparse.ArgumentParser`
- `make_setting_row(...) -> dict`
- `manifest_path(log_root: Path, dataset: str, name: str) -> Path`
- `summary_path(log_root: Path, dataset: str) -> Path`

### 4.3 공통 import 정책

새 ablation `.py`는 아래 helper를 그대로 재사용하는 쪽이 맞다.

- `run_phase9_auxloss._apply_base_overrides`
- `run_phase9_auxloss._base_fixed_overrides`
- `run_phase9_auxloss._parse_csv_ints`
- `run_phase9_auxloss._parse_csv_strings`
- `run_phase9_auxloss.hydra_literal`
- `run_phase_wide_common.build_summary_fieldnames`
- `run_phase_wide_common.launch_wide_rows`

즉, runner 뼈대는 `fmoe_n4/stage1_a12_broad_templates.py`를 따르고, setting matrix 스타일은 `fmoe_n3/ablation/*.py`를 따른다.

### 4.4 base override에서 직접 patch할 항목

실험군 구현 시 새 setting은 아래 키만 바꾸는 것을 원칙으로 한다.

- `layer_layout`
- `stage_compute_mode`
- `stage_router_mode`
- `stage_router_source`
- `stage_router_granularity`
- `stage_feature_injection`
- `stage_router_wrapper`
- `stage_feature_family_mask`
- `stage_feature_drop_keywords`
- `route_consistency_lambda`
- `z_loss_lambda`
- `balance_loss_lambda`

그 외 `embedding_size`, `d_ff`, `d_router_hidden`, `d_feat_emb`, `expert_scale` 같은 용량 관련 값은 base에서 그대로 물려받는다. 이 문서는 그 값을 아직 정하지 않는다.

### 4.5 shell wrapper 규칙

각 `.sh`는 `stage1_a12_broad_templates.sh`와 동일한 구조로 만든다.

- repo root 계산
- `RUN_PYTHON_BIN` fallback
- `cd ${ROOT_DIR}/experiments`
- python entrypoint 호출
- env var로 `N4_*` 기본값 받기

새로운 shell wrapper에서도 dataset 기본값은 반드시 `KuaiRecLargeStrictPosV2_0.2`로 둔다.

## 5. 공통 실행 정책

### 5.1 공통 예산

모든 ablation은 크게 세 단계로 굴린다.

1. `smoke`
   - `--dry-run`
   - `--smoke-test`
   - `--smoke-max-runs 2`
2. `kuairec_scout`
   - dataset: `KuaiRecLargeStrictPosV2_0.2`
   - seeds: `1`
   - `lr_mode=screen5`
   - `search_algo=random`
   - `max_evals=5`
   - `tune_epochs=30`
   - `tune_patience=4`
   - batch size / eval batch size는 base에서 상속
3. `kuairec_confirm`
   - top setting만 추려서 seeds `1,2,3`
   - `lr_mode=tight3`
   - `search_algo=random`
   - `max_evals=3`
   - `tune_epochs=50`
   - `tune_patience=6`

### 5.2 selection rule

- scout 단계 정렬: `best_valid_mrr20`
- confirm 단계 정렬: `mean(test_mrr20 over seeds)`
- 차이가 `0.002` 이하이면 아래 순서로 tie-break
  - routing collapse가 덜한 setting
  - route consistency가 더 높은 setting
  - batch divergence / NaN가 없는 setting

### 5.3 logging policy

- routing control, stage structure, objective variants
  - `fmoe_diag_logging=true`
  - `fmoe_special_logging=true`
- cue family
  - `fmoe_diag_logging=true`
  - `fmoe_feature_ablation_logging=true`
- portability follow-up
  - `fmoe_special_logging=true`
  - `fmoe_diag_logging=true`

### 5.4 중복 row 제거 규칙

base가 나중에 정해지므로, 어떤 setting은 base와 완전히 같아질 수 있다. 아래 경우에는 row를 만들지 않는다.

- delta override 적용 후 base override와 완전히 동일
- base의 wrapper가 이미 `w5_exd`인데 `WRAPPER_ALL_W5_EXD` row를 다시 만드는 경우
- base의 router source가 이미 `feature`인데 `ROUTER_SOURCE_FEATURE` row를 다시 만드는 경우
- base에서 특정 aux lambda가 이미 `0`인데 `*_OFF` row가 동일해지는 경우

## 6. 실행 순서

실험 순서는 아래 순서를 권장한다.

1. routing control
   - 가장 작은 실험군으로 core claim을 먼저 확인한다.
   - hidden-only / both / feature-only가 구분되지 않으면 뒤의 stage/cue 실험 의미가 약해진다.
2. stage structure
   - routing source 축이 어느 정도 정리된 뒤 구조 축을 본다.
3. cue family
   - 구조가 정리된 뒤 어떤 cue를 빼도 gain이 남는지 본다.
4. objective variants
   - 최종 구조/feature 축이 정리된 뒤 regularization만 바꿔서 본다.
5. beauty / retail portability follow-up
   - main KuaiRec 결과를 다 정리한 후 top variant만 옮긴다.

## 7. 논문/결과 폴더 매핑

| 결과 섹션 | 사용 launcher | 목적 |
| --- | --- | --- |
| `writing/results/02_routing_control` | `ablation_routing_control.py` | shared FFN / hidden / mixed / behavior-guided 비교 |
| `writing/results/03_stage_structure` | `ablation_stage_structure.py` | stage removal, dense vs staged, wrapper / order 비교 |
| `writing/results/04_cue_ablation` | `ablation_cue_family.py` | category/time drop, sequence-only retention |
| `writing/results/A02_objective_variants` | `ablation_objective_variants.py` | consistency / z-loss / balance 기여 |
| `writing/results/A03_routing_diagnostics` | routing/objective/cue 런처 중 diag logging 켠 run | expert usage / entropy / consistency / feature bucket |
| `writing/results/A05_transfer` | `ablation_portability_followup.py` 또는 이후 transfer 전용 런처 | optional appendix only |

## 8. 교차 재사용할 setting

같은 변형을 여러 study에서 재사용해야 한다.

- `SHARED_FFN`
  - routing control과 stage structure 둘 다에서 필요하다.
  - 가능하면 routing-control summary를 stage-structure 추출에서 그대로 참조한다.
- `SEQUENCE_ONLY_PORTABLE`
  - cue ablation main panel과 beauty/retail follow-up 둘 다에서 쓴다.
- `ALL_AUX_OFF`
  - objective appendix 하한선이자 portability 단순화 후보로 재사용 가능하다.

## 9. 구현 시 권장 pseudo-code

```python
base_spec = resolve_base_spec(args.base_result_json)
settings = build_settings(args, base_spec)

rows = []
for setting in settings:
    overrides = apply_delta_overrides(clone_base_overrides(base_spec), setting["delta_overrides"])
    fixed_values = dict(base_spec["fixed_values"])
    search_space = {
        "learning_rate": build_lr_choices(base_spec["learning_rate"], mode=args.lr_mode),
    }
    rows.append(
        make_setting_row(
            dataset=args.dataset,
            base_spec=base_spec,
            setting=setting,
            overrides=overrides,
            fixed_values=fixed_values,
            search_space=search_space,
        )
    )

launch_wide_rows(...)
```

핵심은 `fixed_values`를 새로 설계하지 않는 것이다. base를 읽고, override delta만 얹는다.

## 10. 이 폴더에서 먼저 읽을 순서

나중에 실제 구현에 들어갈 사람은 아래 순서로 읽으면 된다.

1. `plan.md`
2. `01_routing_control.md`
3. `02_stage_structure.md`
4. `03_cue_ablation.md`
5. `04_objective_variants.md`
6. `05_portability_followup.md`
