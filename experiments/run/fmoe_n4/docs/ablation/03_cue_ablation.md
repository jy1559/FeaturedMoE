# FMoE_N4 Cue Family Ablation

## 1. 목적

이 study는 논문 본문 `Cue Family Ablation`을 직접 만들기 위한 실험이다.

핵심 질문은 두 가지다.

- category/time 계열 cue를 빼면 얼마나 성능이 떨어지는가
- rich metadata가 줄어도 sequence-derived cue만으로 gain이 얼마나 남는가

여기서 중요한 것은 "full model을 다시 튜닝한다"가 아니라, **later selected base에서 cue만 빼거나 줄인 delta 실험**이라는 점이다.

## 2. 나중에 만들 파일

- `experiments/run/fmoe_n4/ablation_cue_family.py`
- `experiments/run/fmoe_n4/ablation_cue_family.sh`

권장 axis / phase:

- `AXIS = ablation_kuairec_cue_family_v1`
- `PHASE_ID = P4C`
- `PHASE_NAME = N4_CUE_ABLATION`

## 3. 직접 재사용할 코드 anchor

- `experiments/run/fmoe_n3/ablation/run_ablation_12.py`
- `experiments/run/fmoe_n3/ablation/ablation_2_12.py`
- `outputs/p6_v2_dryrun_manifest.json`

`run_ablation_12.py`의 category / timestamp drop keyword를 그대로 재사용하는 것이 가장 안전하다.

### 3.1 keyword list를 그대로 복제할 것

```python
CATEGORY_DROP_KEYWORDS = ["cat", "theme"]
TIMESTAMP_DROP_KEYWORDS = [
    "timestamp",
    "gap",
    "pace",
    "int_",
    "_int",
    "sess_age",
    "ctx_valid_r",
    "valid_r",
    "delta_vs_mid",
]
```

## 4. 이 study에서 바꾸는 키

- `stage_feature_family_mask`
- `stage_feature_drop_keywords`

router source, granularity, wrapper, layout, aux lambda는 base에서 그대로 가져간다.

## 5. setting matrix

### 5.1 main paper 핵심 3-setting

| setting_id | setting_key | delta override | export label |
| --- | --- | --- | --- |
| `CF-01` | `DROP_CATEGORY_DERIVED` | `stage_feature_family_mask={macro:[Tempo,Memory,Exposure],mid:[Tempo,Memory,Exposure],micro:[Tempo,Memory,Exposure]}`, `stage_feature_drop_keywords=CATEGORY_DROP_KEYWORDS` | `remove_category` |
| `CF-02` | `DROP_TIMESTAMP_DERIVED` | `stage_feature_family_mask={macro:[Focus,Memory,Exposure],mid:[Focus,Memory,Exposure],micro:[Focus,Memory,Exposure]}`, `stage_feature_drop_keywords=TIMESTAMP_DROP_KEYWORDS` | `remove_time` |
| `CF-03` | `SEQUENCE_ONLY_PORTABLE` | `stage_feature_family_mask={macro:[Memory,Exposure],mid:[Memory,Exposure],micro:[Memory,Exposure]}`, `stage_feature_drop_keywords=CATEGORY_DROP_KEYWORDS + TIMESTAMP_DROP_KEYWORDS` | `sequence_only` |

later selected base row는 export label `full`로 같이 들어간다.

### 5.2 appendix 확장 6-setting

| setting_id | setting_key | delta override | 메모 |
| --- | --- | --- | --- |
| `CF-04` | `ONLY_MEMORY` | `stage_feature_family_mask={macro:[Memory],mid:[Memory],micro:[Memory]}`, `stage_feature_drop_keywords=[]` | sequence-only에서 무엇이 남는지 분해 |
| `CF-05` | `ONLY_EXPOSURE` | `stage_feature_family_mask={macro:[Exposure],mid:[Exposure],micro:[Exposure]}`, `stage_feature_drop_keywords=[]` | popularity/exposure 단독 |
| `CF-06` | `MEMORY_EXPOSURE` | `stage_feature_family_mask={macro:[Memory,Exposure],mid:[Memory,Exposure],micro:[Memory,Exposure]}`, `stage_feature_drop_keywords=[]` | portable pair |
| `CF-07` | `FOCUS_MEMORY` | `stage_feature_family_mask={macro:[Focus,Memory],mid:[Focus,Memory],micro:[Focus,Memory]}`, `stage_feature_drop_keywords=[]` | category-sensitive pair |
| `CF-08` | `NO_TEMPO` | `stage_feature_family_mask={macro:[Focus,Memory,Exposure],mid:[Focus,Memory,Exposure],micro:[Focus,Memory,Exposure]}`, `stage_feature_drop_keywords=[]` | time-free family ablation |
| `CF-09` | `NO_EXPOSURE` | `stage_feature_family_mask={macro:[Tempo,Focus,Memory],mid:[Tempo,Focus,Memory],micro:[Tempo,Focus,Memory]}`, `stage_feature_drop_keywords=[]` | popularity-free family ablation |

## 6. 구현 방식

### 6.1 `_stage_mask()` helper를 둘 것

`ablation_2_12.py`와 같은 helper를 그대로 둔다.

```python
def _stage_mask(selected_groups: list[str]) -> dict[str, list[str]]:
    return {
        "macro": list(selected_groups),
        "mid": list(selected_groups),
        "micro": list(selected_groups),
    }
```

### 6.2 structural feature drop 정책

이 study에서는 zero-fill이 아니라 structural removal로 간다. 따라서 아래 값을 항상 고정한다.

- `feature_perturb_mode = none`
- `feature_perturb_apply = none`
- `feature_perturb_family = []`
- `feature_perturb_keywords = []`

즉 feature를 노이즈로 바꾸지 말고, 아예 mask/drop로 제거한다.

## 7. 실행 순서

### 7.1 KuaiRec scout core

- `CF-01`~`CF-03`
- later selected base row와 함께 비교
- settings 수: 3 + base
- seed: `1`
- `max_evals=5`

### 7.2 KuaiRec scout appendix

- `CF-04`~`CF-09`
- settings 수: 6
- seed: `1`
- `max_evals=5`

### 7.3 confirm

confirm으로 올릴 것은 아래만 남긴다.

- `CF-03_SEQUENCE_ONLY_PORTABLE`
- `CF-01` / `CF-02` 중 더 해석 가치가 큰 1개
- appendix 중 best portable subset 1개

총 3개 이하로 제한한다.

## 8. 결과를 어떤 CSV에 넣을지

### 8.1 `writing/results/04_cue_ablation/04a_cue_ablation.csv`

필수 row:

- `full` -> later selected base row
- `remove_category` -> `CF-01`
- `remove_time` -> `CF-02`
- `sequence_only` -> `CF-03`

권장 metric:

- `metric = MRR`
- `cutoff = 20`
- `split = test`

### 8.2 `writing/results/04_cue_ablation/04b_cue_retention.csv`

retention은 아래 공식을 고정한다.

```text
relative_gain = (variant_value - shared_ffn_value) / (full_value - shared_ffn_value)
```

여기서 `shared_ffn_value`는 `01_routing_control.md`의 `RC-01_SHARED_FFN` 또는 `02_stage_structure.md`의 `ST-07_DENSE_FULL_ONLY`에서 가져온다.

필수 row:

- `remove_category`
- `remove_time`
- `sequence_only`

## 9. 구현 시 주의사항

- `CF-03_SEQUENCE_ONLY_PORTABLE`는 Beauty / Retail follow-up에서도 재사용한다.
- family mask와 keyword drop을 동시에 쓰는 이유는, family 이름만으로는 category/time derived 세부 feature를 완전히 제거하지 못할 수 있기 때문이다.
- base가 이미 category-free 또는 time-free setting이면 동일 row는 자동 제거한다.

## 10. 이 study가 끝나면 다음 문서로 넘어갈 기준

아래가 정리되면 `04_objective_variants.md`로 넘어간다.

- `sequence_only`가 dense baseline 대비 어느 정도 gain을 유지하는지
- category/time cue 중 어느 쪽이 더 중요한지
- portable subset이 Beauty / Retail로 옮길 가치가 있는지
