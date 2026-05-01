# RouteRec Rebuttal Plan (Modified v2)

> 작성일: 2026-04-30  
> 기준 논문: `writing/ACM_FInal/sample-sigconf_final.tex` (RecSys '26 제출본)  
> 기반 계획: `routerec_rebuttal_1month_plan.md`

---

## 0. 핵심 방어 명제

> RouteRec의 이득은 MoE capacity 증가가 아니라 behaviorally aligned cue를 routing prior로 사용했기 때문이며, cue construction과 evaluation protocol은 leakage 없이 설계되었고, dataset별 gain 차이는 behavioral routing demand 관점에서 설명 가능하다.

---

## 1. 코드 및 폴더 현황

### 1.1 실험 모델

**현재 논문에서 쓰는 모델: `FeaturedMoE_N3`**  
(`experiments/models/FeaturedMoE_N3/featured_moe_n3.py`)

- `feature_perturb_mode` / `feature_perturb_apply` / `feature_perturb_family` 파라미터 이미 구현됨
- modes: `none`, `zero`, `shuffle`, `family_permute` 지원
- apply: `eval`, `train`, `both` 지원
- `stage_router_source`: `feature`, `hidden`, `both` 지원
- **shuffled/random cue ablation이 코드 변경 없이 config 수준에서 가능**

### 1.2 실험 run 폴더 현황 (`experiments/run/`)

#### 현재 유지 트랙 (핵심)

| 폴더 | 내용 | 역할 |
|---|---|---|
| `FMoE_first/` | 이전 `fmoe_n4` (이름 변경) | RouteRec 메인 실험 트랙 |
| `baseline_first/` | 이전 `baseline_2` (이름 변경) | Baseline 메인 실험 트랙 |
| `final_experiment/` | 기존 ablation 스택 | real_final_ablation, Q2~Q5 등 |

#### 기존 트랙 archive

`_archive/` 폴더로 이동 완료:
```
_archive/
├── fmoe/          ← 구 fmoe track
├── fmoe_full/
├── fmoe_hgr/
├── fmoe_hgr_v4/
├── fmoe_n/
├── fmoe_n2/
├── fmoe_n3/
├── fmoe_rule/
├── fmoe_v2/
├── fmoe_v3/
├── fmoe_v4_distillation/
├── baseline/      ← 구 baseline (baseline_first로 대체됨)
└── baseline_3/
```

#### 새로 만들 rebuttal 전용 폴더

```
experiments/run/rebuttal/   ← [NEW]
```

### 1.3 기존 코드 재활용 포인트

| 재활용 대상 | 위치 |
|---|---|
| Cue perturb config 설정 참고 | `FMoE_first/ablation/ablation_routing_control.py` (RC-15~18) |
| Hidden-only router 설정 참고 | `final_experiment/real_final_ablation/common.py` → `q2_settings()` |
| Stage structure ablation 참고 | `final_experiment/real_final_ablation/q3_stage_structure.py` |
| Routing weight 추출 | `models/FeaturedMoE_N3/analysis_logger.py` |
| Behavior bin 분석 참고 | `final_experiment/real_final_ablation/appendix/dataset_appendix_analysis.py` |
| Best config 파일 | `final_experiment/real_final_ablation/configs/base_candidates.csv` |

### 1.4 ablation_routing_control.py에 이미 있는 perturbation 설정

`FMoE_first/ablation/ablation_routing_control.py`에 이미 정의된 설정들:

| Setting | 내용 |
|---|---|
| RC-02 `ROUTER_SOURCE_HIDDEN` | hidden-only router |
| RC-03 `ROUTER_SOURCE_BOTH` | hidden + cue 혼합 router |
| RC-04 `ROUTER_SOURCE_FEATURE` | feature(cue)-only router |
| RC-15 `FEATURE_ROUTER_EVAL_ZERO` | eval time cue zero |
| RC-16 `FEATURE_ROUTER_EVAL_SHUFFLE_ALL` | eval time all cue shuffle |
| RC-17 `FEATURE_ROUTER_EVAL_SHUFFLE_MEMORY` | eval time Memory family shuffle |
| RC-18 `HIDDEN_ROUTER_EVAL_SHUFFLE_ALL` | hidden router + shuffle control |

**이 설정들을 `FeaturedMoE_N3` 모델과 함께 rebuttal 실험에 직접 활용할 수 있다.**

---

## 2. Rebuttal 실험 계획

### 2.1 실험 폴더 구조

```
experiments/run/rebuttal/
├── README.md
├── common.py               ← FMoE_first best config 로딩 헬퍼
├── LEAKAGE_AUDIT.md        ← leakage 방어 문서
│
├── exp_rb1_cue_perturb/    ← P0-A: shuffled/random cue ablation
│   ├── rb1_cue_perturb.py
│   └── rb1_cue_perturb.sh
│
├── exp_rb2_capacity/       ← P0-B: capacity-matched baseline
│   ├── rb2_capacity.py
│   └── rb2_capacity.sh
│
├── exp_rb3_seed/           ← P0-C: seed variance
│   ├── rb3_seed.py
│   └── rb3_seed.sh
│
├── exp_rb4_cue_corr/       ← P1-A: cue↔route correlation
│   ├── rb4_extract_weights.py
│   ├── rb4_compute_corr.py
│   └── rb4_plot_heatmap.py
│
├── exp_rb5_bin_gain/       ← P1-B: behavior bin별 route/gain
│   ├── rb5_bin_users.py
│   ├── rb5_compute_gain.py
│   └── rb5_plot.py
│
└── exp_rb6_masking/        ← P1-C: expert family masking
    ├── rb6_masking.py
    └── rb6_masking.sh
```

---

### 2.2 P0-A: Shuffled/Random Cue Ablation (최우선)

**방어 목표**: behavioral cue alignment 없이는 RouteRec gain이 약해진다.

**핵심**: `FeaturedMoE_N3`에 이미 perturbation 지원 있음 → 코드 작성 없이 config만으로 가능.

#### 파일: `exp_rb1_cue_perturb/rb1_cue_perturb.py`

기반: `FMoE_first/ablation/ablation_routing_control.py` 의 `_perturb_delta()` 패턴 그대로 활용.

| Setting | delta_overrides | 의미 |
|---|---|---|
| RB1-00 | 없음 (base) | RouteRec true cues (논문 결과 재확인) |
| RB1-01 | `feature_perturb_mode=zero, apply=eval` | cue zero-out at eval |
| RB1-02 | `feature_perturb_mode=shuffle, apply=eval` | cross-sample cue shuffle at eval |
| RB1-03 | `stage_router_source={all: hidden}` | hidden-only router (RC-02 재실행) |
| RB1-04 | `stage_router_source={all: both}` | hidden + cue 혼합 (RC-03) |
| RB1-05 | `feature_perturb_mode=family_permute, family=[Memory], apply=eval` | Memory family만 shuffle |
| RB1-06 | SASRec (base) | 공통 baseline |

**우선 데이터셋**: KuaiRec → LastFM → Foursquare (시간 부족 시 KuaiRec + LastFM만)

**common.py에서**: `FMoE_first` best config CSV (`final_experiment/real_final_ablation/configs/base_candidates.csv`) 에서 KuaiRec, LastFM 최적 설정 불러오는 헬퍼 작성.

---

### 2.3 P0-B: Capacity-Matched Baseline

**방어 목표**: parameter 수 증가만으로는 RouteRec gain이 재현되지 않음.

#### 파일: `exp_rb2_capacity/rb2_capacity.py`

**접근법**: 기존 SASRec 모델의 FFN hidden dim을 RouteRec과 parameter 수가 비슷하도록 확장.

```
RouteRec total params 계산 스크립트: rb2_count_params.py
  → RouteRec best config 로드 후 sum(p.numel() for p in model.parameters())
  → SASRec의 d_ff 조정해서 동일 param count 맞추기
```

| Setting | 설명 |
|---|---|
| RB2-00 | SASRec (base) |
| RB2-01 | SASRec-wide (param-matched, d_ff 증가) |
| RB2-02 | Flat MoE + hidden router (RC-02 기반) |
| RB2-03 | RouteRec |

**우선 데이터셋**: KuaiRec + LastFM

---

### 2.4 P0-C: Seed Variance Robustness

**방어 목표**: single-run lucky result 의심 방어.

#### 파일: `exp_rb3_seed/rb3_seed.py`

`FMoE_first` best config에서 seed만 변경: seeds = [1, 2, 3]

| 대상 모델 | 데이터셋 |
|---|---|
| SASRec | KuaiRec, LastFM, ML-1M |
| Best non-RouteRec baseline | KuaiRec, LastFM, ML-1M |
| Hidden MoE | KuaiRec, LastFM |
| RouteRec | KuaiRec, LastFM, ML-1M |

ML-1M은 RouteRec이 약한 케이스를 정직하게 포함.

---

### 2.5 P0-D: Leakage Audit (실험 없음, 문서 작업)

#### 파일: `exp_rb1_cue_perturb/` 또는 `LEAKAGE_AUDIT.md`

`feature_meta_v3.json`의 내용과 `dataset_preprocessing_feature_add.md`를 기반으로 작성.

핵심 포인트:
- 모든 cue는 `strict_prefix` rule 적용 (target item 이후 정보 미사용)
- `mac*` feature: prior sessions에서만 계산 (train split 기준 정규화)
- `mid` feature: current session prefix에서만 계산
- `mic` feature: 최근 5 interactions prefix에서만 계산
- RC-15 (cue zero) 실험으로 eval-time에도 cue에 의존함을 보일 수 있음

---

### 2.6 P1-A: Cue Score ↔ Routing Weight Correlation

**방어 목표**: router가 실제로 behavioral cue에 반응함.

#### 파일: `exp_rb4_cue_corr/`

`models/FeaturedMoE_N3/analysis_logger.py`의 `ExpertAnalysisLogger`를 활용:
- trained checkpoint에서 test set routing weights 추출
- cue feature value ↔ corresponding family routing weight의 Spearman correlation
- scope (macro/mid/micro) × family (Tempo/Focus/Memory/Exposure) × dataset heatmap

**기대 출력**: `rb4_corr_heatmap_{dataset}.csv`, Figure용 heatmap

---

### 2.7 P1-B: Behavior Bin별 Route Profile + Gain

**방어 목표**: behavior cue → route activation → performance gain chain.

#### 파일: `exp_rb5_bin_gain/`

`final_experiment/real_final_ablation/appendix/dataset_appendix_analysis.py` 패턴 참고.

Binning dimensions:
- `mid_repeat_r` (low/mid/high)
- `mid_int_mean` (slow/mid/fast)
- `mid_cat_top1` (broad/mid/narrow)
- session length (short/mid/long)

각 bin에서:
1. 평균 cue score
2. 평균 family routing weight (analysis_logger 활용)
3. RouteRec NDCG@10
4. SASRec NDCG@10
5. RouteRec gain

---

### 2.8 P1-C: Expert Family Masking

**방어 목표**: expert family가 기능적으로 대응하는 behavioral regime에 실제 기여.

#### 파일: `exp_rb6_masking/`

`FeaturedMoE_N3`에 `expert_mask_family` 파라미터 추가 필요 (inference-time only):
```python
# featured_moe_n3.py에 추가할 것
self.expert_mask_family = resolver.get("expert_mask_family", [])
# → inference 시 해당 family experts의 routing weight을 0으로 강제
```

trained checkpoint 재사용 (새 training 불필요).  
**우선 KuaiRec 1개에서만** → 잘 나오면 LastFM 확장.

---

### 2.9 Full Dataset 추가 실험 (P0-E: optional)

**방어 목표**: sample이 너무 작다는 reviewer 지적 방어.

현재: `lastfm0.03` (130 users), `KuaiRecLargeStrictPosV2_0.2` (1,122 users)

Full dataset 실험을 원한다면:
1. `dataset_preprocessing_feature_add.md` 참조하여 preprocessing 코드 재작성
2. `experiments/tools/build_kuairec_basic.py`, `build_lastfm_basic.py` 신규 작성
3. `experiments/tools/build_feature_v3_from_basic.py` 신규 작성 (feature_meta_v3.json 역공학 기반)
4. `experiments/tools/build_feature_v4_from_v3.py` 재활용 (현존)

**주요 불확실 파라미터**:
- KuaiRec: watch_ratio threshold (0.5 유력), session gap threshold (30분 or 1시간)
- LastFM: user sampling 방식, session gap threshold (30분 유력)

→ 현실적으로 rebuttal 1개월 안에 full dataset preprocessing까지 하기는 무리.  
→ **P0-A~C를 먼저 완료 후 시간이 남으면 시도.**

---

## 3. 1개월 일정 (수정판)

### Week 1 (4/30 ~ 5/6): 폴더 정리 + 실험 설계

| 작업 | 내용 | 완료 여부 |
|---|---|---|
| 폴더 정리 | `fmoe_n4` → `FMoE_first`, `baseline_2` → `baseline_first`, 구 트랙 → `_archive/` | ✓ 완료 |
| `rebuttal/` 생성 | 폴더 구조 + README | 예정 |
| `LEAKAGE_AUDIT.md` 작성 | feature_meta + preprocessing 역공학 기반 | 예정 |
| FMoE_first best config 확인 | base_candidates.csv → KuaiRec/LastFM best 설정 추출 | 예정 |
| `rb1_cue_perturb.py` 초안 | RC-15~18 패턴 기반, FeaturedMoE_N3 활용 | 예정 |

### Week 2 (5/7 ~ 5/13): P0-A + P0-B 실험 실행

| 작업 | 데이터셋 |
|---|---|
| exp_rb1 (cue perturb) | KuaiRec + LastFM |
| exp_rb2 (capacity) | KuaiRec + LastFM |
| exp_rb3 (seed) start | KuaiRec |

### Week 3 (5/14 ~ 5/20): P0 마무리 + P1 시작

| 작업 | 데이터셋 |
|---|---|
| exp_rb1 Foursquare 확장 | Foursquare |
| exp_rb3 (seed) 마무리 | LastFM, ML-1M |
| exp_rb4 (cue corr) | KuaiRec + LastFM |
| exp_rb5 (bin gain) | KuaiRec |

### Week 4 (5/21 ~ 5/27): P1 마무리 + Rebuttal 패키징

| 작업 | 내용 |
|---|---|
| exp_rb6 (masking) | KuaiRec |
| 결과 표/그림 선택 | 3~4개 정리 |
| Rebuttal 답변 템플릿 | concern별 초안 |
| (시간 허용 시) full dataset preprocessing | lastfm_full or KuaiRec_full |

---

## 4. 코드 작업 체크리스트

### 즉시 사용 가능 (코드 수정 없이)

- `FeaturedMoE_N3` + `feature_perturb_mode/apply` → cue perturbation 실험
- `FMoE_first/ablation/ablation_routing_control.py` RC-02~18 → routing control 설정 참고
- `final_experiment/real_final_ablation/common.py` → q2_settings 구조 재활용
- `experiments/tools/build_feature_v4_from_v3.py` → v3→v4 split 재사용

### 신규 작성 필요

| 파일 | 내용 | 우선순위 |
|---|---|---|
| `rebuttal/common.py` | FMoE_first best config 로딩 헬퍼 | P0 |
| `rebuttal/LEAKAGE_AUDIT.md` | leakage 방어 문서 | P0 |
| `rebuttal/exp_rb1/rb1_cue_perturb.py` | cue perturb 실험 설정 | P0-A |
| `rebuttal/exp_rb2/rb2_capacity.py` | capacity-matched 실험 | P0-B |
| `rebuttal/exp_rb3/rb3_seed.py` | seed variance 실험 | P0-C |
| `rebuttal/exp_rb4/rb4_*.py` | cue-route correlation 분석 | P1-A |
| `rebuttal/exp_rb5/rb5_*.py` | behavior bin gain 분석 | P1-B |
| `rebuttal/exp_rb6/rb6_masking.py` | expert masking 실험 | P1-C |

### 모델 코드 수정 필요

| 파일 | 수정 내용 | 우선순위 |
|---|---|---|
| `models/FeaturedMoE_N3/featured_moe_n3.py` | `expert_mask_family` 파라미터 추가 (inference-time family masking) | P1-C |

---

## 5. 데이터셋 전처리 재구성 (선택사항)

`dataset_preprocessing_feature_add.md` 에 상세 설계 포함.

Full dataset 재구성이 필요한 경우 작성할 파일:

| 파일 | 내용 |
|---|---|
| `experiments/tools/build_kuairec_basic.py` | KuaiRec raw → basic (sessionization + filtering) |
| `experiments/tools/build_lastfm_basic.py` | LastFM raw → basic |
| `experiments/tools/build_feature_v3_from_basic.py` | basic → feature_added_v3 |
| `experiments/tools/run_full_pipeline.sh` | 전체 파이프라인 실행 |

불확실한 파라미터가 있으므로 `dataset_preprocessing_feature_add.md`의 "역공학 불확실 요소" 섹션 먼저 해결 필요.

---

## 6. 결과 패턴별 대응 전략

| Case | 대응 |
|---|---|
| A. Shuffled cue 크게 하락 | "behavioral alignment 필수" — capacity 주장 강하게 반박 |
| B. Capacity-matched도 비슷 | diagnostic / interpretability 중심 이동; bin gain + masking 강조 |
| C. Cue-route correlation 있지만 masking 약함 | "behavior-aligned routing"까지만 주장; masking은 appendix |
| D. ML-1M 계속 약함 | routing demand 낮은 negative case로 정직하게 사용 |

---

## 7. Rebuttal 답변 템플릿 (Week 4 작성 예정)

1. **Extra capacity concern** → RB1 (cue perturb) + RB2 (capacity-matched) 결과
2. **Feature engineering concern** → "cue is routing prior, not representation feature"
3. **Hidden router concern** → RB1-03 (hidden-only MoE) vs RB1-00 (true cues)
4. **Leakage concern** → LEAKAGE_AUDIT.md + feature_meta의 strict_prefix rule
5. **Dataset-dependent gain concern** → behavioral routing demand framework + ML-1M negative case

---

## 8. 보류 실험 (다음 제출용)

- Counterfactual Cue Intervention (`dataset_preprocessing_feature_add.md` 섹션 2.8 참조)
- Router Transfer / Low-resource Adaptation
- SRPFN / PFN 연결
