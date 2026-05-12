# Cue Perturbation Ablation (P1-A)

## 1. 실험 배경과 핵심 주장

RouteRec의 contribution을 한 문장으로 요약하면:

> **"behavioral cue는 sequential model의 hidden state와 독립적인 routing signal을 제공하며, 이를 통해 이질적인 세션 유형에 맞는 expert를 선택함으로써 성능이 향상된다."**

이 주장은 세 개의 독립된 sub-claim으로 분해된다.

| Sub-claim | 의미 | 증명 방법 |
|-----------|------|-----------|
| **C1. Cue Content** | cue 값 자체가 routing에 중요하다 | 값을 제거/교란했을 때 성능이 하락해야 함 |
| **C2. Cue Independence** | cue는 hidden state에서 얻을 수 없는 독립 정보다 | hidden만으로 routing해도 비슷하면 C2 반증 |
| **C3. Training Effect** | 모델이 cue를 활용하도록 학습된다 | 학습 중 cue 없애도 eval 때 cue 주면 회복되는가 |

이 실험은 각 sub-claim에 대한 **falsification test** 역할을 한다.  
각 조건에서 성능이 하락할수록 논문 주장이 더 강하게 지지된다.

---

## 2. 가설과 예상 결과

### 가설 H1 — cue 콘텐츠 의존성

**"cue를 제거하거나 교란하면 routing signal이 손상되어 성능이 크게 하락한다."**

| 조건 | 예상 | 근거 |
|------|------|------|
| `eval_zero` | MRR@20 큰 폭 하락 | routing이 0으로 collapsed → expert 선택 무작위화 |
| `eval_shuffle` | `eval_zero`보다 하락폭 작음 | cue 분포 유지, identity만 손상 |
| `eval_global_permute` | `eval_shuffle`와 유사하거나 약간 큼 | 배치 수준 치환으로 correlation 완전 파괴 |

→ **논문 내러티브:** `intact` >> `eval_zero` ≈ `eval_shuffle` >> baseline  
 "cue 값과 세션 간의 대응관계가 routing에 필수적이다."

### 가설 H2 — hidden state 비대체성

**"hidden state만으로 routing해도 큰 차이가 없다면 cue는 불필요하다."**  
(이 가설이 기각되어야 논문 주장이 성립)

| 조건 | 예상 | 근거 |
|------|------|------|
| `hidden_only` | `eval_zero`와 유사한 수준으로 하락 | hidden state는 cue가 포착하는 세션 heterogeneity를 충분히 인코딩하지 못함 |
| `feature_only` | `hidden_only`보다 좋거나 비슷 | cue 단독으로도 meaningful routing이 가능 |

→ **논문 내러티브:** `intact` >> `hidden_only` ≈ `eval_zero`  
 "hidden state는 cue의 대체재가 될 수 없다. behavioral cue는 hidden에 없는 routing signal을 제공한다."

### 가설 H3 — 학습 시 cue 의존성

**"학습 중 cue 없이 학습하면, 모델은 cue를 활용하는 능력을 잃는다."**

| 조건 | 예상 | 근거 |
|------|------|------|
| `train_zero` | eval 때 cue 주면 partial 회복 (intact보다는 낮음) | 모델이 cue를 활용하도록 학습하는 능력이 부분적으로 존재 |
| `both_zero` | `train_zero`보다 낮음 (MoE 구조 기여만 남음) | eval 때도 cue 없으니 routing 자체가 무의미 |

→ **논문 내러티브:** `intact` > `train_zero` > `both_zero`  
 "`both_zero`는 'MoE 구조만 있고 cue routing 없는' 상태이므로 구조 자체보다 cue routing이 더 중요하다."

### 가설 H4 — Behavioral axis별 기여도

**"특정 behavioral axis (memory, focus, tempo, exposure)가 routing에 더 중요한 신호를 제공한다."**

| 조건 | 예상 |
|------|------|
| `eval_family_tempo` | foursquare에서 하락 가장 클 것 (체크인 데이터 특성: 시간 패턴 중요) |
| `eval_family_memory` | KuaiRec에서 하락 가장 클 것 (롱테일 반복 소비 패턴) |
| `eval_family_focus` / `eval_family_exposure` | 중간 수준 |

→ **논문 내러티브:** 패밀리별 Δ MRR@20 막대그래프 (Figure)로 시각화

### 가설 H5 — 의미론적 구조 의존성

**"routing은 단순히 cue 벡터 값이 아니라 각 axis의 의미론적 역할을 이용한다."**

| 조건 | 예상 |
|------|------|
| `eval_role_swap` | 값은 real이나 semantic 뒤집힘 → `eval_zero`보다 오히려 더 큰 하락 가능 |
| `eval_stage_mismatch` | macro/mid/micro hierarchy 파괴 → 중간 수준 하락 |

→ **논문 내러티브:** "routing은 cue 값의 크기만 보는 게 아니라 각 axis의 semantic role에 의존한다."

---

## 3. 실험 조건 전체 목록

### 그룹 A: Eval-only (P0 checkpoint 재사용, 새 학습 없음)

`eval_perturb.py`로 실행. cfg override만으로 perturbation 동작 변경.

| 조건 | mode | apply | family | 가설 |
|------|------|-------|--------|------|
| `intact` | none | none | - | 기준값 |
| `eval_zero` | zero | eval | - | H1 |
| `eval_shuffle` | shuffle | eval | - | H1 |
| `eval_global_permute` | global_permute | eval | - | H1 |
| `eval_role_swap` | role_swap | eval | - | H5 |
| `eval_stage_mismatch` | stage_mismatch | eval | - | H5 |
| `eval_family_memory` | zero | eval | memory | H4 |
| `eval_family_focus` | zero | eval | focus | H4 |
| `eval_family_tempo` | zero | eval | tempo | H4 |
| `eval_family_exposure` | zero | eval | exposure | H4 |

### 그룹 B: Train-time (새 학습 필요)

`train_perturb.py`로 실행. P0 best lr/wd 고정 (max-evals=1), FIXED_PARAMS arch 동일.

| 조건 | new_arch | overrides 핵심 | 가설 |
|------|----------|---------------|------|
| `hidden_only` | ★ | router_use_feature=false | H2 |
| `feature_only` | ★ | router_use_hidden=false | H2 |
| `train_zero` | - | perturb_mode=zero, apply=train | H3 |
| `both_zero` | - | perturb_mode=zero, apply=both | H3 |
| `train_shuffle` | - | perturb_mode=shuffle, apply=train | H3 |

★ router 구조 변경 → P0 checkpoint와 state_dict 호환 불가 → 반드시 새 학습

---

## 4. 논문 활용 매핑

### Table 3: Cue Perturbation Ablation (Sec. 4.3)

```
조건                      KuaiRec MRR@20    Δ       Foursq MRR@20    Δ     가설
─────────────────────────────────────────────────────────────────────────────
intact (RouteRec)            X.XXX          —          X.XXX          —
── Eval-only ────────────────────────────────────────────────────────────────
eval_zero                    X.XXX        -Δ₁         X.XXX         -Δ₁    H1
eval_shuffle                 X.XXX        -Δ₂         X.XXX         -Δ₂    H1
eval_global_permute          X.XXX        -Δ₃         X.XXX         -Δ₃    H1
── Train-time ───────────────────────────────────────────────────────────────
hidden_only                  X.XXX        -Δ₄         X.XXX         -Δ₄    H2
both_zero                    X.XXX        -Δ₅         X.XXX         -Δ₅    H3
train_zero                   X.XXX        -Δ₆         X.XXX         -Δ₆    H3
```

### Figure: Family Contribution (Supplement / Sec. 4.3)

`eval_family_{memory,focus,tempo,exposure}`의 Δ MRR@20 막대그래프.  
데이터셋별로 어떤 behavioral axis가 더 중요한지 시각화.

### 핵심 해석 시나리오

**이상적인 결과 (논문 주장 강하게 지지):**
```
intact >> eval_zero ≈ hidden_only >> both_zero > train_zero
```
- `eval_zero` ≈ `hidden_only` → "cue 없는 것과 hidden만 있는 것이 비슷하게 나쁘다" → H1 + H2 동시 지지
- `both_zero` < `train_zero` → "eval 때 cue 있으면 partial 회복" → H3 지지

**해석이 복잡해지는 경우:**
- `hidden_only` ≈ `intact` → "hidden state가 이미 cue 정보를 포함" → C2 약화, 논문 re-framing 필요
- `both_zero` ≈ `intact` → "MoE 구조만으로도 충분" → MoE 구조 contribution이 주라는 다른 story로 전환

---

## 5. 코드 구조

```
exp_cue_perturb/
│
├── eval_perturb.py      ← 그룹 A 실행
│   ├── run_p0()         └─ --auto-p0 시 P0 자동 학습 + ckpt 저장
│   ├── find_p0_checkpoint()   summary CSV → result JSON → ckpt path
│   ├── build_base_cfg()       P0 payload에서 arch params 복원
│   └── run_condition()        run_checkpoint_evaluation 호출
│
├── train_perturb.py     ← 그룹 B 실행
│   ├── get_p0_best_params()   P0 best lr/wd 읽기 (없으면 fallback)
│   └── build_train_cmd()      hyperopt_tune.py --max-evals 1 명령 구성
│
├── collect_results.py   ← 두 CSV 병합 → 논문용 통합 테이블
│
├── run_eval_perturb.sh  ← eval_perturb.py 런처
├── run_train_perturb.sh ← train_perturb.py 런처
├── run_all_perturb.sh   ← 전체 실행 (eval → train → collect)
└── demo_dry_run.sh      ← 명령어/조건 확인 (GPU 없이 실행 가능)
```

**configs 추가:**
- `experiments/configs/tune_foursq_cikm.yaml` — foursquare 전용 CIKM config (topk [1,5,10,20])
- `tune_kuai_cikm.yaml`, `tune_lfm_cikm.yaml` — topk에 1 추가

### 구현 메모

**eval-only perturbation이 가능한 이유:**  
`feature_perturb_mode/apply/family`는 모델 `__init__`에서 cfg로 초기화되는 Python instance attribute다. `torch.load(checkpoint)` → `model.load_state_dict(state)` 시 이 attribute들은 덮어씌워지지 않는다. 따라서 cfg만 달리해서 모델을 새로 초기화하면 학습된 가중치는 그대로이고 perturbation 동작만 바뀐다.

**hidden_only / feature_only 재학습:**  
`router_use_feature=False`로 바뀌면 router의 feature projection layer가 없어져 모델 파라미터 shape이 달라진다. P0 checkpoint와 state_dict key/shape 불일치 → 재학습 필수. TRAIN_CONDITIONS의 `new_arch=True` 플래그로 명시.

**max-evals=1의 의미:**  
`++search.learning_rate=<lr>` 형태로 scalar를 넘기면 hyperopt는 이를 singleton (fixed) 으로 취급하여 sampling하지 않는다. 단 1회 학습만 실행 = "P0 best hparam에서 조건만 바꾼 controlled experiment".

---

## 6. 실행 방법

### 사전 조건

```bash
# P0 결과 확인 (있으면 ckpt 자동 탐색)
cat experiments/run/CIKM/results/main_routerec_summary.csv
```

### 권장 실행 순서 (GPU 2개)

```bash
cd experiments/run/CIKM/exp_cue_perturb

# P0 checkpoint 있을 때
bash run_all_perturb.sh 0 1

# P0 checkpoint 없을 때 (eval step에서 자동 P0 학습)
bash run_all_perturb.sh 0 1 --auto-p0
```

### 단계별 / 부분 실행

```bash
# eval 조건만 (P0 ckpt 있을 때, ~1.5h)
bash run_eval_perturb.sh 0

# eval 조건만 (P0 없음, 자동 학습 포함)
python eval_perturb.py --gpu 0 --auto-p0

# checkpoint 직접 지정
python eval_perturb.py --gpu 0 \
  --checkpoint /path/to/KuaiRec_best.pth \
  --intact-mrr20 0.1234

# train: 핵심 조건만 (hidden_only + both_zero, ~13h)
python train_perturb.py --gpus 0 --conditions hidden_only both_zero

# 결과 취합
python collect_results.py
```

### dry run (GPU 없이 확인)

```bash
bash demo_dry_run.sh
```

---

## 7. 데이터셋 선정 이유

| 데이터셋 | 선정 이유 | train 규모 | 추정 학습 시간/조건 |
|---------|----------|-----------|-------------------|
| **KuaiRec** | 메인 full dataset, behavioral diversity 충분 (8.9K items, 3.2M train) | 3.2M | ~5-6시간 |
| **foursquare** | 체크인 데이터 (tempo/focus 패턴 강함), 빠른 학습 가능 (30.6K items, 105K train) | 105K | ~45분 |
| ~~lastfm~~ | 학습 ~30분/epoch × 40 epochs = 20h/조건 → 5조건 100h → 마감 전 불가 | 547K items | ~20시간 |

foursquare config: `tune_foursq_cikm.yaml` (이 실험에서 새로 생성)

---

## 8. 예상 소요 시간

| 단계 | KuaiRec | foursquare |
|------|---------|-----------|
| eval_perturb (9 조건) | ~1시간 | ~20분 |
| hidden_only (새 학습) | ~5-6시간 | ~45분 |
| feature_only (새 학습) | ~5-6시간 | ~45분 |
| train_zero | ~5-6시간 | ~45분 |
| both_zero | ~5-6시간 | ~45분 |
| train_shuffle | ~5-6시간 | ~45분 |
| **합계** | **~26-31시간** | **~4.5시간** |

GPU 2개 병렬: **~26-31시간** (KuaiRec train이 bottleneck)

**최소 논문 요건 (핵심 조건만):**
- `eval_zero`, `eval_shuffle`, `hidden_only`, `both_zero` → ~13-15시간

---

## 9. 결과 파일

| 파일 | 설명 |
|------|------|
| `results/cue_perturb_eval_summary.csv` | 그룹 A 상세 (hit/ndcg/mrr @1/5/10/20, delta_mrr20) |
| `results/cue_perturb_train_summary.csv` | 그룹 B 상세 (valid/test, elapsed, status) |
| `results/cue_perturb_summary.csv` | 두 그룹 통합 논문용 (데이터셋별 병렬 컬럼) |
| `logs/cue_perturb_eval/p0_<ds>.log` | --auto-p0 시 P0 학습 로그 |
| `logs/cue_perturb_train/<ds>_<cond>.log` | 그룹 B 각 조건 학습 로그 |

---

## 10. 주의사항

- foursquare에는 item 컬럼(category)이 없다. `build_base_cfg`에서 자동으로 `load_col` 제외.
- eval 조건은 P0 checkpoint가 필수다. `--auto-p0` 플래그로 자동 학습 가능.
- `train_shuffle`은 보조 evidence이므로 시간 제약 시 생략해도 논문 최소 요건 충족.
- `collect_results.py`는 멱등하다 (부분 완료 상태에서도 실행 가능).
- train 조건에서 status가 `completed` (mrr=0)이면 결과 JSON은 있으나 metric 파싱 실패 → log 확인 필요.
