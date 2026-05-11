# CIKM 2026 RouteRec 제출 로드맵

> 마지막 업데이트: 2026-05-08  
> 대상: CIKM 2026 Full Research Paper  
> 기준 논문: `writing/RouteRec__Behavior_Guided_Expert_Routing_for_Sequential_Recommendation.pdf`  
> 핵심 데이터: `Datasets/processed/final_dataset/`  
> 현재 판단: **논문 아이디어는 살리고, 실험 설계와 지면 구성을 CIKM 기준으로 다시 조여야 한다.**

---

## 0. 먼저 읽을 결론

CIKM full paper로 가려면 RouteRec을 "새 MoE 모델"로만 밀면 약하다. 더 강한 포지션은 다음이다.

> Sequential recommendation에서 MoE의 핵심 질문은 expert를 더 많이 두는 것이 아니라, **expert routing을 어떤 behavioral signal로 제어할 것인가**이다. RouteRec은 raw interaction logs에서 반복적으로 나타나는 behavioral heterogeneity axis를 cue로 만들고, 이를 separate control path에서 coarse-to-fine expert routing에 사용한다.

현재 RecSys 제출본의 story는 좋다. 특히 "same computation path for different sessions"에서 "what should guide routing?"으로 넘어가는 흐름은 유지해야 한다. 대신 CIKM에서는 아래 네 가지를 반드시 보강해야 한다.

1. **분량/형식**: 공식 full paper 마감은 2026-05-23 AoE이고, 10쪽 제한이다. 3주가 아니라 사실상 15일 남았다.
2. **Full dataset 결과**: KuaiRec/LastFM sampled dagger를 없애고 `final_dataset` 기준으로 Table 2를 다시 만든다.
3. **Causal controls**: cue zero/shuffle, hidden-only router, capacity-matched SASRec-wide로 "behavioral cue routing 때문"임을 보여준다.
4. **Motivation evidence**: intro에 raw-log behavioral diversity evidence를 넣어 "왜 routing demand가 있는가"를 데이터로 먼저 보여준다.

최소 accept 가능한 CIKM 패키지는 다음이다.

| 필요도 | 들어가야 하는 결과 | 논문 위치 |
|---|---|---|
| P0 | full `final_dataset` main table | Sec. 4.2 |
| P0 | cue perturbation: intact vs zero/shuffle/global-permute | Sec. 4.3 |
| P0 | capacity control: SASRec, SASRec-wide, hidden-MoE, RouteRec | Sec. 4.4 |
| P1 | behavioral routing demand / raw-log axes | Intro + Sec. 4.5 |
| P1 | 3-seed check on key datasets | compact main note or supplement |
| P2 | routing profile / cue-route correlation | Sec. 4.5 or supplement |

---

## 1. 공식 CIKM 제약

CIKM 2026 공식 페이지 기준이다.

| 항목 | 정확한 정보 |
|---|---|
| Abstract deadline | **2026-05-16 23:59 AoE** |
| Full paper deadline | **2026-05-23 23:59 AoE** |
| Notification | 2026-08-07 |
| Camera-ready | 2026-08-20 |
| Page limit | **10 pages including figures, tables, appendices + up to 2 pages references** |
| Review | double-blind |
| GenAI disclosure | required, placed before references |
| Preprint/arXiv | non-anonymized version must be disclosed in EasyChair if it exists |
| 요구 톤 | innovative, significant, reproducible research |

중요한 수정:

- 기존 로드맵의 `deadline ~May 28-30`은 full research paper에는 맞지 않는다.
- 기존 로드맵의 `9 pages + references`도 현재 공식 정보와 다르다. CIKM 2026 full research는 10쪽 + reference 최대 2쪽이다.
- Appendix도 10쪽 안에 포함된다. 따라서 RecSys 16쪽 구조를 그대로 줄이는 방식은 위험하다. 본문에 필요한 evidence만 남기고, full metric grid와 상세 preprocessing은 anonymized artifact 또는 supplementary material로 분리해야 한다.
- GenAI Usage Disclosure section이 필요하다. 이 섹션은 references 직전에 둔다.
- arXiv/website 등에 non-anonymized version이 있거나 올릴 계획이면 EasyChair에 disclosure해야 한다. 공식 policy는 CIKM submission과 archival version의 title/abstract가 충분히 달라야 desk reject risk가 낮아진다고 안내한다.

CIKM fit:

- CIKM 2026 topics에는 recommender systems, neural recommendation, sequential/time-series data, user behavior analysis, evaluation/reproducibility가 포함된다.
- CIKM 2025 accepted full paper 목록에도 sequential/session-based recommendation, MoE recommendation, streaming recommendation, LLM recommendation 등 관련 논문이 다수 있다.
- 최근 CIKM sequential recommendation paper들은 보통 `RQs -> setup -> overall performance -> ablation -> further discussion/sensitivity` 구조를 쓴다. RouteRec도 이 형식으로 실험 섹션을 재배치하는 편이 안전하다.

---

## 2. 6회 반복 점검 로그

요청한 "읽고 평가하고 다시 보완" 과정을 실제 의사결정 단위로 나누면 아래처럼 정리된다.

| Pass | 본 것 | 판단 | 로드맵 반영 |
|---|---|---|---|
| 1 | RecSys PDF | story는 좋지만 sampled datasets, capacity control, cue causality가 약함 | P0/P1 실험을 causal proof 중심으로 재배치 |
| 2 | 기존 `CIKM_roadmap.md` | 큰 방향은 맞지만 공식 마감/분량, 일부 경로, eval protocol이 부정확 | 공식 일정/페이지/경로 수정 |
| 3 | `final_dataset` 실제 파일/summary | KuaiRec/LastFM full 구축 완료. valid/test row 수가 기존 로드맵과 일부 다름 | dataset table을 실제 v4 split 기준으로 교체 |
| 4 | 실험 코드 | `feature_perturb_mode`는 이미 구현됨. 단, 기존 ablation common은 `feature_added_v4`와 sampled dataset을 hardcode | CIKM용 fork 필요. post-hoc intervention 우선 |
| 5 | CIKM accepted paper 구성 | RQ 기반 experiments와 clear ablation이 중요 | Sec. 4를 RQ1-RQ4로 재구성 |
| 6 | accept 관점 재검토 | "더 좋은 모델"보다 "왜 behavior-guided routing이어야 하는가"를 증명해야 함 | title/contribution/intro/experiment mapping 재작성 |

---

## 3. 현재 논문 진단

### 유지할 강점

- **문제 설정**: 다른 session behavior가 같은 computation path를 거친다는 문제 제기가 직관적이다.
- **routing-design framing**: MoE를 단순히 붙이는 것이 아니라 "what should guide expert selection?"로 framing한 점이 novelty다.
- **method decomposition**: behavioral cue construction, multi-scope routing, hierarchical sparse allocation의 세 축이 명확하다.
- **routing profile figure**: repeat-heavy, fast-tempo, narrow-focus subset에서 family routing weight가 달라지는 그림은 설득력이 있다.

### CIKM에서 찔릴 약점

| 약점 | 왜 위험한가 | 보완 |
|---|---|---|
| RecSys 제출본이 16쪽 | CIKM 10쪽 제한. desk reject 경험이 이미 있음 | page budget 먼저 확정 |
| KuaiRec/LastFM sampled subset | "full benchmark가 아닌 cherry-picked subset인가?" 의심 | `final_dataset`으로 main table 교체 |
| capacity control 없음 | MoE라서 parameter가 많아 이긴 것처럼 보임 | SASRec-wide / extra-FFN / hidden-MoE |
| cue causality 부족 | hand-crafted feature engineering처럼 보일 수 있음 | eval-time cue zero/shuffle/global-permute |
| single seed 중심 | margin이 작은 dataset에서 noise 의심 | key datasets 3 seeds |
| ML-1M negative result | 설명 없으면 "not robust"로 읽힘 | low routing demand / stable shared-path suitability로 해석 |
| eval protocol ambiguity | sampled/full ranking이 섞이면 치명적 | candidate protocol을 하나로 정하고 명시 |

---

## 4. 제목과 핵심 주장

### 추천 제목

```
RouteRec: Behavior-Guided Expert Routing for Sequential Recommendation
```

이유:

- "Sparse Routing"은 구현 디테일처럼 들린다.
- "Expert Routing"은 논문 기여가 routing design이라는 점을 바로 보여준다.
- CIKM 독자에게 MoE novelty를 설명하기 쉽다.

대안:

| 제목 | 평가 |
|---|---|
| `RouteRec: Behavior-Guided Expert Routing for Sequential Recommendation` | 가장 안전. 추천 |
| `RouteRec: Behavioral Cue Routing for Sequential Recommendation` | cue 중심이 선명하지만 MoE/expert 느낌이 약함 |
| `Routing by Behavior: Expert Allocation for Sequential Recommendation` | 일반적이고 RouteRec brand가 약함 |

### revised contribution

1. We formulate MoE sequential recommendation as a **routing-design problem**, asking what signal should control expert allocation.
2. We propose RouteRec, which builds lightweight behavioral cues from raw interaction logs and uses them through coarse-to-fine, hierarchical expert routing.
3. We show that the gains are not explained by extra capacity or feature injection alone, but by behaviorally aligned routing, using full-dataset evaluation, cue perturbations, and capacity-matched controls.

---

## 5. Introduction 보완 방향

현재 introduction 흐름은 유지한다.

```
Different sessions behave differently
-> shared transformation ignores this difference
-> MoE enables conditional computation
-> key question: what should guide routing?
-> behavioral cues are routing controls, not just predictive side features
-> RouteRec operationalizes this with multi-scope expert routing
```

보완해야 할 것은 "behavioral diversity is real"을 데이터로 보여주는 부분이다.

### 새 Figure 1 구성

기존 qualitative session 그림만 두지 말고, 오른쪽 또는 아래에 raw-log evidence를 붙인다.

추천:

- Fig. 1(a): 기존 three sessions cartoon 유지
- Fig. 1(b): dataset-level behavioral axes heatmap
  - Session volatility
  - Transition branching
  - Repeat variability
  - Context availability
- Fig. 1 caption 핵심: "These axes are computed from training logs before model training."

사용할 local source:

- `outputs/dataset_appendix_analysis/dataset_router_motivation.md`
- `outputs/dataset_appendix_analysis/dataset_appendix_report.md`

주의:

- 본문에서는 BRD를 복잡한 scalar로 크게 밀지 않는다.
- "correlation proves the model"처럼 쓰지 않는다. dataset 수가 6개라 통계적 주장은 약하다.
- 더 좋은 표현은 "routing headroom / routing demand를 설명하는 descriptive evidence"다.

추천 문단:

> Before introducing RouteRec, we first inspect the training logs and find that sessions differ recurrently along a small number of behavioral axes: temporal volatility, transition ambiguity, repetition variability, and the availability of earlier-session context. These axes are not model outputs; they are descriptive statistics computed before training. This motivates using behavioral cues as routing controls rather than treating all sessions as requiring the same transformation.

---

## 6. 논문 구조와 page budget

CIKM용 권장 구조:

| Section | 목표 분량 | 내용 |
|---|---:|---|
| Abstract | 0.25p | problem, routing question, RouteRec, 핵심 결과 |
| 1 Introduction | 1.1p | motivation + raw-log evidence + contributions |
| 2 Related Work | 0.8p | sequential rec, MoE rec, routing design |
| 3 Method | 2.1p | cue families, scopes, hierarchical router, objective |
| 4 Experiments | 4.6p | RQ1-RQ4 중심 |
| 5 Conclusion | 0.2p | 짧게 |
| Appendix inside PDF | 0p 또는 최대 0.5p | 가능하면 넣지 않음 |

### 실험 섹션 RQ 구조

| RQ | 질문 | 결과물 |
|---|---|---|
| RQ1 | Does RouteRec improve recommendation quality on full datasets? | main table |
| RQ2 | Are behavioral cues necessary for routing? | cue zero/shuffle + hidden-only |
| RQ3 | Is the gain just additional capacity? | SASRec-wide / hidden-MoE / RouteRec |
| RQ4 | When and how does behavior-guided routing help? | raw-log axes + routing profiles |

### Figure/Table budget

| Asset | 위치 | 포함 여부 |
|---|---|---|
| Fig. 1 motivation + raw-log axes | Intro | 반드시 |
| Fig. 2 architecture | Method | 반드시 |
| Table 1 dataset summary | Setup | compact로 |
| Table 2 main results | RQ1 | NDCG@10/MRR@20 중심, full grid는 artifact |
| Fig. 3 cue perturb/routing control | RQ2 | 반드시 |
| Table 3 capacity + seed | RQ3 | 반드시 |
| Fig. 4 routing profiles or BRD scatter | RQ4 | 하나만 main, 나머지 artifact |

---

## 7. 데이터셋 ground truth

모든 CIKM 실험은 아래 경로만 사용한다.

```text
/workspace/FeaturedMoE/Datasets/processed/final_dataset/<dataset>/<dataset>.{train,valid,test}.inter
```

Hydra override:

```bash
cd /workspace/FeaturedMoE/experiments
/venv/FMoE/bin/python recbole_train.py \
  model=<model> dataset=<dataset> feature_mode=final \
  epochs=100 gpu_id=0
```

`feature_mode=final`은 다음 파일을 사용한다.

```text
/workspace/FeaturedMoE/experiments/configs/feature_mode/final.yaml
```

현재 `final.yaml`:

```yaml
data_path: ${dataset_root}/final_dataset
```

### 실제 final_dataset 통계

아래 값은 `*.v4_split_summary.json`과 `*.session_split_summary.json` 기준이다.

| Dataset | Total rows | Train rows | Valid rows | Test rows | Sessions | Train sessions | Items |
|---|---:|---:|---:|---:|---:|---:|---:|
| KuaiRec | 3,862,479 | 3,200,980 | 384,026 | 277,473 | 209,312 | 146,518 | 8,966 |
| lastfm | 15,105,785 | 11,043,519 | 2,125,564 | 1,936,702 | 602,977 | 422,083 | 547,913 |
| foursquare | 136,544 | 105,683 | 15,638 | 15,223 | 25,369 | 17,758 | 30,588 |
| movielens1m | 575,157 | 406,633 | 89,000 | 79,524 | 14,539 | 10,177 | 3,883 |
| retail_rocket | 790,051 | 585,386 | 105,278 | 99,387 | 153,092 | 107,164 | 417,053 |
| beauty | 30,249 | 22,529 | 4,252 | 3,468 | 4,243 | 2,970 | 3,625 |

중요:

- 기존 RecSys table의 `KuaiRec†`, `LastFM†` 숫자와 CIKM 숫자를 절대 섞지 않는다.
- `lastfm`과 `retail_rocket`은 item 수가 매우 크다. training 중 eval은 sampled/masked eval로 돌릴 수 있지만, paper-facing final score protocol은 명시해야 한다.
- `real_final_ablation/common.py`를 그대로 쓰면 안 된다. 이 파일은 `DEFAULT_FEATURE_DATA_ROOT=feature_added_v4`, sampled dataset names, `feature_mode=full_v4`를 hardcode한다. CIKM용으로 fork해서 `feature_mode=final`, `data_path=.../final_dataset`, dataset 이름을 바꿔야 한다.

---

## 8. Evaluation protocol 결정

이 부분은 reviewer trust에 직접 연결된다.

현재 코드에는 large item set에서 `eval_sampling.auto_full_threshold`를 넘으면 precomputed/popularity negative mask를 쓰는 경로가 있다. 기본 config의 threshold는 `20,000`이고, `recbole_train.py` 내부 fallback default는 `500,000`이다.

CIKM 제출용 권장 원칙:

1. **paper main table의 candidate protocol을 하나로 고정한다.**
2. 가능하면 final checkpoint reporting은 full-sort seen-target evaluation으로 맞춘다.
3. full-sort가 현실적으로 불가능한 dataset이 있으면 "fixed candidate evaluation"으로 명시하고, 그 dataset만 표시한다.

추천 실행 전략:

| Dataset | 추천 reporting eval | 이유 |
|---|---|---|
| beauty, ML-1M, KuaiRec | full-sort | item 수 작음 |
| foursquare | full-sort 우선 | 30K items라 가능성이 높음 |
| retail_rocket | final full-sort 시도, 실패 시 fixed 3000 negatives | 417K items |
| lastfm | final full-sort 시도, 실패 시 fixed 3000 negatives + strong disclosure | 548K items |

최종 논문 문장:

- 좋은 경우: "All reported main results use full ranking over training-seen items."
- 차선 경우: "For LastFM and Retail Rocket, due to very large item vocabularies, we use a fixed candidate protocol with precomputed negatives shared by all methods; all other datasets use full ranking."  
  이 경우 main table footnote가 반드시 필요하다.

실험 명령 예시:

```bash
# final full-sort reporting 시도
/venv/FMoE/bin/python recbole_train.py \
  model=featured_moe_n3 dataset=lastfm feature_mode=final \
  eval_sampling.auto_full_threshold=1000000000 \
  eval_batch_size=64 epochs=100 gpu_id=0
```

---

## 9. 핵심 실험 계획

### P0-0. Protocol smoke test

목표: 모든 dataset/model 조합이 `final_dataset`을 제대로 읽고, old sampled path를 보지 않는지 확인한다.

폴더:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/00_protocol_smoke/
```

체크:

- `dataset=<final dataset name>`
- `feature_mode=final`
- log에 `data_path=.../Datasets/processed/final_dataset`
- valid/test session overlap 0
- unseen target accounting 정상
- RouteRec `feature_meta_v3` loaded

최소 smoke matrix:

| Dataset | Models |
|---|---|
| KuaiRec | SASRec, FDSA, RouteRec |
| lastfm | SASRec, RouteRec |
| retail_rocket | SASRec, RouteRec |
| beauty | SASRec, RouteRec |

---

### P0-1. Full dataset main results

목표: RecSys Table 2를 CIKM용 full dataset table로 교체한다.

Models:

- Shared backbone: GRU4Rec, SASRec, TiSASRec
- Strong sequential: FEARec, DuoRec, BSARec
- Side-info / expert: FDSA, DIF-SR, FAME
- Ours: RouteRec (`featured_moe_n3`)

Datasets:

```text
KuaiRec, lastfm, foursquare, movielens1m, retail_rocket, beauty
```

우선순위:

| Priority | Dataset | 이유 |
|---|---|---|
| 1 | KuaiRec | main motivation과 routing profile 핵심 |
| 2 | lastfm | full dataset 전환의 가장 큰 변화 |
| 3 | retail_rocket | sparse/session context 한계 설명에 중요 |
| 4 | movielens1m | 약한 케이스를 정직하게 설명 |
| 5 | foursquare | 중간 규모, POI domain |
| 6 | beauty | 작아서 variance check 중요 |

결과 선택 규칙:

- 모든 모델은 같은 selection rule 사용: validation 9 metrics 평균 또는 primary `NDCG@10`.
- paper에서는 하나만 고정해서 명시한다.
- 추천: **mean of HR/NDCG/MRR at 5/10/20 on validation**. 기존 paper와 연속성이 있다.

Table 2 구성:

- main paper에는 HR@10, NDCG@10, MRR@20만 싣는다.
- full k={5,10,20} grid는 artifact/supplement로 뺀다.
- average rank는 유지하되, rank만으로 과장하지 않는다.

성공 조건:

- KuaiRec full에서 RouteRec > strongest baseline on NDCG@10/MRR@20.
- lastfm full에서 absolute margin이 작아도 average rank가 높거나 cue perturb/capacity evidence가 좋으면 유지 가능.
- ML-1M이 약해도 괜찮다. 대신 low routing demand explanation이 필요하다.

---

### P0-2. Cue perturbation / routing causality

목표: "behavioral cue alignment가 실제 원인"임을 보인다.

이미 구현된 모델 옵션:

- `feature_perturb_mode=zero`
- `feature_perturb_mode=shuffle`
- `feature_perturb_mode=global_permute`
- `feature_perturb_mode=batch_permute`
- `feature_perturb_mode=family_permute`
- `feature_perturb_apply=eval`
- `feature_perturb_family=[tempo|focus|memory|exposure]`

중요: 이 실험은 retraining보다 **trained checkpoint post-hoc eval intervention**을 우선한다. 그래야 "같은 모델, 같은 parameter, cue만 깨뜨림"이 된다.

폴더:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/01_cue_perturb/
```

실험 rows:

| ID | Setting | 목적 |
|---|---|---|
| CP-00 | intact RouteRec | 기준 |
| CP-01 | eval zero all cues | cue absence |
| CP-02 | eval shuffle all cues | cross-sample alignment 파괴 |
| CP-03 | eval global_permute all cues | token-position alignment까지 파괴 |
| CP-04 | eval shuffle Memory | repeat-heavy sensitivity |
| CP-05 | eval shuffle Tempo | fast-tempo sensitivity |
| CP-06 | eval shuffle Focus | narrow-focus sensitivity |
| CP-07 | eval shuffle Exposure | popularity/exposure sensitivity |

Datasets:

- 필수: KuaiRec, lastfm
- 권장: foursquare
- 시간이 남으면: retail_rocket

측정:

- ΔNDCG@10, ΔMRR@20 vs intact
- route-change score: routing family top-1 distribution이 얼마나 바뀌는지
- family-specific perturb는 해당 behavioral subset에서 더 큰 drop이 나오는지

논문에서 기대되는 해석:

- zero/shuffle가 intact보다 떨어지면: cue가 단순 decoration이 아니라 routing control임.
- hidden-only보다 behavior-guided가 높으면: hidden representation만으로는 충분하지 않음.
- family perturb가 관련 subset에서 더 크게 떨어지면: cue family semantics 강화.

주의:

- `shuffle`과 `global_permute`는 의미가 다르다. 논문에서는 용어를 명확히 쓴다.
- `batch_permute`는 intra-sequence position control이라 main에는 넣지 않아도 된다.

---

### P0-3. Capacity-matched baseline

목표: "RouteRec은 parameter가 많아서 이긴다"를 직접 반박한다.

폴더:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/02_capacity/
```

비교:

| Model | 설명 |
|---|---|
| SASRec | standard shared-path baseline |
| SASRec-wide | `inner_size` 또는 hidden width를 늘려 RouteRec total params와 match |
| Extra-FFN / extra-attn SASRec | 가능하면 기존 scope ablation의 extra-attn control 재사용 |
| Hidden-MoE | routing은 있지만 behavioral cue 없음 |
| RouteRec | behavior-guided expert routing |

Datasets:

- 필수: KuaiRec, lastfm
- 권장: movielens1m

실행 원칙:

1. RouteRec best config에서 total params와 active params를 로그로 추출한다.
2. SASRec-wide의 `inner_size` 후보를 잡는다: 512, 768, 1024, 1536, 2048.
3. total params가 RouteRec과 가장 가까운 setting을 선택한다.
4. training budget과 selection rule은 main table과 동일하게 둔다.

논문 table:

| Dataset | SASRec | SASRec-wide | Hidden-MoE | RouteRec |
|---|---:|---:|---:|---:|
| KuaiRec NDCG@10 |  |  |  |  |
| lastfm NDCG@10 |  |  |  |  |
| Params |  |  |  |  |

좋은 결과 패턴:

```text
SASRec-wide > SASRec, but SASRec-wide < RouteRec
Hidden-MoE > SASRec, but Hidden-MoE < RouteRec
```

이 패턴이면 "capacity + MoE structure만으로는 부족하고, behavioral routing control이 필요하다"가 된다.

---

### P1-1. Multi-seed robustness

목표: 작은 margin과 random seed 의심을 줄인다.

폴더:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/03_seed/
```

Seeds:

```text
1, 2, 3
```

최소 matrix:

| Dataset | Models |
|---|---|
| KuaiRec | SASRec, strongest baseline, RouteRec |
| lastfm | SASRec, strongest baseline, RouteRec |
| movielens1m | SASRec, strongest baseline, RouteRec |
| beauty | strongest baseline, RouteRec |

왜 ML-1M을 넣는가:

- RouteRec이 약한 케이스를 숨기지 않는 편이 신뢰도를 높인다.
- "routing demand가 낮으면 gain도 작다"는 논문 해석과 연결된다.

논문 반영:

- main에는 "3-seed mean ± std on key datasets confirms the ranking" 한 문장 + compact table.
- full seed table은 artifact.

---

### P1-2. Behavioral routing demand analysis

목표: dataset별 gain 차이를 설명한다.

폴더:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/04_behavior_demand/
```

원칙:

- training split만 사용한다.
- RouteRec 내부 routing weight를 쓰지 않는 raw-log 지표로 먼저 계산한다.
- 지표를 "모델이 이길 dataset을 맞추는 score"처럼 쓰지 않는다.

사용할 axes:

| Axis | Metric | Cue family link |
|---|---|---|
| Temporal volatility | session length CV, timing irregularity | Tempo |
| Transition ambiguity | normalized transition branching | Focus |
| Memory regime | repeat intensity / repeat variability | Memory |
| Exposure regime | popularity concentration | Exposure |
| Context availability | users with >=2/5 sessions, sessions per user | macro/mid support |

main paper에 넣을 방식:

- Intro Fig. 1(b): axes heatmap.
- RQ4: RouteRec gains are largest when routing headroom exists, but not every axis predicts gains alone.
- ML-1M: long, stable, preference-driven sequences -> strong shared-path baselines already work.
- Beauty/Retail Rocket: context availability가 낮아 macro routing headroom이 제한됨.

주의:

- correlation table은 appendix/artifact에 둔다.
- n=6이므로 p-value류 주장은 하지 않는다.

---

### P1-3. Routing profiles and cue-route alignment

목표: router가 behavioral subsets에 다르게 반응함을 보여준다.

폴더:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/05_routing_profiles/
```

분석:

1. Full KuaiRec checkpoint에서 routing family weights 추출.
2. subsets:
   - repeat-heavy
   - fast-tempo
   - narrow-focus
   - head/exposure-heavy
3. 각 subset의 macro/mid/micro family share를 시각화.
4. 가능하면 cue value와 routing family weight의 Spearman correlation heatmap 추가.

논문에서:

- main에는 routing profile 하나만 넣는다.
- cue-route correlation heatmap은 공간이 있으면 RQ4, 없으면 artifact.

---

### P2. Optional experiments

시간이 남을 때만 한다.

| 실험 | 가치 | 비용 | 판단 |
|---|---|---:|---|
| Expert family masking | family가 실제 performance에 기여함을 직접 보여줌 | 코드 수정 필요 | optional |
| Efficiency table | sparse routing overhead 방어 | 낮음 | 가능하면 포함 |
| Modern baseline 추가 | CIKM reviewer 눈높이 대응 | 높음 | 기존 코드 없으면 무리하지 않음 |
| Full hyperparameter sensitivity | 안정성 보강 | 중간 | artifact로만 |

---

## 10. CIKM용 실험 폴더 구조

새로 만들 구조:

```text
/workspace/FeaturedMoE/experiments/run/CIKM/
├── README.md
├── common_cikm.py
├── config_inventory.md
├── protocol_checklist.md
│
├── 00_protocol_smoke/
│   ├── smoke_final_dataset.py
│   └── smoke_final_dataset.sh
│
├── 01_main_full/
│   ├── run_main_baselines.py
│   ├── run_main_routerec.py
│   ├── main_queue.sh
│   └── aggregate_main_table.py
│
├── 02_cue_perturb/
│   ├── eval_cue_interventions.py
│   ├── aggregate_cue_perturb.py
│   └── plot_cue_perturb.py
│
├── 03_capacity/
│   ├── count_params.py
│   ├── run_sasrec_wide.py
│   ├── run_hidden_moe.py
│   └── aggregate_capacity.py
│
├── 04_seed/
│   ├── run_seed_matrix.py
│   └── aggregate_seed.py
│
├── 05_behavior_demand/
│   ├── compute_raw_axes.py
│   ├── build_motivation_figure.py
│   └── join_gain_axes.py
│
└── 06_routing_profiles/
    ├── extract_routing_weights.py
    ├── build_behavior_subsets.py
    └── plot_routing_profiles.py
```

### 경로 재사용 주의

재사용 가능:

| 목적 | 기존 위치 | 주의 |
|---|---|---|
| q2 routing settings | `experiments/run/final_experiment/real_final_ablation/common.py` | 그대로 import하지 말고 CIKM fork |
| checkpoint intervention | `experiments/run/final_experiment/ablation/eval_checkpoint_interventions.py` | `build_eval_config_from_result_payload`가 old path 사용하므로 수정 |
| RouteRec best configs | `experiments/run/final_experiment/ablation/configs/base_candidates.csv` | sampled dataset name -> final dataset name mapping 필요 |
| baseline summaries | `experiments/run/baseline_first/docs/baseline_2_best_valid_test_tables_v4.md` | old feature_v4 기준 참고용만 |
| dataset motivation | `outputs/dataset_appendix_analysis/dataset_router_motivation.md` | final full dataset으로 재계산 권장 |

Dataset name mapping:

| RecSys/old | CIKM final |
|---|---|
| `KuaiRecLargeStrictPosV2_0.2` | `KuaiRec` |
| `lastfm0.03` | `lastfm` |
| `foursquare` | `foursquare` |
| `movielens1m` | `movielens1m` |
| `retail_rocket` | `retail_rocket` |
| `beauty` | `beauty` |

---

## 11. 15일 실행 일정

오늘이 2026-05-08이므로 full paper deadline까지 약 15일이다.

### May 8-9: protocol lock

- `experiments/run/CIKM/` 생성
- `common_cikm.py` 작성
- final dataset smoke tests
- eval protocol 결정: full-sort vs fixed candidate
- abstract draft 시작

산출물:

```text
experiments/run/CIKM/protocol_checklist.md
experiments/run/CIKM/00_protocol_smoke/smoke_report.md
```

### May 10-13: main results first wave

- KuaiRec full: all baselines + RouteRec
- lastfm full: long-running jobs 먼저 시작
- retail_rocket / movielens1m / beauty 병렬 실행
- failed jobs 즉시 재시작

중간 체크:

- May 13까지 KuaiRec RouteRec vs strongest baseline 방향이 나와야 한다.
- lastfm이 너무 오래 걸리면 training eval을 sampled로 두고 final full-sort reporting만 따로 시도한다.

### May 14-16: causal controls + abstract

- cue perturbation on KuaiRec/lastfm
- hidden-only / mixed router rerun
- capacity-matched SASRec-wide 시작
- abstract deadline: **May 16 AoE**

May 16 abstract에는 숫자가 완전히 없어도 된다. 대신 claim은 보수적으로:

> We introduce RouteRec, a behavior-guided expert routing framework for sequential recommendation, and evaluate whether behavioral cues provide a better routing control signal than hidden-state routing or capacity-matched shared backbones.

### May 17-19: capacity + seed + motivation figure

- capacity table 완성
- 3 seeds 최소 matrix 실행
- raw-log axes 재계산
- Fig. 1 motivation figure 생성
- Table 2 draft 작성

May 19 gate:

| 질문 | Yes이면 | No이면 |
|---|---|---|
| KuaiRec full에서 gain 유지? | main story 유지 | preprocessing/eval protocol 먼저 점검 |
| cue shuffle가 하락을 만드나? | RQ2 main | routing profile/hidden-only 중심으로 조정 |
| SASRec-wide < RouteRec? | RQ3 main | capacity table은 honest하게 쓰고 cue perturb 강조 |
| lastfm full이 끝났나? | main table 포함 | fixed candidate/full-sort final eval fallback |

### May 20-22: writing sprint

- Method 2쪽으로 압축
- Experiments RQ1-RQ4 작성
- Related work 최신화
- figure/table final
- double-blind, GenAI disclosure, page limit 확인

### May 23: submission

- PDF page count 확인
- author/affiliation 제거
- self-citation third person
- anonymous code/data link 확인
- EasyChair submission

---

## 12. 결과 패턴별 대응

| 결과 | 논문 전략 |
|---|---|
| Full KuaiRec/LastFM에서 RouteRec 우세 | main claim 강하게 유지 |
| Full lastfm gain 약함 | item vocabulary/long-history setting 설명, cue/capacity controls로 보완 |
| ML-1M에서 계속 약함 | low routing headroom 사례로 정직하게 설명 |
| cue zero/shuffle가 크게 하락 | RQ2를 핵심 evidence로 전면 배치 |
| cue perturb 효과가 작음 | hidden-only vs behavior-guided, routing profiles, capacity table을 더 강조 |
| SASRec-wide가 RouteRec에 근접 | "capacity helps, but does not explain cue alignment"로 톤 조정 |
| SASRec-wide가 RouteRec을 이김 | main claim 위험. RouteRec을 universal SOTA가 아니라 routing-demand model로 축소해야 함 |
| final full-sort eval이 너무 느림 | fixed candidate protocol을 명시하고 모든 methods에 동일 candidates 사용 |

---

## 13. Related Work 보완

CIKM 독자는 RecSys보다 IR/data mining/neural ranking 쪽도 많다. Related Work는 짧되 더 전략적으로 간다.

### 넣을 축

1. Sequential/session-based recommendation
   - GRU4Rec, SASRec, TiSASRec
   - contrastive/frequency-aware: DuoRec, FEARec, BSARec
   - side information: FDSA, DIF-SR
2. MoE in recommendation
   - MMoE/PLE/PEPNet style routing-design intuition
   - FAME / multimodal or multi-behavior MoE recommenders
3. Routing design beyond recommendation
   - task/dataset/language-aware routing examples
   - RouteRec의 차이: explicit task labels가 아니라 raw interaction behavior에서 cue를 만든다.
4. Evaluation/reproducibility in sequential recommendation
   - temporal split, leakage, sampled/full evaluation protocol 이슈를 짧게 언급

주의:

- 최신 LLM/large sequential recommender를 baseline으로 무리하게 추가하려고 하지 않는다.
- 대신 "our contribution is orthogonal to stronger sequence encoders: the routing control signal"이라고 정리한다.

---

## 14. 논문 문장 톤 가이드

피해야 할 문장:

- "RouteRec universally outperforms all baselines."
- "Behavioral cues explain user intent."
- "Our routing is interpretable."
- "BRD predicts performance."

추천 문장:

- "RouteRec is most useful when the data provides routing headroom: heterogeneous behavior and enough context for routing to act on."
- "The cue perturbation results test whether the cues act as routing controls rather than merely as auxiliary features."
- "The weaker ML-1M result is consistent with the lower need for behavior-dependent computation in long, stable preference sequences."
- "The dataset-level axes are descriptive training-log statistics, not model-derived scores."

---

## 15. 최종 accept checklist

### 제출 전 반드시

- [ ] CIKM official deadline/page limit 기준으로 PDF 10쪽 이하
- [ ] Abstract submitted by 2026-05-16 AoE
- [ ] Full paper submitted by 2026-05-23 AoE
- [ ] `KuaiRec†`, `LastFM†` dagger 제거
- [ ] Table 2가 전부 `final_dataset` 결과
- [ ] old `feature_added_v4` path가 paper-facing runs에 없음
- [ ] candidate/eval protocol이 paper에 명시됨
- [ ] cue zero/shuffle 결과 포함
- [ ] capacity-matched baseline 포함
- [ ] ML-1M 약점 설명 포함
- [ ] double-blind 처리
- [ ] GenAI Usage Disclosure section 추가
- [ ] non-anonymized preprint가 있으면 EasyChair disclosure 및 title/abstract re-identification 점검
- [ ] code/data availability 문구 anonymized 처리

### 있으면 accept 확률이 올라감

- [ ] 3-seed mean ± std
- [ ] raw-log motivation figure
- [ ] routing profile full KuaiRec figure
- [ ] cue-route Spearman heatmap
- [ ] efficiency/active parameter table
- [ ] leakage audit 문서

---

## 16. 바로 다음 액션

1. `experiments/run/CIKM/` 생성.
2. `real_final_ablation/common.py`를 복사하지 말고, `common_cikm.py`로 필요한 부분만 fork.
3. `feature_mode=final`, `data_path=/workspace/FeaturedMoE/Datasets/processed/final_dataset`, final dataset names를 hardcode 또는 config로 고정.
4. KuaiRec/lastfm smoke run.
5. lastfm과 retail_rocket long-running main runs를 가장 먼저 시작.
6. 동시에 Fig. 1용 raw-log axes figure를 만든다.

---

## 17. 참고한 외부 기준

- CIKM 2026 Full Research Papers: `https://cikm2026.diag.uniroma1.it/full-research-papers/`
- CIKM 2026 Submission Policies and Information: `https://cikm2026.diag.uniroma1.it/submission-policies-and-information/`
- CIKM 2026 Important Dates: `https://cikm2026.diag.uniroma1.it/`
- CIKM 2025 accepted papers list: `https://cikm2025.org/program/accepted-papers`
- Example CIKM sequential recommendation paper structure, PTSR CIKM 2024: `https://www.atailab.cn/seminar2024Fall/pdf/2024_CIKM_PTSR_%20Prefix-Target%20Graph-based%20Sequential%20Recommendation.pdf`
