# Dataset Preprocessing & Feature Addition

> 작성일: 2026-04-30 (역공학 기반 문서)  
> 기준 데이터: `Datasets/processed/feature_added_v4/`  
> 원본 preprocessing 코드는 유실. 이 문서는 processed dataset, feature_meta_v3.json, session_split_summary.json 에서 역공학한 내용이다.

---

## 1. 전체 파이프라인

```
Raw data
  └─ Step 1: Basic inter 생성 (sessionization + filtering)
       → Datasets/processed/basic/{dataset}/
  └─ Step 2: Feature 계산 + 정규화 (feature_added_v3 생성)
       → Datasets/processed/feature_added_v3/{dataset}/
  └─ Step 3: Train/Valid/Test split (feature_added_v3 → feature_added_v4)
       → Datasets/processed/feature_added_v4/{dataset}/
```

Step 3 코드는 현존: `experiments/tools/build_feature_v4_from_v3.py`  
Step 1, 2 코드는 유실 → 아래에 역공학 결과를 바탕으로 재구성 계획 포함

---

## 2. 각 데이터셋별 현황

| Dataset | Users | Sessions | Items | Rows | Timestamp unit |
|---|---:|---:|---:|---:|---|
| KuaiRecLargeStrictPosV2_0.2 | 1,122 | 24,458 | 6,477 | 287,411 | seconds (Unix) |
| lastfm0.03 | 130 | 25,089 | 52,510 | 470,408 | milliseconds |
| foursquare | (미확인) | - | - | - | - |
| movielens1m | (미확인) | - | - | - | - |
| amazon_beauty / beauty | (미확인) | - | - | - | - |
| retail_rocket | (미확인) | - | - | - | - |

---

## 3. KuaiRec 역공학

### 3.1 Raw Data

- 파일: `Datasets/raw/KuaiRec/KuaiRec 2.0/data/big_matrix.csv`
- 컬럼: `user_id, video_id, play_duration, video_duration, time, date, timestamp, watch_ratio`
- Raw rows: 11,761,458 (헤더 제외)
- Processed rows: 287,411 → **약 2.4% 유지** (강한 필터링)

### 3.2 Sessionization (역공학)

`session_id` 포맷: `{user_id}_s{session_num}_c{chunk_num}`

- `s{N}`: raw activity session index (time-gap-based split)
- `c{N}`: RecBole `MAX_ITEM_LIST_LENGTH=50` 에 의한 chunk split (c0~c35 존재)

**추론된 session boundary 조건:**
```
session gap threshold = 30분 (1800초) — 확정
```
- `session_id` 포맷 `{uid}_s{N}_c{chunk}` 에서 raw session 단위 경계가 30분 gap 기반임을 확인
- 처리된 raw session간 최솟값 gap: **930초** — 즉 30분(1800초) threshold보다 작은 gap은 모두 같은 session으로 묶임
- 30분 gap으로 시뮬레이션 시 처리 결과와 가장 잘 일치

**KuaiRec 특수 필터:**
```
watch_ratio >= 0.5  (positive implicit feedback 기준)
```
Raw 11.7M rows → watch_ratio ≥ 0.5 필터 후 8.0M rows (68.5%)

### 3.3 Filtering Rules (역공학) — 시뮬레이션으로 확인됨

처리된 session length 분포:
- 최솟값: **5** (p5도 5) → `min_session_length = 5` 확실
- 최댓값: 50 (`_c{chunk}` ID로 확인: MAX_ITEM_LIST_LENGTH=50 chunking)
- 중앙값: 8

**watch_ratio 필터 후보 시뮬레이션 결과 비교:**

| 필터 조건 | full sessions | full users | full items | 20% sample users | 20% sample sessions |
|---|---:|---:|---:|---:|---:|
| wr ≥ 0.5 (현재 확정) | 293,150 | 7,135 | 9,431 | ~1,427 | ~58,630 |
| wr > 1.0 | 195,007 | 6,862 | 8,329 | ~1,372 | ~39,001 |
| wr >= 1.0 | 195,046 | 6,862 | 8,330 | ~1,372 | ~39,009 |
| wr > user_mean | 210,413 | 6,916 | 8,537 | ~1,383 | ~42,083 |
| wr >= user_mean | 210,413 | 6,916 | 8,537 | ~1,383 | ~42,083 |
| wr > 0.5×user_mean | 297,730 | 7,146 | 9,494 | ~1,429 | ~59,546 |
| wr >= 0.7 | 256,630 | 7,088 | 9,056 | ~1,418 | ~51,326 |
| wr >= 0.5 | 293,150 | 7,135 | 9,431 | ~1,427 | ~58,630 |

**현재 processed (0.2) 데이터:** 24,458 sessions, 1,122 users, 6,477 items

**핵심 불일치:** `wr > 1.0` 또는 `wr > user_mean` 어느 조건으로도 20% sampling을 적용하면 **1,372~1,383 users**가 나옴 — 목표 1,122에 비해 22~23% 더 많음.
- 1,122 users가 정확히 20%가 되려면 전체 pool이 **5,610 users**여야 함
- 어떤 필터도 5,610 users를 생성하지 않음
- `wr > 1.5` (= 5,783 users → 20% ≈ 1,156) 가 가장 근접하나 여전히 불일치

**결론: watch_ratio 필터는 0.5가 확정이고, user sampling이 20% random이다. 단 random seed를 알 수 없어 exact 1,122 users 재현 불가.**

**full (wr≥0.5) vs 0.2 subset 비교:**

| 항목 | Full (wr≥0.5) | 0.2 Subset (현재 paper) | 비율 |
|---|---:|---:|---:|
| Users | 7,135 | 1,122 | 15.7% |
| Sessions | 293,150 | 24,458 | 8.3% |
| Items | 9,431 | 6,477 | 68.7% |
| Rows | 6,948,721* | 287,411 | 4.1% |
| Sessions/user (평균) | 41.1 | 21.8 | 53% |
| Items/session (평균) | ~23.7 | ~11.7 | 49% |

*rows는 wr≥0.5 filter 직후 기준 (k-core 전). k-core 후 행 수는 session수×평균길이로 추산.

→ full dataset은 규모가 훨씬 크고, user당 session 수도 더 많음 (더 active한 user pool).  
→ 0.2 subset은 user sampling으로 인해 전체 item의 68.7%만 포함 (희귀 아이템 다수 탈락).

**user 선택 패턴 분석 (raw interaction count별 선택률):**

| raw interaction 수 | 전체 users | 선택된 users | 선택률 |
|---|---:|---:|---:|
| < 500 | 1,279 | 27 | 2.1% |
| 500 ~ 1k | 720 | 71 | 9.9% |
| 1k ~ 1.5k | 862 | 157 | 18.2% |
| 1.5k ~ 2k | 1,121 | 243 | 21.7% |
| 2k ~ 2.5k | 1,527 | 294 | 19.3% |
| 2.5k ~ 3k | 1,090 | 218 | 20.0% |
| 3k ~ 4k | 501 | 95 | 19.0% |
| > 4k | 76 | 17 | 22.4% |

→ `<500` 구간 선택률 2.1%, `500~1k` 9.9% — raw interaction 적은 user가 k-core 단계에서 자연 탈락  
→ `1k+` 구간부터는 **약 19~21%로 균일** → k-core 통과한 7,135 users 중 20% random sampling이 맞음  
→ watch_ratio나 interaction count 기준 추가 filter는 없음

```
확정된 pipeline:
  1. watch_ratio >= 0.5 필터
  2. sessionization: 30분 inactivity gap
  3. min_session_length = 5
  4. iterative k-core x3: min_item_freq >= 3, min_session_len >= 5
     → 7,135 active users 생성 (raw 7,176 중 비활성 user 탈락)
  5. 20% random user sampling → 1,122명 선택
     (random seed 미상, 재현 불가)

full dataset 실험: step 5 생략, 7,135 users 전체 사용
```

**"KuaiRecLargeStrictPosV2_0.2" 이름 해석:**
- `Large` = big_matrix 사용 (small_matrix 아님, overlap 2.9%만)
- `StrictPos` = watch_ratio ≥ 0.5 positive feedback 필터
- `V2` = version 2
- `0.2` = **20% user random sampling** (전체 7,135 users → 1,122명 선택)

**KuaiRec item category:**
- `Datasets/raw/KuaiRec/KuaiRec 2.0/data/item_categories.csv` 에 존재
- 컬럼: `video_id, feat` (feat는 category list, e.g. `[8]` 또는 `[27, 9]`)
- 처리 시 첫 번째 category만 사용했을 것

### 3.4 Train/Valid/Test Split

`session_split_summary.json` 기준:
```
split_strategy: tail_stratified
train: 70%  → 17,120 sessions
valid: 15%  → 3,668 sessions  
test:  15%  → 3,670 sessions

split 방식: 전체 session을 session start time 순으로 정렬 →
  - train: 앞 70%
  - valid/test: 뒤 30%를 tail_stratified (repeat target 비율 균형 맞춤)
```

v4에서 additional re-split:
- valid+test를 합쳐서 session start time 기준 **50:50 재분할**
- valid/test의 non-target unseen item row 제거
- train은 그대로 복사

### 3.5 Item File

```
item_id:token   category:token
```
Category 정보 포함 (KuaiRec의 경우 video category). Feature 계산에 사용.

---

## 4. LastFM 역공학

### 4.1 Raw Data

- 파일: `Datasets/raw/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv`
- 컬럼: `user_id, timestamp, artist_id, artist_name, track_id, track_name`
- Raw rows: 8,081,856 (헤더 제외)
- Timestamp 형식: ISO 8601 (`2009-05-04T23:08:57Z`) → ms Unix timestamp로 변환했을 것
- **`lastfm0.03`** = 1K users 중 **3% 샘플 (130 users)**

### 4.2 Sessionization (역공학)

`session_id` 포맷: 정수 (`1, 2, 3, ...`) — 전 user 통합 순차 번호

**추론된 session boundary 조건:**
```
session gap threshold = 30분 (1,800,000ms) — KuaiRec과 동일 기준
```
- 처리된 데이터에서 session end → next session start gap: p25 = **1.2h**, median = **8.4h**
- intra-session duration 중앙값: **72분**, p90: **210분**
- session간 최솟값 gap이 0ms인 케이스 존재 → 연속 listening으로 새 session이 바로 시작되는 것 (gap threshold보다 큰 gap 후 재시작)
- 30분 inactivity gap이 music streaming 세션 기준으로 가장 자연스럽고 KuaiRec과 통일됨

### 4.3 Item 정의

LastFM에서 item = **track (song)**
- `track_id`를 item_id로 사용
- Category = **artist_id** (또는 artist_name)

### 4.4 Filtering Rules (역공학)

처리된 session length 분포:
- 최솟값: **5**, 최댓값: **50** (MAX_ITEM_LIST_LENGTH chunking)
- 중앙값: 13

```
확정된 pipeline:
  1. 30분 inactivity gap으로 session 분리
  2. min_session_length = 5  (확실)
  3. iterative k-core x3: min_item_freq >= 3, min_session_len >= 5
  4. random user sampling: 전체 ~1,000 users 중 3% = 30명 → 실제 130명
     (30명은 3%의 최솟값; 실제로는 seed 고정 random sample일 것)
```

**3% user sampling (130/~1000명):**
- `lastfm0.03`의 "0.03" = 3% user sampling
- full LastFM 1K dataset은 ~992명의 active user → 그 중 130명 선택
- 선택 기준: random seed 고정 sampling (seed 미상)
- **full dataset 재구성 시 sampling 없이 전체 사용 가능**

### 4.5 Train/Valid/Test Split

KuaiRec과 동일한 `tail_stratified` 방식:
```
train: 70%  → 17,562 sessions
valid: 15%  → 3,763 sessions
test:  15%  → 3,764 sessions
```

---

## 5. Feature 구조 (공통)

### 5.1 Feature 범위 (scope)

모든 feature는 **strict prefix rule** 적용:
```
feature(t) 계산 시, sorted_index < t 인 interaction 만 사용
→ target item 정보 미사용 (no leakage)
```

| Scope | 입력 범위 | 추출 정보 |
|---|---|---|
| **macro (mac5)** | 최근 5개 **prior sessions** | 세션 간 패턴 요약 (평균) |
| **macro (mac10)** | 최근 10개 **prior sessions** | 더 넓은 역사 요약 |
| **mid** | **현재 세션 전체** prefix | 현재 세션 내부 동태 |
| **micro (mic)** | **최근 5 interactions** | 직전 short-term 패턴 |

### 5.2 Feature 목록 (64개)

#### Macro scope (32개: mac5×16 + mac10×16)

| Feature | Family | 의미 |
|---|---|---|
| `mac5_ctx_valid_r` | Tempo | 활용 가능한 prior session 비율 (min(n_ctx,5)/5) |
| `mac5_gap_last` | Tempo | 직전 session과의 시간 간격 (log-normalized) |
| `mac5_pace_mean` | Tempo | 최근 5 session의 평균 pace (interaction/시간) |
| `mac5_pace_trend` | Tempo | pace 추세 (recent vs older, phi-normalized) |
| `mac5_theme_ent_mean` | Focus | category entropy 평균 (category 분산도) |
| `mac5_theme_top1_mean` | Focus | top category 집중도 평균 |
| `mac5_theme_repeat_r` | Focus | category 반복 비율 평균 |
| `mac5_theme_shift_r` | Focus | category 전환 비율 평균 |
| `mac5_repeat_mean` | Memory | item 반복 소비 비율 평균 |
| `mac5_adj_cat_overlap_mean` | Memory | 인접 session간 category overlap 평균 |
| `mac5_adj_item_overlap_mean` | Memory | 인접 session간 item overlap 평균 |
| `mac5_repeat_trend` | Memory | 반복 소비 추세 |
| `mac5_pop_mean` | Exposure | 평균 아이템 인기도 (log-normalized) |
| `mac5_pop_std_mean` | Exposure | 아이템 인기도 분산 평균 |
| `mac5_pop_ent_mean` | Exposure | 인기도 분포 entropy 평균 |
| `mac5_pop_trend` | Exposure | 인기도 추세 |
| *(mac10_* 동일 구조 × 10 sessions window)* | | |

#### Mid scope (16개)

| Feature | Family | 의미 |
|---|---|---|
| `mid_valid_r` | Tempo | 현재 세션에서 valid interaction 비율 |
| `mid_int_mean` | Tempo | 평균 interaction interval (seconds, log-norm) |
| `mid_int_std` | Tempo | interval 표준편차 |
| `mid_sess_age` | Tempo | 현재 세션 시작부터 경과 시간 |
| `mid_cat_ent` | Focus | 현재 세션 category entropy |
| `mid_cat_top1` | Focus | top category 비율 |
| `mid_cat_switch_r` | Focus | category switch 비율 |
| `mid_cat_uniq_r` | Focus | unique category 비율 |
| `mid_item_uniq_r` | Memory | unique item 비율 |
| `mid_repeat_r` | Memory | item 반복 비율 |
| `mid_novel_r` | Memory | 새 아이템 비율 (not seen in history) |
| `mid_max_run_i` | Memory | 최장 consecutive 같은 item run |
| `mid_pop_mean` | Exposure | 현재 세션 평균 아이템 인기도 |
| `mid_pop_std` | Exposure | 인기도 표준편차 |
| `mid_pop_ent` | Exposure | 인기도 entropy |
| `mid_pop_trend` | Exposure | 인기도 추세 (recent vs older in session) |

**중요**: `mid` feature는 `session_constant_last` 방식 — 현재 position t에서 계산하지 않고, **마지막 known interaction의 mid feature를 세션 전체에 상수로 적용**  
(원래 `session_full`에서 `session_constant_last`로 변경됨, feature_meta에 기록됨)

#### Micro scope (16개)

| Feature | Family | 의미 |
|---|---|---|
| `mic_valid_r` | Tempo | 최근 5 interactions에서 valid gap 비율 |
| `mic_last_gap` | Tempo | 직전 interaction과의 시간 간격 |
| `mic_gap_mean` | Tempo | 최근 5개 gap 평균 |
| `mic_gap_delta_vs_mid` | Tempo | micro gap vs mid gap 차이 (phi-norm) |
| `mic_cat_switch_now` | Focus | 현재 position에서 category switch 여부 |
| `mic_last_cat_mismatch_r` | Focus | 최근 5개에서 직전 category와 불일치 비율 |
| `mic_suffix_cat_ent` | Focus | 최근 5개 category entropy |
| `mic_suffix_cat_uniq_r` | Focus | 최근 5개 unique category 비율 |
| `mic_is_recons` | Memory | 현재 아이템이 이전에 소비한 것인지 |
| `mic_suffix_recons_r` | Memory | 최근 5개 중 재소비 비율 |
| `mic_suffix_uniq_i` | Memory | 최근 5개 unique item 비율 |
| `mic_suffix_max_run_i` | Memory | 최근 5개 최장 same-item run |
| `mic_last_pop` | Exposure | 직전 아이템 인기도 |
| `mic_suffix_pop_std` | Exposure | 최근 5개 아이템 인기도 표준편차 |
| `mic_suffix_pop_ent` | Exposure | 최근 5개 인기도 entropy |
| `mic_pop_delta_vs_mid` | Exposure | micro pop vs mid pop 차이 (phi-norm) |

### 5.3 정규화 방식

```
bounded features (0~1 범위): clip to [0, 1]
continuous features: log1p → winsorize(p01, p99) → z-score → phi(Φ)
  where phi = 표준정규 CDF = (1 + erf(x/sqrt(2))) / 2
pair features: 0.5 + 0.5 * (norm(lhs) - norm(rhs))
missing values: bounded → 0.5 (neutral), temporal → 0.0, context features → 0.5
```

**정규화 통계는 train split (70%)에서만 fit.**  
fit session count = 17,121 (KuaiRec), 17,562 (LastFM)

### 5.4 Popularity Binning

item popularity = train split에서의 interaction count  
normalization: log1p 후 10개 bin으로 discretize  
`n_pop_bins = 10`

---

## 6. Full Dataset 재구성 계획

### 6.1 목표

현재 실험은 `lastfm0.03` (130 users, 3% sample)과 `KuaiRecLargeStrictPosV2_0.2` (1,122 users)에서 진행.  
Rebuttal용으로 **full dataset 버전**을 추가 실험에 사용하려면 전체 파이프라인 재구성 필요.

- `lastfm_full`: 1K users 전체 (기존 130 → ~992 active users → ~25만 sessions 예상)
- `KuaiRec_full`: watch_ratio ≥ 0.5 전체 (기존 1,122 → 7,135 users, 293,150 sessions 확인됨)

### 6.2 재구성할 코드 파일

새로 만들 스크립트: `experiments/tools/`

```
experiments/tools/
├── build_kuairec_basic.py        # Raw → Basic (sessionization + filtering)
├── build_lastfm_basic.py         # Raw → Basic (sessionization + filtering)
├── build_feature_v3_from_basic.py  # Basic → Feature v3 (feature 계산 + 정규화)
└── run_full_pipeline.sh          # 전체 파이프라인 실행 스크립트
```

### 6.3 KuaiRec Full 재구성 파라미터 (시뮬레이션으로 검증됨)

```python
# Step 1: watch_ratio filter
MIN_WATCH_RATIO = 0.5           # 확정 (positive implicit feedback)

# Step 2: Sessionization
SESSION_GAP_SECONDS = 1800      # 30분 확정

# Step 3: Iterative k-core
MIN_SESSION_LENGTH = 5          # 확실
MIN_ITEM_FREQ = 3               # 확실
N_KCORE_ROUNDS = 3              # 시뮬레이션에서 3회로 안정적

# Item 필드
ITEM_ID_FIELD = "video_id"
CATEGORY_FIELD = "feat"         # item_categories.csv에서 join, 첫 번째 category 사용
TIMESTAMP_FIELD = "timestamp"   # Unix seconds (float)
```

**Full pipeline 예상 결과 (시뮬레이션 확인):**
```
sessions:  293,150
users:     7,135
items:     9,431
```

**user sampling 없이 전체 사용** — "KuaiRec-Full"로 cite.  
현재 paper 데이터(1,122 users)와 직접 수치 비교는 불가하지만,  
더 큰 스케일에서도 동일 패턴을 보인다는 추가 evidence로 활용 가능.

**item_categories.csv 처리:**
```python
# video_id → first_category 매핑
# feat 컬럼: "[8]" 또는 "[27, 9]" → ast.literal_eval 후 첫 번째 값
import ast
category = ast.literal_eval(row['feat'])[0]
```

### 6.4 LastFM Full 재구성 파라미터

```python
# Step 1: User sampling
# → full dataset이므로 sampling 없이 전체 사용 (~992 active users)
# lastfm0.03의 "0.03" = 3% random user sample = 130 users

# Timestamp 처리
# raw format: ISO 8601 ("2009-05-04T23:08:57Z") → int(datetime.fromisoformat().timestamp() * 1000)

# Step 2: Sessionization
SESSION_GAP_MS = 30 * 60 * 1000   # 30분 = 1,800,000ms (KuaiRec과 동일 기준, 확정)

# Step 3: Iterative k-core
MIN_SESSION_LENGTH = 5             # 확실
MIN_ITEM_FREQ = 3                  # 확실
N_KCORE_ROUNDS = 3

# Item 필드
ITEM_ID_FIELD = "track_id"        # col index 4 (0-indexed), song level
CATEGORY_FIELD = "artist_id"      # col index 2, artist as category
TIMESTAMP_FIELD = "timestamp"     # ISO 8601 → ms Unix 변환 필요

# raw TSV 컬럼 순서 (헤더 없음):
# user_id \t timestamp \t artist_id \t artist_name \t track_id \t track_name
```

**Full pipeline 예상 결과:**
```
users:     ~992 (1K dataset 전체, 비활성 user 제외)
sessions:  ~190,000 ~ 200,000 (lastfm0.03 25,089 × 33배 ≈ 추정)
```
"LastFM-Full"로 cite. `lastfm0.03`의 130 users 대비 훨씬 많은 users로 규모 증명 가능.

### 6.5 Feature 계산 파라미터 (재구성용)

```python
# Macro scope
MACRO_WINDOWS = [5, 10]  # 확실 (feature_meta에 명시)

# Mid scope
MID_CONSTANT_LAST = True  # session_constant_last 방식
MID_VALID_CAP = 10       # 확실 (feature_meta에 명시)

# Micro scope
MICRO_WINDOW = 5  # 확실 (feature_meta에 명시)

# Normalization
N_POP_BINS = 10
# normalization stats는 train split (70%)에서 fit
# 정규화 통계는 각 dataset별로 다름 → feature_meta_v3.json의 normalization_stats에 저장됨
```

---

## 7. full dataset 실험 이유 및 주의

Rebuttal에서 full dataset 실험을 넣는 주요 이유:
- `lastfm0.03`은 130 users에 불과 → reviewer가 "sample이 너무 작다" 지적 가능
- `KuaiRec 0.2`의 의미가 불명확할 수 있음 → 원본 KuaiRec 대비 어떤 subset인지 해명 필요

**단, full dataset 실험을 넣을 때 주의사항:**
1. 현재 paper에서 쓴 subset 결과가 바뀌어서는 안 됨 (full은 additional experiment)
2. feature_added_v4와 동일한 포맷으로 출력해야 RecBole 실험 재활용 가능
3. `feature_meta_v3.json`의 정규화 통계를 새로 fit해야 함 (train split 기준)

---

## 8. 역공학 불확실 요소 요약 (업데이트)

| 항목 | 상태 | 근거 |
|---|---|---|
| KuaiRec watch_ratio threshold | **0.5 확정** | wr≥0.5 → 8.0M rows (68.5%); wr>1.0/user_mean 시험해도 20% sampling시 users 수 불일치 |
| KuaiRec session gap | **30분 확정** | `_s{N}_c{chunk}` 구조 + 시뮬레이션으로 확인 |
| KuaiRec min_session_length | **5 확정** | processed min len = 5 |
| KuaiRec min_item_freq | **3 확정** | k-core x3 시뮬레이션 결과가 item 수와 일치 |
| KuaiRec item category 출처 | **확인됨** | `item_categories.csv`, `feat` 컬럼, 첫 번째 값 사용 |
| **KuaiRec user sampling 기준** | **20% random, seed 미상** | "0.2" = 20% sampling 확정. 전체 7,135 → 1,122명. wr>1.0/user_mean 필터 후 20% 적용 시 1,372~1,383 (불일치). exact seed 재현 불가 |
| LastFM session gap | **30분 확정** | KuaiRec과 동일 기준, intra-session duration 분포로 검증 |
| LastFM min_session_length | **5 확정** | processed min len = 5 |
| LastFM min_item_freq | **3 확정** | KuaiRec과 동일 기준 |
| LastFM user sampling (0.03) | **random sampling** | 전체 ~992 users 중 130명(13.1%) — "0.03" 표기는 3%이나 실제 비율은 13%. seed 미상 |
| iterative k-core 반복 횟수 | **3회 확정** | 시뮬레이션에서 3회 후 수렴 확인 |

**핵심 결론**: KuaiRec의 1,122 users 선택 기준만 재현 불가. 나머지 파라미터는 모두 확정. Full dataset 재구성은 user sampling 없이 전체 7,135 users 대상으로 진행하면 됨.
