# FMoE Experiments - Technical Manual

> **버전**: 2026.01.24  
> **RecBole**: 1.2.1  
> **PyTorch**: 2.5.1+cu121

이 문서는 FMoE baseline 실험 프레임워크의 **기술적 세부사항**을 다룹니다.  
빠른 사용법은 [README.md](README.md)를 참고하세요.

---

## 목차

1. [아키텍처 개요](#1-아키텍처-개요)
2. [데이터 흐름](#2-데이터-흐름)
3. [Evaluation Modes 상세](#3-evaluation-modes-상세)
4. [Feature Modes 상세](#4-feature-modes-상세)
5. [Config 시스템 심화](#5-config-시스템-심화)
6. [RecBole 패치 상세](#6-recbole-패치-상세)
7. [wandb 통합](#7-wandb-통합)
8. [성능 최적화](#8-성능-최적화)
9. [트러블슈팅 가이드](#9-트러블슈팅-가이드)
10. [확장 가이드](#10-확장-가이드)

---

## 1. 아키텍처 개요

### 1.1 디렉토리 구조

```
2026_FMoE/
├── Datasets/
│   ├── raw/                          # 원본 데이터 (백업용)
│   ├── processed/
│   │   ├── basic/                    # Feature 없는 데이터 (~2.7GB)
│   │   │   └── {dataset}/
│   │   │       ├── {dataset}.inter           # 전체 interactions
│   │   │       ├── {dataset}.train.inter     # Train split
│   │   │       ├── {dataset}.valid.inter     # Validation split
│   │   │       ├── {dataset}.test.inter      # Test split
│   │   │       └── {dataset}.item            # Item metadata
│   │   │
│   │   ├── feature_added/            # 50+ feature 포함 (~37GB)
│   │   │   └── (basic과 동일 구조)
│   │   │
│   ├── create_basic_datasets.py      # Basic 데이터셋 생성 스크립트
│   ├── feature_injection.py          # Feature 주입 스크립트
│   └── split.py                      # Train/valid/test 분할 스크립트
│
└── experiments/                      # 실험 코드 (Datasets와 평행)
  ├── configs/
  │   ├── config.yaml               # 메인 Hydra config (dataset_root=../Datasets/processed)
  │   ├── eval_mode/                # SESSION / INTERACTION
  │   ├── feature_mode/             # full / basic
  │   ├── model/                    # 모델별 오버라이드
  │   └── dataset/                  # 데이터셋별 오버라이드
  ├── run/
  │   └── artifacts/                # 통합 로그/결과/타임라인
  ├── docs/                         # 운영/구조 문서
  ├── tools/                        # 운영 유틸리티 스크립트
  │   └── cleanup_gpu.sh            # GPU 프로세스 정리
  ├── recbole_train.py              # 메인 진입점
  ├── recbole_patch.py              # RecBole 패치
  └── hydra_utils.py                # Hydra 유틸리티
```

  > 실행은 `cd experiments` 후 진행 (config의 `dataset_root=../Datasets/processed`).

### 1.2 실행 흐름

```
┌─────────────────────────────────────────────────────────────────────────┐
│  python recbole_train.py model=SASRec dataset=KuaiRec                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Hydra Config 로드 (hydra_utils.py)                                  │
│     - config.yaml (base)                                                │
│     - eval_mode/*.yaml                                                  │
│     - feature_mode/*.yaml                                               │
│     - datasets/*.yaml                                                   │
│     - models/*.yaml                                                     │
│     - CLI overrides                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. RecBole 패치 적용 (recbole_patch.py)                                │
│     - SequentialDataset._benchmark_presets()                            │
│     - SequentialDataset.build()                                         │
│     - get_flops()                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. wandb 초기화 (log_wandb=true인 경우)                                │
│     - Custom run name 생성                                              │
│     - wandb.init() 호출                                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. RecBole run_recbole() 호출                                          │
│     - Dataset 로드                                                      │
│     - Model 생성                                                        │
│     - Training loop                                                     │
│     - Evaluation                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. 결과 처리                                                           │
│     - max_val_metric 기록                                               │
│     - wandb summary 업데이트                                            │
│     - 로그 출력                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 흐름

### 2.1 데이터 파일 형식

#### .inter 파일 (Interaction)
```
session_id:token	item_id:token	timestamp:float	user_id:token	[features...]
28_c0	503	1092602547000	28	-0.21	0	1	...
```

- **session_id**: 세션 고유 ID (USER_ID_FIELD로 사용)
- **item_id**: 아이템 고유 ID
- **timestamp**: 상호작용 시간 (Unix timestamp)
- **user_id**: 실제 사용자 ID (참고용)
- **features**: 50+ 엔지니어링된 feature (feature_added만)

#### .item 파일 (Item Metadata)
```
item_id:token	category:token
1458	Makeup_Face
2092	Skincare_Treatment
```

### 2.2 데이터 로딩 과정

```
┌──────────────────────────────────────────────────────────────────────┐
│ SESSION 모드                                                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  {dataset}.train.inter ──┐                                          │
│  {dataset}.valid.inter ──┼──> RecBole 로드 (benchmark mode)         │
│  {dataset}.test.inter  ──┘                                          │
│                              │                                       │
│                              ▼                                       │
│                    _patched_benchmark_presets()                      │
│                              │                                       │
│                              ▼                                       │
│                    _convert_inter_to_sequence()                      │
│                    (interaction → sequence 변환)                     │
│                              │                                       │
│                              ▼                                       │
│  Train: Augmented sequences (각 위치마다 샘플 생성)                  │
│  Valid/Test: 마지막 item 예측 (1 sample per session)                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ INTERACTION 모드                                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  {dataset}.inter ──> RecBole 로드 (LS split)                        │
│                              │                                       │
│                              ▼                                       │
│                    RecBole native split                              │
│                    (Leave-one-out within session)                    │
│                              │                                       │
│                              ▼                                       │
│  Train: 각 세션의 앞부분 interactions                                │
│  Valid: 각 세션의 마지막-1 item                                      │
│  Test:  각 세션의 마지막 item                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 Sequence 변환 상세

SESSION 모드에서 `_convert_inter_to_sequence()` 함수가 수행하는 변환:

**입력 (interaction format)**:
```
session_id  item_id  timestamp
S1          A        100
S1          B        101
S1          C        102
S2          D        200
S2          E        201
```

**출력 - Training (augmented)**:
```
session_id  item_id_list  item_length  target_id
S1          [A]           1            B
S1          [A,B]         2            C
S2          [D]           1            E
```

**출력 - Evaluation**:
```
session_id  item_id_list  item_length  target_id
S1          [A,B]         2            C
S2          [D]           1            E
```

---

## 3. Evaluation Modes 상세

### 3.1 SESSION 모드 (`eval_mode=session`)

#### 개념
- **세션 단위**로 완전히 분리
- Train 세션은 학습에만, Valid 세션은 검증에만, Test 세션은 테스트에만 사용
- **Data leakage 없음**: 평가 세션의 정보가 학습에 사용되지 않음

#### 구현
```yaml
# configs/eval_mode/session.yaml
benchmark_filename: null  # Pre-split 파일 사용 안 함
eval_args:
  split: {'RS': [0.7, 0.15, 0.15]}  # 70/15/15 session 분할
  group_by: user  # USER_ID_FIELD (=session_id)로 그룹핑
  order: TO  # 시간순 정렬
  mode: full
```

**동작 방식:**
1. 전체 데이터 로드 (모든 sessions)
2. SESSION_ID (=USER_ID_FIELD) 기준으로 그룹핑
3. 시간순 정렬 (TIME_FIELD)
4. Sessions를 70/15/15 비율로 분할
5. Train sessions: 모든 interactions 사용 (data augmentation)
6. Valid/Test sessions: 마지막 item만 예측

#### 데이터 분할 비율
- Train: ~70% sessions (약 530개 세션, amazon_beauty 기준)
- Valid: ~15% sessions (약 114개 세션)
- Test: ~15% sessions (약 114개 세션)
- **중요**: Sessions 기준 분할이므로, interactions는 불균등할 수 있음

#### 장점
- 실제 production 환경과 유사
- Cold-start 세션 평가 가능
- 시간적 일관성 유지
- RecBole 네이티브 split 사용 (안정적)

#### 단점
- 학습 데이터 상대적으로 적음
- 세션 수가 적은 데이터셋에서 불안정

### 3.2 INTERACTION 모드 (`eval_mode=interaction`)

#### 개념
- 모든 세션이 **모델에 입력**되지만
- 세션 **내부**에서 interactions을 분리
- 마지막 item → test, 그 전 item → valid, 나머지 → train

#### 구현
```yaml
# configs/eval_mode/interaction.yaml
benchmark_filename: null
eval_args:
  split: {'LS': 'valid_and_test'}  # Leave-one-out Split
  group_by: user  # USER_ID_FIELD (=session_id)
  order: TO  # 시간순 정렬
  mode: full
```

**동작 방식:**
1. 전체 데이터 로드 (모든 sessions)
2. SESSION_ID 기준으로 그룹핑
3. 각 session 내에서 position-based split:
   - Last interaction → test
   - 2nd-to-last interaction → valid
   - All other interactions → train
4. 최소 session 길이: 3 interactions

#### 예시
```
Session: [item1, item2, item3, item4, item5]
Train: item1, item2, item3 (+ data augmentation)
Valid: item4
Test: item5
```

#### 장점
- 더 많은 학습 데이터 (모든 sessions 기여)
- 모든 세션의 패턴 학습 가능
- RecBole 네이티브 LS split 사용 (안정적)

#### 단점
- 평가 세션의 앞부분이 학습에 사용됨 (약한 data leakage)
- 길이 2 이하 세션은 제외됨

### 3.3 모드 선택 가이드

| 상황 | 권장 모드 |
|------|----------|
| 논문 실험 (엄격한 평가) | SESSION |
| Hyperparameter 튜닝 | INTERACTION (빠른 피드백) |
| Cold-start 연구 | SESSION |
| 소규모 데이터셋 | INTERACTION |
| Production 시뮬레이션 | SESSION |

---

## 4. Feature Modes 상세

### 4.1 FULL 모드 (`feature_mode=full`)

#### 데이터 위치
```
processed/feature_added/{dataset}/
```

#### 포함 Features (50+개)

**Macro-level (mac_*)**:
- `mac_user_level`: 사용자 활동 수준
- `mac_sess_gap`: 세션 간 시간 간격
- `mac_is_new`: 신규 사용자 여부
- `mac_hist_len`: 사용자 히스토리 길이
- `mac_time_sin/cos`: 시간의 sin/cos 인코딩
- `mac_is_weekend`: 주말 여부
- `mac_hist_speed`: 히스토리 속도
- `mac_hist_ent`: 히스토리 엔트로피
- 등...

**Meso-level (mid_*)**:
- `mid_win_ent`: 윈도우 엔트로피
- `mid_sess_time`: 세션 내 시간
- `mid_pop_avg/std`: 인기도 평균/표준편차
- `mid_accel`: 가속도
- 등...

**Micro-level (mic_*)**:
- `mic_last_int`: 마지막 상호작용 간격
- `mic_switch`: 아이템 전환 여부
- `mic_cat_ent`: 카테고리 엔트로피
- `mic_is_recons`: 재구매 여부
- 등...

#### 용량
- 전체: ~37GB
- 로딩 시간: 데이터셋에 따라 1-10분

### 4.2 BASIC 모드 (`feature_mode=basic`)

#### 데이터 위치
```
processed/basic/{dataset}/
```

#### 포함 컬럼
```
session_id:token  item_id:token  timestamp:float  user_id:token
```

#### 용량
- 전체: ~2.7GB
- 로딩 시간: 수 초

### 4.3 모드별 용량 비교

| Dataset | FULL | BASIC | 절감률 |
|---------|------|-------|--------|
| amazon_beauty | 7MB | 0.8MB | 89% |
| foursquare | 126MB | 10MB | 92% |
| KuaiRec | 12GB | 788MB | 93% |
| lastfm | 15.7GB | 1.2GB | 93% |
| movielens1m | 486MB | 36MB | 93% |
| retail_rocket | 636MB | 75MB | 88% |

### 4.4 사용 가이드

```bash
# Baseline 모델 실험 (SASRec, GRU4Rec 등) - BASIC 권장
python recbole_train.py model=SASRec dataset=KuaiRec feature_mode=basic

# Feature-aware 모델 실험 - FULL 필요
python recbole_train.py model=FeaturedMoE dataset=KuaiRec feature_mode=full

# Hyperparameter 튜닝 - BASIC 권장 (빠른 반복)
python recbole_train.py model=SASRec dataset=KuaiRec feature_mode=basic learning_rate=0.0005
```

---

## 5. Config 시스템 심화

### 5.1 Hydra 기본 개념

Hydra는 계층적 config 관리 시스템입니다:

```yaml
# 기본값은 defaults 섹션에서 지정
defaults:
  - eval_mode: session      # configs/eval_mode/session.yaml 로드
  - feature_mode: full      # configs/feature_mode/full.yaml 로드
  - _self_                  # 현재 파일의 나머지 부분

# 이후 설정은 defaults를 override
model: ???  # 필수 지정
dataset: ???  # 필수 지정
```

### 5.2 Config 우선순위

낮음 → 높음:
1. `config.yaml` (base)
2. `eval_mode/*.yaml`
3. `feature_mode/*.yaml`
4. `models/*.yaml` (optional)
5. `datasets/*.yaml` (optional)
6. **CLI arguments** (최우선)

### 5.3 CLI Override 문법

```bash
# 단순 값 override
python recbole_train.py model=SASRec dataset=KuaiRec learning_rate=0.0005

# 중첩 구조 override
python recbole_train.py model=SASRec dataset=KuaiRec eval_args.mode.valid=uni100

# 리스트 override
python recbole_train.py model=SASRec dataset=KuaiRec topk=[5,10,20,50]
python recbole_train.py model=SASRec dataset=KuaiRec 'metrics=["Hit","NDCG"]'

# 딕셔너리 override
python recbole_train.py model=SASRec dataset=KuaiRec 'load_col={inter:[session_id,item_id,timestamp]}'

# Config 그룹 선택
python recbole_train.py model=SASRec dataset=KuaiRec eval_mode=interaction feature_mode=basic
```

### 5.4 주요 Config 파라미터 상세

#### Core Parameters
| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `model` | str | (필수) | RecBole 모델명 |
| `dataset` | str | (필수) | 데이터셋명 (소문자) |
| `seed` | int | 42 | Random seed |
| `reproducibility` | bool | true | 재현성 보장 |

#### Device Parameters
| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `gpu_id` | int | 0 | GPU ID (단일 GPU) |
| `use_gpu` | bool | true | GPU 사용 여부 |

> **Tip**: 서버에서는 `CUDA_VISIBLE_DEVICES` 환경변수가 더 안전합니다.
> ```bash
> CUDA_VISIBLE_DEVICES=3 python recbole_train.py ...
> ```

#### Training Parameters
| 파라미터 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|-------|------|------|
| `epochs` | int | 100 | 10-500 | 최대 학습 epoch |
| `train_batch_size` | int | 256 | 32-2048 | 학습 배치 크기 |
| `learning_rate` | float | 0.001 | 1e-5 ~ 1e-2 | 학습률 |
| `stopping_step` | int | 10 | 3-30 | Early stopping patience |
| `weight_decay` | float | 0.0 | 0-1e-3 | L2 정규화 |
| `clip_grad_norm` | float | null | 0-10 | Gradient clipping |

#### Model Parameters (공통)
| 파라미터 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|-------|------|------|
| `hidden_size` | int | 128 | 32-512 | Hidden dimension |
| `embedding_size` | int | 128 | 32-512 | Embedding dimension |
| `num_layers` | int | 2 | 1-6 | Layer 수 |
| `hidden_dropout_prob` | float | 0.2 | 0-0.5 | Hidden dropout |

#### Transformer-specific (SASRec, BERT4Rec)
| 파라미터 | 타입 | 기본값 | 범위 | 설명 |
|---------|------|-------|------|------|
| `num_heads` | int | 4 | 1-8 | Attention heads |
| `attn_dropout_prob` | float | 0.2 | 0-0.5 | Attention dropout |
| `hidden_act` | str | 'gelu' | gelu/relu | Activation |
| `layer_norm_eps` | float | 1e-12 | - | LayerNorm epsilon |

#### Sequence Parameters
| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `MAX_ITEM_LIST_LENGTH` | int | 50 | 최대 시퀀스 길이 |
| `USER_ID_FIELD` | str | 'session_id' | 사용자/세션 ID 필드 |
| `ITEM_ID_FIELD` | str | 'item_id' | 아이템 ID 필드 |
| `TIME_FIELD` | str | 'timestamp' | 시간 필드 |

#### Evaluation Parameters
| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `metrics` | list | [Hit, NDCG, MRR] | 평가 지표 |
| `topk` | list | [5, 10, 20] | Top-K 값들 |
| `valid_metric` | str | 'NDCG@10' | Early stopping 기준 |
| `eval_batch_size` | int | 512 | 평가 배치 크기 |

#### Saving Parameters
| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `saved` | bool | false | 모델 저장 여부 |
| `save_dataset` | bool | false | 처리된 데이터셋 저장 |
| `checkpoint_dir` | str | 'saved' | 체크포인트 디렉토리 |

#### Logging Parameters
| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `log_wandb` | bool | false | W&B 로깅 활성화 |
| `wandb_project` | str | 'FMoE_2026' | W&B 프로젝트명 |
| `show_progress` | bool | true | Progress bar 표시 |
| `state` | str | 'INFO' | 로그 레벨 |

### 5.5 Dataset-specific Configs

각 데이터셋에 맞는 권장 설정:

```yaml
# configs/datasets/kuairec.yaml
# @package _global_

dataset: KuaiRec

# Large dataset: larger batches, more epochs
train_batch_size: 512
epochs: 100
MAX_ITEM_LIST_LENGTH: 50
```

```yaml
# configs/datasets/amazon_beauty.yaml
# @package _global_

dataset: amazon_beauty

# Small dataset: smaller batches, fewer epochs
train_batch_size: 64
epochs: 20
MAX_ITEM_LIST_LENGTH: 50
```

### 5.6 Model-specific Configs

```yaml
# configs/models/sasrec.yaml
# @package _global_

# SASRec defaults
hidden_size: 128
num_layers: 2
num_heads: 4
hidden_dropout_prob: 0.2
attn_dropout_prob: 0.2
hidden_act: 'gelu'

# Grid search 설정 (--search 플래그 사용 시)
search:
  hidden_size: [64, 128, 256]
  learning_rate: [0.001, 0.0005, 0.0001]
  num_layers: [1, 2, 3]
```

---

## 6. RecBole 패치 상세

### 6.1 패치가 필요한 이유

RecBole 1.2.1의 기본 동작:
1. **benchmark 모드**: pre-split 파일을 로드하지만, **sequence format**을 기대
2. **LS split**: 단일 `.inter` 파일에서 내부 split 수행

문제점:
- 우리 데이터는 **interaction format** (session_id, item_id, timestamp)
- RecBole의 RS split은 **세션 내부**에서 분할 (세션 간 분할 아님)
- FLOPS 계산이 sequence format을 가정

### 6.2 적용된 패치

#### Patch 1: `_patched_benchmark_presets()`

**위치**: `SequentialDataset._benchmark_presets()`

**기능**: Benchmark 파일 로드 후, interaction format을 sequence format으로 변환

```python
def _patched_benchmark_presets(self):
    """Convert interaction-format benchmark files to sequence format."""
    # 원본 메서드 호출 (파일 로드)
    _original_benchmark_presets(self)
    
    # 변환 필요 여부 체크
    if 'item_id_list' not in self.inter_feat.columns:
        # Train: augmented sequences
        self.inter_feat = _convert_inter_to_sequence(self, self.inter_feat, for_training=True)
```

#### Patch 2: `_patched_build()`

**위치**: `SequentialDataset.build()`

**기능**: Valid/Test 데이터도 sequence format으로 변환

```python
def _patched_build(self):
    """Handle SESSION mode with interaction-format pre-split files."""
    datasets = _original_build(self)
    
    # SESSION 모드 (benchmark 사용)일 때만
    if 'benchmark_filename' in self.config:
        for ds in datasets[1:]:  # valid, test
            if 'item_id_list' not in ds.inter_feat.columns:
                ds.inter_feat = _convert_inter_to_sequence(ds, ds.inter_feat, for_training=False)
    
    return datasets
```

#### Patch 3: `_patched_get_flops()`

**위치**: `recbole.utils.utils.get_flops()`

**기능**: SESSION 모드에서 FLOPS 계산 스킵 (sequence data 없음)

```python
def _patched_get_flops(model, dataset, device, logger, transform=None, verbose=False):
    """Skip FLOPS calculation in SESSION mode."""
    if 'item_id_list' not in dataset.inter_feat.columns:
        logger.info("Skipping FLOPS calculation (SESSION mode)")
        return 0
    return _original_get_flops(...)
```

### 6.3 Sequence 변환 알고리즘

```python
def _convert_inter_to_sequence(dataset, inter_feat, for_training=True):
    """
    Interaction format → Sequence format 변환
    
    Parameters:
        dataset: SequentialDataset instance
        inter_feat: Interaction or DataFrame
        for_training: True=augmented, False=single sample per session
    """
    # 1. 세션별 그룹핑
    groups = df.groupby(session_id_field)
    
    # 2. 각 세션을 시퀀스로 변환
    for session_id, group in groups:
        items = group[item_id_field].values
        
        if for_training:
            # Augmentation: 각 위치에서 다음 아이템 예측
            # [A,B,C,D] → ([A]→B), ([A,B]→C), ([A,B,C]→D)
            for i in range(1, len(items)):
                sequences.append(items[:i])
                targets.append(items[i])
        else:
            # Evaluation: 마지막 아이템 예측
            # [A,B,C,D] → ([A,B,C]→D)
            sequences.append(items[:-1])
            targets.append(items[-1])
    
    # 3. Padding 및 Interaction 객체 생성
    padded_seqs = pad_sequences(sequences, maxlen=max_len)
    return Interaction({
        session_id_field: session_ids,
        item_id_list_field: padded_seqs,
        item_length_field: lengths,
        target_item_field: targets
    })
```

---

## 7. wandb 통합

### 7.1 Run Name 형식

```
{DATA3}_{MODEL3}_{INFO}_{MMDDHHMM}
```

- 그룹 사이만 `_`, 내부 INFO는 구분자 없음, 모두 대문자/소문자 혼용 유지
- INFO 토큰: `eS`/`eI`(eval), `e{n}`(epochs, 기본 100 제외), `lr{v}`(lr≠0.001), `h{v}`(hidden_size≠128), `b{v}`(batch≠256)을 이어붙임
- TIMESTAMP: `MMDDHHMM`

예시:
- `AMA_SAS_eS_e1_01241430` (amazon_beauty, SASRec, session, epochs=1)
- `KUA_GRU_eI_e50_lr0.0005_01241530` (KuaiRec, GRU4Rec, interaction, epochs=50, lr=0.0005)

#### 약어 매핑

**Dataset**:
| Full Name | Abbreviation |
|-----------|--------------|
| amazon_beauty | AMA |
| foursquare | NYC |
| KuaiRec | KUA |
| KuaiRec0.3 | KU3 |
| lastfm | LFM |
| lastfm0.3 | LF3 |
| movielens1m | ML1 |
| retail_rocket | ReR |

**Model**:
| Full Name | Abbreviation |
|-----------|--------------|
| SASRec | SAS |
| GRU4Rec | GRU |
| NARM | NRM |
| BERT4Rec | BRT |
| STAMP | STP |
| SRGNN | GNN |

### 7.2 기록되는 Metrics

**Training**:
- `train/train_loss`: 학습 loss
- `train/epoch`: 현재 epoch

**Validation**:
- `valid/hit@{k}`: Validation Hit@K
- `valid/ndcg@{k}`: Validation NDCG@K
- `valid/mrr@{k}`: Validation MRR@K

**Summary**:
- `max_val_metric`: 최고 validation metric (overfitting 추적용)
- `best_valid_*`: Best epoch의 validation 결과
- `test_*`: Test 결과
- `total_time_seconds/minutes`: 총 실행 시간

### 7.3 사용법

```bash
# 기본 로깅
python recbole_train.py model=SASRec dataset=KuaiRec log_wandb=true

# 프로젝트 변경
python recbole_train.py model=SASRec dataset=KuaiRec log_wandb=true wandb_project=Experiments

# 오프라인 모드
WANDB_MODE=offline python recbole_train.py model=SASRec dataset=KuaiRec log_wandb=true
```

---

## 8. 성능 최적화

### 8.1 데이터 로딩 최적화

```bash
# 1. BASIC 모드 사용 (가장 효과적)
python recbole_train.py ... feature_mode=basic

# 2. RecBole 캐싱 활성화
python recbole_train.py ... save_dataset=true
# 두 번째 실행부터 캐시 사용
```

### 8.2 학습 최적화

```bash
# 1. 배치 크기 조정 (GPU 메모리에 맞게)
python recbole_train.py ... train_batch_size=512

# 2. Mixed precision (자동 적용됨)
# RecBole 1.2.1은 자동으로 AMP 사용

# 3. Worker 수 조정
python recbole_train.py ... worker=4
```

### 8.3 메모리 최적화

```bash
# 1. 시퀀스 길이 제한
python recbole_train.py ... MAX_ITEM_LIST_LENGTH=30

# 2. 평가 배치 크기 조정
python recbole_train.py ... eval_batch_size=256

# 3. Gradient accumulation (큰 effective batch)
# RecBole에서는 직접 구현 필요
```

### 8.4 GPU 메모리 사용량 가이드

| Dataset | Model | Batch Size | GPU Memory |
|---------|-------|------------|------------|
| amazon_beauty | SASRec | 256 | ~1GB |
| movielens1m | SASRec | 256 | ~2GB |
| KuaiRec | SASRec | 512 | ~8GB |
| lastfm | SASRec | 512 | ~10GB |
| KuaiRec | GRU4Rec | 1024 | ~6GB |

---

## 9. 트러블슈팅 가이드

### 9.1 CUDA Out of Memory

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**해결책**:
```bash
# 1. 배치 크기 줄이기
python recbole_train.py ... train_batch_size=64 eval_batch_size=128

# 2. 시퀀스 길이 줄이기
python recbole_train.py ... MAX_ITEM_LIST_LENGTH=30

# 3. 모델 크기 줄이기
python recbole_train.py ... hidden_size=64 num_layers=1

# 4. GPU 캐시 정리
./tools/cleanup_gpu.sh --kill
```

### 9.2 데이터 로딩 오류

**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../KuaiRec.train.inter'
```

**해결책**:
```bash
# 1. 데이터 경로 확인
ls -la ../feature_added/KuaiRec/

# 2. feature_mode 확인
python recbole_train.py ... feature_mode=full  # or basic

# 3. dataset 이름 확인 (대소문자 주의)
# 파일: KuaiRec.inter
# config: dataset=kuairec (소문자)
```

### 9.3 Config 오류

**증상**:
```
omegaconf.errors.ConfigAttributeError: Key 'xxx' is not in struct
```

**해결책**:
```bash
# 1. Config 키 확인
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('configs/config.yaml'))"

# 2. 올바른 override 문법 사용
python recbole_train.py model=SASRec dataset=kuairec learning_rate=0.0005
# NOT: python recbole_train.py --learning_rate 0.0005
```

### 9.4 wandb 오류

**증상**:
```
wandb.errors.CommError: Network connection error
```

**해결책**:
```bash
# 1. 로그인 확인
wandb login

# 2. 오프라인 모드 사용
WANDB_MODE=offline python recbole_train.py ... log_wandb=true

# 3. 나중에 동기화
wandb sync wandb/offline-run-xxx
```

### 9.5 NaN Loss

**증상**:
```
train loss: nan
```

**해결책**:
```bash
# 1. 학습률 낮추기
python recbole_train.py ... learning_rate=0.0001

# 2. Gradient clipping 추가
python recbole_train.py ... clip_grad_norm=1.0

# 3. Weight decay 추가
python recbole_train.py ... weight_decay=1e-5
```

### 9.6 느린 학습

**증상**: Epoch당 시간이 너무 오래 걸림

**해결책**:
```bash
# 1. BASIC 모드 사용
python recbole_train.py ... feature_mode=basic

# 2. 배치 크기 늘리기 (GPU 여유 있을 때)
python recbole_train.py ... train_batch_size=1024

# 3. Worker 수 조정
python recbole_train.py ... worker=4

# 4. 데이터셋 캐싱
python recbole_train.py ... save_dataset=true
```

---

## 10. 확장 가이드

### 10.1 새 데이터셋 추가

1. **데이터 준비**:
```
processed/feature_added/{new_dataset}/
├── {new_dataset}.inter       # 전체 interactions
├── {new_dataset}.train.inter # Train split
├── {new_dataset}.valid.inter # Valid split
├── {new_dataset}.test.inter  # Test split
└── {new_dataset}.item        # Item metadata
```

2. **Config 생성**:
```yaml
# configs/datasets/new_dataset.yaml
# @package _global_

dataset: new_dataset

# 데이터셋 특성에 맞게 조정
train_batch_size: 256
epochs: 100
MAX_ITEM_LIST_LENGTH: 50
```

3. **Basic 버전 생성**:
```python
# create_basic_datasets.py의 DATASETS 리스트에 추가
DATASETS = [..., "new_dataset"]
python create_basic_datasets.py
```

### 10.2 새 모델 추가

1. **RecBole 지원 모델**: 바로 사용 가능
```bash
python recbole_train.py model=NewModel dataset=kuairec
```

2. **커스텀 모델**:
```python
# models/my_model.py
from recbole.model.abstract_recommender import SequentialRecommender

class MyModel(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # 모델 초기화
    
    def forward(self, interaction):
        # Forward pass
        return output
    
    def calculate_loss(self, interaction):
        # Loss 계산
        return loss
    
    def predict(self, interaction):
        # 예측
        return scores
```

3. **Config 생성**:
```yaml
# configs/models/my_model.yaml
# @package _global_

# 모델별 기본값
my_param1: 128
my_param2: 0.1

search:
  my_param1: [64, 128, 256]
```

### 10.3 새 Feature 추가

1. **feature_injection.py 수정**:
```python
def compute_new_feature(df):
    # 새 feature 계산
    return df['new_feature']

# 메인 함수에서 호출
df['new_feature'] = compute_new_feature(df)
```

2. **load_col 설정**:
```yaml
load_col:
  inter: [session_id, item_id, timestamp, new_feature]
```

### 10.4 Grid Search 커스터마이징

```yaml
# configs/models/sasrec.yaml
search:
  hidden_size: [64, 128, 256]
  learning_rate: [0.001, 0.0005]
  num_layers: [1, 2, 3]
  # 조합 수: 3 * 2 * 3 = 18
```

```bash
python recbole_train.py model=SASRec dataset=kuairec --search
```

---

## 부록 A: 자주 사용하는 명령어

```bash
# ===== 기본 실행 =====
python recbole_train.py model=SASRec dataset=amazon_beauty

# ===== 모드 조합 =====
# SESSION + BASIC (가장 빠른 baseline)
python recbole_train.py model=SASRec dataset=kuairec eval_mode=session feature_mode=basic

# INTERACTION + FULL (가장 많은 데이터)
python recbole_train.py model=SASRec dataset=kuairec eval_mode=interaction feature_mode=full

# ===== Hyperparameter 튜닝 =====
python recbole_train.py model=SASRec dataset=kuairec \
    learning_rate=0.0005 hidden_size=256 num_layers=3 \
    hidden_dropout_prob=0.3 feature_mode=basic

# ===== wandb 로깅 =====
python recbole_train.py model=SASRec dataset=kuairec log_wandb=true

# ===== GPU 선택 =====
CUDA_VISIBLE_DEVICES=3 python recbole_train.py model=SASRec dataset=kuairec

# ===== 디버그 실행 =====
python recbole_train.py model=SASRec dataset=amazon_beauty epochs=2 feature_mode=basic

# ===== 모델 저장 =====
python recbole_train.py model=SASRec dataset=kuairec saved=true checkpoint_dir=./checkpoints
```

---

## 부록 B: 데이터셋 통계

| Dataset | Sessions | Items | Interactions | Avg Len | 권장 Batch |
|---------|----------|-------|--------------|---------|-----------|
| amazon_beauty | 757 | 3.9K | 8.6K | 11.4 | 64 |
| foursquare | 2.2K | 20K | 35K | 15.9 | 128 |
| movielens1m | 14.5K | 3.7K | 390K | 26.9 | 256 |
| retail_rocket | 153K | 48K | 540K | 3.5 | 512 |
| KuaiRec | 829K | 7.5K | 8.9M | 10.7 | 512 |
| lastfm | 856K | 14K | 12.5M | 14.6 | 512 |

---

## 부록 C: 변경 이력

| 날짜 | 변경 사항 |
|------|----------|
| 2026.01.24 | SESSION/INTERACTION 모드 구현 |
| 2026.01.24 | feature_mode (full/basic) 추가 |
| 2026.01.24 | wandb max_val_metric, run naming 개선 |
| 2026.01.24 | 불필요한 데이터 정리 (~5GB 절약) |
| 2026.01.24 | Split된 .item 파일 삭제 (~200MB 절약) |
