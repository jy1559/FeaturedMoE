# FMoE Baseline Experiments

Sequential recommendation baseline 실험을 위한 RecBole 기반 프레임워크입니다.

## 🚀 빠른 참조

### Config 계층 (우선순위 낮음→높음)
```
config.yaml < eval_mode < feature_mode < model < CLI override
```

### 모델별 시퀀스 길이
- **Caser**: MAX_ITEM_LIST_LENGTH=**10** (CNN 효율)
- **Others**: MAX_ITEM_LIST_LENGTH=**50** (기본값)

### 예제들
```bash
# SASRec + amazon_beauty (기본: 세션 분리, full features)
python recbole_train.py model=sasrec dataset=amazon_beauty

# Caser + 기본 features (MAX_ITEM_LIST_LENGTH=10 자동 적용)
python recbole_train.py model=caser dataset=amazon_beauty feature_mode=basic

# SASRec + interaction 모드 + 50 epoch
python recbole_train.py model=sasrec dataset=lastfm eval_mode=interaction epochs=50

# GPU 지정 및 커스텀 하이퍼파라미터
python recbole_train.py model=gru4rec dataset=foursquare gpu_id=5 learning_rate=0.0005

# Config 확인 (모든 설정이 맞는지 테스트)
python tests/test_config_load.py
```

**📖 전체 문서**: [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md)

---

## 핵심 기능

- **두 가지 Evaluation Mode**: SESSION (세션 단위 분리) / INTERACTION (세션 내 LOO)
- **Hydra config 시스템**: 계층적 설정 관리 + CLI override + 모델별 하이퍼파라미터
- **Feature support**: 50개 feature 자동 로드 및 sequence 변환
- **wandb 통합**: 실험 추적, max_val_metric 기록, 자동 run naming
- **서버 친화적**: CPU 스레드 제한, GPU 정리 스크립트

---

## 빠른 시작

```bash
cd experiments

# 기본 실행 (SESSION 모드)
python recbole_train.py model=SASRec dataset=amazon_beauty

# GPU 지정
CUDA_VISIBLE_DEVICES=6 python recbole_train.py model=SASRec dataset=amazon_beauty

# INTERACTION 모드
python recbole_train.py model=SASRec dataset=amazon_beauty eval_mode=interaction

# wandb 로깅 활성화
python recbole_train.py model=SASRec dataset=amazon_beauty log_wandb=true

# 파라미터 오버라이드
python recbole_train.py model=SASRec dataset=amazon_beauty epochs=50 learning_rate=0.0005 hidden_size=256
```

---

## Evaluation Modes

### 1. SESSION 모드 (`eval_mode=session`, 기본값)

세션을 **완전히 분리**하여 train/valid/test로 사용:

```
전체 데이터 (757 sessions)
  ↓ 시간순 정렬 후 70/15/15 분할
Train Sessions (530개) ──────────> 학습에만 사용 (all interactions, data aug)
Valid Sessions (114개) ──────────> 검증에만 사용 (마지막 item 예측)
Test Sessions (114개)  ──────────> 테스트에만 사용 (마지막 item 예측)
```

**핵심 특징:**
- **완전한 session 분리**: train/valid/test 간 session overlap 없음 (data leakage 방지)
- **분할 방식**: RecBole RS split + `group_by=user` (USER_ID_FIELD=session_id)
- **비율**: 70% / 15% / 15% (sessions 기준, interactions 아님!)
- **정렬**: 시간순 (order=TO, TIME_FIELD 기준)
- **Training**: train sessions의 모든 interactions 사용 (data augmentation)
- **Evaluation**: valid/test sessions의 마지막 item 예측 (나머지는 sequence)

### 2. INTERACTION 모드 (`eval_mode=interaction`)

모든 세션이 모델에 들어가되, 세션 **내부**에서 train/valid/test 분리:

```
전체 데이터 (757 sessions, 8635 interactions)
  ↓ 모든 sessions 사용
각 session 내에서 position-based split:
  Session: [item1, item2, item3, item4, item5]
           ├─────────────────────┤  ├──┤  ├──┤
                Train              Valid  Test
                              (item4) (item5)
```

**핵심 특징:**
- **Session overlap O**: 모든 sessions이 train/valid/test에 모두 등장 (단, 다른 interactions)
- **분할 방식**: RecBole LS (Leave-one-out Split)
- **최소 session 길이**: 3 interactions 필요
- **Training**: 각 session의 앞부분 interactions (data augmentation)
- **Evaluation**: 각 session의 마지막 2개 items (valid=4번째, test=5번째)
- **더 많은 데이터**: 모든 sessions가 학습에 기여

---

## Config 시스템

### 계층 구조

```
configs/
├── config.yaml                 # 메인 설정 (base)
├── eval_mode/
│   ├── session.yaml            # SESSION 모드 설정
│   └── interaction.yaml        # INTERACTION 모드 설정
├── feature_mode/
│   ├── full.yaml               # 전체 feature (~37GB)
│   └── basic.yaml              # 기본 컬럼만 (~2.7GB, 빠른 로딩)
├── models/                     # (선택) 모델별 오버라이드
└── datasets/                   # (선택) 데이터셋별 오버라이드
```

설정 우선순위: `config.yaml` → `eval_mode/*.yaml` → `feature_mode/*.yaml` → `models/*.yaml` → `datasets/*.yaml` → **CLI override**

### Feature Modes

| 모드 | 용량 | 컬럼 | 용도 |
|------|------|------|------|
| `feature_mode=full` | ~37GB | 50+ feature 컬럼 | FeaturedMoE, feature-aware 모델 |
| `feature_mode=basic` | ~2.7GB | session_id, item_id, timestamp, user_id | SASRec, GRU4Rec 등 기본 모델 |

```bash
# Baseline 실험 (빠른 로딩)
python recbole_train.py model=SASRec dataset=amazon_beauty feature_mode=basic

# Feature 모델 실험
python recbole_train.py model=FeaturedMoE dataset=amazon_beauty feature_mode=full
```

### 주요 Config 파라미터

#### Core
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `model` | (필수) | 모델명 (SASRec, GRU4Rec, NARM 등) |
| `dataset` | (필수) | 데이터셋명 (amazon_beauty, KuaiRec 등) |
| `dataset_root` | `../Datasets/processed` | 데이터 경로 베이스 (feature_mode에서 사용) |
| `eval_mode` | `session` | 평가 모드 (session / interaction) |
| `feature_mode` | `full` | Feature 모드 (full / basic) |

#### Device
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `gpu_id` | 0 | 사용할 GPU ID |
| `use_gpu` | true | GPU 사용 여부 |

> **Tip**: 서버에서는 `CUDA_VISIBLE_DEVICES=N`으로 GPU를 지정하는 것이 더 안전합니다.

#### Training
| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `epochs` | 100 | 10-500 | 최대 학습 epoch |
| `train_batch_size` | 256 | 32-1024 | 학습 배치 크기 |
| `eval_batch_size` | 512 | 64-2048 | 평가 배치 크기 |
| `learning_rate` | 0.001 | 1e-4 ~ 1e-2 | 학습률 |
| `stopping_step` | 10 | 5-20 | Early stopping patience |

#### Model
| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `hidden_size` | 128 | 32-512 | Hidden dimension |
| `embedding_size` | 128 | 32-512 | Embedding dimension |
| `num_layers` | 2 | 1-6 | Transformer/GRU layers |
| `num_heads` | 4 | 1-8 | Attention heads (Transformer) |
| `hidden_dropout_prob` | 0.2 | 0-0.5 | Hidden dropout |
| `attn_dropout_prob` | 0.2 | 0-0.5 | Attention dropout |

#### Sequence
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `MAX_ITEM_LIST_LENGTH` | 50 | 최대 시퀀스 길이 |

#### Evaluation
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `metrics` | [Hit, NDCG, MRR] | 평가 지표 |
| `topk` | [5, 10, 20] | Top-K 값들 |
| `valid_metric` | NDCG@10 | Early stopping 기준 metric |

#### Saving
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `saved` | false | 모델 체크포인트(.pth) 저장 |
| `save_dataset` | false | 처리된 데이터셋 저장 |
| `checkpoint_dir` | saved | 저장 디렉토리 |

#### Logging
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `log_wandb` | false | W&B 로깅 활성화 |
| `wandb_project` | FMoE_2026 | W&B 프로젝트명 |

---

## 실행 예시

### 기본 실행

```bash
# SASRec on KuaiRec
python recbole_train.py model=SASRec dataset=KuaiRec

# GRU4Rec on MovieLens
python recbole_train.py model=GRU4Rec dataset=movielens1m

# NARM on lastfm
python recbole_train.py model=NARM dataset=lastfm
```

### 파라미터 튜닝

```bash
# Learning rate 조정
python recbole_train.py model=SASRec dataset=KuaiRec learning_rate=0.0005

# 모델 크기 조정
python recbole_train.py model=SASRec dataset=KuaiRec hidden_size=256 num_layers=4

# Dropout 조정
python recbole_train.py model=SASRec dataset=KuaiRec hidden_dropout_prob=0.3
```

### Grid Search

```bash
# --search 플래그로 grid search 활성화
python recbole_train.py model=SASRec dataset=KuaiRec --search

# configs/models/sasrec.yaml에 search 섹션 정의:
# search:
#   hidden_size: [64, 128, 256]
#   learning_rate: [0.001, 0.0005]
```

### W&B 로깅

```bash
# 로깅 활성화
python recbole_train.py model=SASRec dataset=KuaiRec log_wandb=true

# 프로젝트명 변경
python recbole_train.py model=SASRec dataset=KuaiRec log_wandb=true wandb_project=MyProject
```

**Run name 형식**: `{DATA}_{MODEL}_{INFO}_{MMDDHHMM}`
- INFO 토큰: `eS`/`eI`(eval), `e{n}`(epochs, 기본 100 제외), `lr{v}`(lr≠0.001), `h{v}`(hidden_size≠128), `b{v}`(batch≠256)을 `_`로 분리
- 예: `AMA_SAS_eS_e1_01241430` (session, epochs=1)
- 예: `KUA_GRU_eI_e50_lr0.0005_01241530` (interaction, epochs=50, lr=0.0005)

**기록되는 주요 metrics**:
- `max_val_metric`: 최고 validation metric (overfitting 추적용)
- `best_valid_*`: Best validation 결과
- `test_*`: Test 결과
- `total_time_*`: 실행 시간

---

## 폴더 구조

```
2026_FMoE/
├── Datasets/
│   ├── raw/                        # 원본 백업
│   └── processed/
│       ├── basic/                  # feature 없는 경량 데이터
│       └── feature_added/          # 50+ feature 포함 데이터
│       └── create_basic_datasets.py / split.py / feature_injection.py
│
└── experiments/                    # 실험 코드 (여기로 cd 후 실행)
     ├── configs/
     │   ├── config.yaml             # 메인 Hydra config (dataset_root=../Datasets/processed)
     │   ├── eval_mode/              # SESSION / INTERACTION
     │   ├── feature_mode/           # full / basic
     │   ├── models/                 # 모델별 오버라이드
     │   └── datasets/               # 데이터셋별 오버라이드
     ├── recbole_train.py            # 메인 실행 스크립트
     ├── recbole_patch.py            # RecBole 패치 (SESSION 지원)
     ├── hydra_utils.py              # Hydra config 유틸리티
     ├── docs/                       # 운영/설정/변경 문서
     ├── tools/                      # 유지보수 유틸 스크립트
     ├── tests/                      # 점검용 테스트 스크립트
     ├── saved/                      # 모델 체크포인트
     ├── log/                        # 로그 파일
     └── wandb/                      # W&B 로그
```

---

## 지원 모델

| Model | 특징 | 추천 사용 |
|-------|------|----------|
| SASRec | Causal self-attention | 기본 baseline |
| GRU4Rec | GRU-based sequential | 빠른 학습 |
| NARM | GRU + attention | SASRec 대안 |
| STAMP | Short-term attention | Session-based |
| BERT4Rec | Bidirectional, masked | Pre-training 있을 때 |
| SR-GNN | Session graph + GNN | Graph 특성 활용 |

---

## 데이터셋

| Dataset | Sessions | Items | 특성 |
|---------|----------|-------|------|
| amazon_beauty | 757 | 3.9K | 소규모, sanity check |
| foursquare | 2.2K | 20K | 위치 기반 |
| KuaiRec | 829K | 7.5K | 대규모, 메인 실험 |
| lastfm | 856K | 14K | 음악, 메인 실험 |
| movielens1m | 14.5K | 3.7K | 영화, 강한 baseline |
| retail_rocket | 153K | 48K | E-commerce |

---

## 유틸리티

### GPU 프로세스 정리

```bash
# GPU 프로세스 확인
./tools/cleanup_gpu.sh

# 프로세스 종료
./tools/cleanup_gpu.sh --kill
```

---

## 트러블슈팅

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
python recbole_train.py ... train_batch_size=64 eval_batch_size=128
```

### 느린 데이터 로딩
Feature가 많은 데이터셋은 로딩이 오래 걸립니다. 대조군 실험은 basic 데이터셋 사용을 권장합니다.

### wandb 오류
```bash
wandb login  # 로그인 확인
```
