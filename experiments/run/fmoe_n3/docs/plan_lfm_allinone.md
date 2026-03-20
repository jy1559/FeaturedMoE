# LFM All-in-One Plan (`phase_lfm_allinOne.sh`)

작성일: 2026-03-19  
대상: `lastfm0.03` + `FeaturedMoE_N3`

## 1) 목적
- phase0~phase2를 하나의 엔트리에서 실행한다.
- anchor(고정값)과 suite(변화축)를 분리해서 해석 충돌을 줄인다.
- 모든 hyperopt는 learning rate만 튜닝한다.

## 2) 실행 파일
- shell wrapper: `experiments/run/fmoe_n3/phase_lfm_allinOne.sh`
- python launcher: `experiments/run/fmoe_n3/run_phase_lfm_allinone.py`

## 3) Anchor (고정값)
공통:
- `d_feat_emb=16` (feature embedding 차원은 suite에서 변경)
- 기본 router recipe: `standard + both + gated_bias + global_flat`

### `AN_S`
- `embedding_size=96`, `d_ff=192`
- `d_router_hidden=64`, `d_expert_hidden=128`, `expert_scale=3`
- `weight_decay=1e-6`, `hidden_dropout_prob=0.10`, `attn_dropout_prob=0.10`
- `lr_range=[1.0e-4, 4.0e-3]`

### `AN_M`
- `embedding_size=128`, `d_ff=256`
- `d_router_hidden=96`, `d_expert_hidden=192`, `expert_scale=4`
- `weight_decay=1e-6`, `hidden_dropout_prob=0.15`, `attn_dropout_prob=0.10`
- `lr_range=[8.0e-5, 6.0e-3]`

### `AN_L`
- `embedding_size=192`, `d_ff=384`
- `d_router_hidden=128`, `d_expert_hidden=256`, `expert_scale=5`
- `weight_decay=1e-5`, `hidden_dropout_prob=0.20`, `attn_dropout_prob=0.15`
- `lr_range=[5.0e-5, 4.0e-3]`

## 4) Suite 축
- `layout_suite`
- `router_suite`
- `feature_embed_suite` (`d_feat_emb={12,16,24,32}`)
- `feature_family_mask_suite` (feature_embed와 독립)
- `topk_suite`
- `expert_scale_suite`
- `seq_len_suite`
- `aux_balance_suite`
- `aux_spec_suite`
- `residual_suite`

추가 결합:
- `router_x_feature_embed` 매트릭스
  - `standard/factored` x `d_feat_emb(4종)`

## 5) Hyperopt 정책 (고정)
- 탐색 대상: `search.learning_rate` only (`loguniform`)
- 고정 대상: `weight_decay`, `hidden_dropout_prob`, `attn_dropout_prob`, 기타 search 항목

## 6) 로깅 규칙 (phase7 스타일)
- logs:
  - `experiments/run/artifacts/logs/fmoe_n3/phase_lfm_allinone_v1/P{phase}/{dataset}/FMoEN3/*.log`
- logging bundle:
  - `fmoe_diag_logging=true`
  - `fmoe_special_logging=true`
  - `fmoe_logging_output_root=run/artifacts/logging`
- 로그 헤더/manifest 필수 필드:
  - `anchor_id`, `suite_id`, `variant_id`, `router_variant`, `d_feat_emb`, `lr_range`, `axis_tags`

## 7) 실행 예시
### 전체 dry-run
```bash
bash experiments/run/fmoe_n3/phase_lfm_allinOne.sh --dry-run
```

### router + feature_embed만
```bash
bash experiments/run/fmoe_n3/phase_lfm_allinOne.sh \
  --suites router_suite,feature_embed_suite \
  --dry-run
```

### phase1~2만 실행
```bash
bash experiments/run/fmoe_n3/phase_lfm_allinOne.sh \
  --from-phase 1 --to-phase 2
```

### 특정 anchor만
```bash
bash experiments/run/fmoe_n3/phase_lfm_allinOne.sh \
  --anchors AN_M
```

## 8) 검증 체크리스트
1. dry-run에서 anchor 고정값(wd/dropout/attn_dropout/expert size) 유지 확인  
2. `feature_embed_suite` 단독에서 `d_feat_emb`만 변경되는지 확인  
3. `router_suite + feature_embed_suite` 결합에서 standard/factored 매트릭스 확인  
4. command에 LR-only search만 포함되는지 확인  
5. logs/logging 산출물의 메타 태그가 누락 없는지 확인
