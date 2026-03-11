# FeaturedMoE_v4_Distillation

`run/fmoe_v4_distillation`는 `FeaturedMoE_v4_Distillation` 전용 flat-router distillation 트랙입니다.

핵심:
- base router: `flat_legacy`
- comparator router: `flat_clone_residual12`
- teacher: direct `12-way` fixed-rule logits
  - `group_local_stat12`: group별 local 통계로 `4x3` teacher logits
  - `group_comp_stat12`: group 경쟁 + group 내부 local stat semantics
  - `group_comp_shape12`: zero/high/peak/std 같은 shape 신호를 더 강하게 쓰는 경쟁형 teacher
  - `rule_soft12`: 기존 `rule_soft` ratio-bin 규칙을 learned router용 teacher logits으로 사용
- delivery: `distill_kl`, `fused_bias`, `distill_and_fused_bias`
- stage mask: `all`, `mid_micro_only`
- rule router variant:
  - `ratio_bins`: 기존 rule-soft ratio-bin router
  - `teacher_gls`: `group_local_stat12` 공식을 direct rule router weight로 사용
- 1차 범위: `ML1 only`

고정 기본값:
- `layout=7`
- `serial`
- `128 / 16 / 256 / 64`
- `expert_scale=3`
- `train/eval batch=4096/8192`
- `weight_decay=5e-5`
- `dropout=0.10`
- `balance_loss_lambda=0.005`
- `moe_top_k=0`

메인 phase:
- `phase_p0_anchor.sh`
  - `legacy_plain`, `cres_plain`
- `phase_p1_teacher_family.sh`
  - teacher family screen
- `phase_p2_delivery_stage_mask.sh`
  - delivery + stage mask screen
- `phase_p2_compare16.sh`
  - Phase 2 compare-16
  - `GLS distill` / `rule_soft teacher` / `GLS direct rule-hybrid` / `plain` / `rule_soft hybrid/full` / layout control을 한 번에 비교
  - `8 GPU x 2 combo = 16 combo`
  - `flat_legacy` 고정, `max-evals=20`, `epochs=30`, `patience=5`
- `phase_p3_router_comparator.sh`
  - `legacy` vs `CRES(0.10/0.20)`
- `phase_p4_strength_until.sh`
  - `weak / main / strong`
- `phase_pfull_distillation32.sh`
  - `8 GPU x 4 combo = 32 combo`
  - `flat_legacy` 고정, distillation 관련 축만 넓게 탐색
  - `group_comp_*`, `fused_bias`, `distill_and_fused_bias`, `mid_micro_only`를 더 두껍게 보는 weighted catalog
  - `lr`만 loguniform, `wd=5e-5`, `dropout=0.10`, `balance=0.005` 고정
  - 로그 파일은 datetime 대신 phase 폴더 기준 `000_...`, `001_...` 식으로 순번 부여

예시:

```bash
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_p0_anchor.sh --gpus 0,1 --dry-run
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_p1_teacher_family.sh --gpus 0,1 --dry-run
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_p2_delivery_stage_mask.sh --gpus 0,1,2,3 --teacher-designs group_comp_stat12,group_comp_shape12 --dry-run
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_p2_compare16.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_p3_router_comparator.sh --teacher-design group_comp_stat12 --teacher-delivery distill_and_fused_bias --teacher-stage-mask mid_micro_only --dry-run
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_p4_strength_until.sh --router-design flat_legacy --teacher-design group_comp_stat12 --teacher-delivery distill_and_fused_bias --teacher-stage-mask mid_micro_only --dry-run
bash /workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/phase_pfull_distillation32.sh --gpus 0,1,2,3,4,5,6,7 --dry-run
```

로그 파일명 해석:
- 형식: `003_C04_GLS_DB_MM_WEAK.log`
- `003`
  - 현재 phase 폴더 안 순번
- `C04`
  - combo id
- teacher 약어:
  - `GLS` = `group_local_stat12`
  - `GCS` = `group_comp_stat12`
  - `GSH` = `group_comp_shape12`
- delivery 약어:
  - `DKL` = `distill_kl`
  - `BIAS` = `fused_bias`
  - `DB` = `distill_and_fused_bias`
- stage mask 약어:
  - `MM` = `mid_micro_only`
  - `ALL` = `all`
- strength 약어:
  - `WEAK` = weak preset
  - `MAIN` = main preset
  - `STRONG` = strong preset
- control 약어:
  - `PLAIN` = teacher 없음
  - `LEGACY_HYBRID` = `mid/micro=rule_soft` comparator

예:
- `003_C04_GLS_DB_MM_WEAK.log`
  - `group_local_stat12` teacher
  - `distill_and_fused_bias`
  - `mid/micro만 teacher 적용`
  - weak strength preset
- `010_C07_GLS_DB_ALL_MAIN.log`
  - 같은 teacher
  - `distill_and_fused_bias`
  - `macro/mid/micro 전부 teacher 적용`
  - main strength preset

핵심 runner:
- [train_single.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/train_single.sh)
- [tune_hparam.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_v4_distillation/tune_hparam.sh)
