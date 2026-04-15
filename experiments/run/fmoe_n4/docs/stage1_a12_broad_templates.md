# FMoE_N4 Stage1 A12 Broad Templates

Entry points:
- `experiments/run/fmoe_n4/stage1_a12_broad_templates.py`
- `experiments/run/fmoe_n4/stage1_a12_broad_templates.sh`

Defaults:
- dataset: `KuaiRecLargeStrictPosV2_0.2`
- templates: `8` (use `--template-count 16` for full bank)
- search budget: `max-evals=8`, `tune-epochs=25`, `tune-patience=3`

Design notes:
- A12 architecture is fixed (`A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5`).
- Stage1 only.
- Broad discrete template search over lr/max_len/d_feat_emb/expert_scale/dropouts/heads/attn/wd.
- Aux terms are limited to paper-core pair only:
  - `route_consistency_lambda`
  - `z_loss_lambda`

Examples:

```bash
bash experiments/run/fmoe_n4/stage1_a12_broad_templates.sh
```

```bash
bash experiments/run/fmoe_n4/stage1_a12_broad_templates.sh --template-count 16 --max-evals 6 --gpus 0,1
```

```bash
python experiments/run/fmoe_n4/stage1_a12_broad_templates.py --datasets KuaiRecLargeStrictPosV2_0.2 --template-count 16 --dry-run
```
