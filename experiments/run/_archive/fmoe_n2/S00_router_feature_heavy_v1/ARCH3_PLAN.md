# ARCH3

`ARCH3` is the first `fmoe_n2` probe on `KuaiRecLargeStrictPosV2_0.2`.

Core axes:

- Wave 1: strong plain anchors
- Wave 2: `MoE_off` and `pure_attention` controls
- Wave 3: `full_rule`, `hybrid`, `bias`
- Wave 4: feature-only learned router
- Wave 5: feature-heavy mixed router
- Wave 6: expert/group/capacity ablations
- Wave 7: auxiliary loss / regularizer
- Wave 8: scheduler comparison on one promising base

Definitions:

- `MoE_off`: same serial skeleton, but each MoE block is replaced by `dense_ffn`
- `pure_attention`: same serial skeleton, but each MoE block is replaced by `identity`
- `full_rule`: all active MoE stages use `rule_soft`
- `feature-only learned router`: learned router with `router_use_hidden=false` and `router_use_feature=true`

Budget:

- `32 combos`
- `8 waves`
- `4 GPU`
- `epochs=100`
- `patience=10`
- `max_evals=3`

Default command:

```bash
bash /workspace/jy1559/FMoE/experiments/run/fmoe_n2/S00_router_feature_heavy_v1/phase_arch3.sh --gpus 0,1,2,3
```
