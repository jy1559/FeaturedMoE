# P1-A: Cue-Perturbation Ablation

**Roadmap priority**: P1-A  
**Goal**: Quantify how much RouteRec's gain comes from the behavioral cue features vs. the MoE routing structure alone.

## Experiment Design

For each trained RouteRec checkpoint (KuaiRec + lastfm):
- `feature_perturb_mode=shuffle` — shuffle cue features across items (destroys content, preserves distribution)
- `feature_perturb_mode=zero`    — zero out all cue features (complete ablation)
- `feature_perturb_mode=none`    — baseline (unperturbed)

The `feature_perturb_mode` override is already implemented in `FeaturedMoE_N3` (search for `feature_perturb_mode` in `recbole/model/sequential_recommender/featured_moe_n3.py`).

## Expected Outcome

| Condition | Expected MRR@20 |
|-----------|----------------|
| none (full) | highest |
| shuffle | ≈ full − routing contribution only |
| zero | lowest (no cue signal) |

A gap between `shuffle` and `zero` indicates the cue *distribution* carries information.  
A large gap between `none` and `shuffle` confirms cue *content* drives expert selection.

## How to Run

**Pending** – will be implemented after P0 main table results are available.

Runner will load the best P0 checkpoint and re-evaluate with perturb overrides, or re-train from scratch with fixed best-config hparams + perturbation.

See `CIKM_roadmap.md` → Section P1-A for full plan.
