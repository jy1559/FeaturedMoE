# P1-B: Capacity-Matched Baseline

**Roadmap priority**: P1-B  
**Goal**: Rule out that RouteRec gains come purely from a larger parameter count.

## Experiment Design

Train a `SASRec-wide` baseline with the same total parameter count as RouteRec:

```
RouteRec (N=3): base_dim=64, 3 experts, MoE overhead
SASRec-wide:    hidden_size=X  (chosen so #params matches RouteRec)
```

Parameter matching procedure:
1. Count RouteRec total parameters (use `model.num_parameters()`).
2. Derive `hidden_size` for SASRec so that `SASRec-wide` has the same count.
3. Train SASRec-wide with same fixed hparams + narrow lr/wd search.

## Expected Outcome

RouteRec should outperform SASRec-wide if the routing mechanism itself adds value beyond raw model capacity.

## How to Run

**Pending** – will be implemented after P0 results confirm RouteRec wins.

Will add a new model config or use `++hidden_size=X` override in SASRec with matching capacity.

See `CIKM_roadmap.md` → Section P1-B for full plan.
