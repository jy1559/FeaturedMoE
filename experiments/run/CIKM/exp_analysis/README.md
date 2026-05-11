# P2-B: BRD Analysis & Routing Visualisation

**Roadmap priority**: P2-B  
**Goal**: Empirically validate the Behavioral Routing Demand (BRD) hypothesis and produce paper figures.

## Sub-experiments

### 1. BRD Correlation Analysis
- Compute per-session BRD = variance of cue features across sessions (measures how much a user's behavioral context changes).
- Split sessions into high-BRD vs. low-BRD quartiles.
- Show RouteRec gain (vs. SASRec) is larger for high-BRD sessions.

### 2. Expert Routing Distribution
- Visualise which expert each item/session gets routed to.
- Show experts specialise on different behavioral patterns (e.g. active vs. casual users).
- Heatmap: expert assignment × behavioral feature group.

### 3. Feature Family Importance
- Ablate one feature family (recency / diversity / popularity / temporal) at a time.
- Identify which family contributes most to routing quality.

## Expected Outputs

- `brd_correlation.png` – scatter plot of BRD vs. RouteRec gain per session
- `expert_routing_heatmap.png` – expert × feature-group assignment heatmap
- `feature_family_ablation.csv` – MRR@20 per ablation condition

## How to Run

**Pending** – will be implemented after P0 and P1 results.

Analysis scripts will load saved model checkpoints and run inference with diagnostic hooks.

See `CIKM_roadmap.md` → Section P2-B for full plan.
