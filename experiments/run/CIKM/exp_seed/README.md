# P2-A: Multi-Seed Robustness

**Roadmap priority**: P2-A  
**Goal**: Confirm RouteRec improvements are stable across random seeds, not seed-specific artefacts.

## Experiment Design

Re-run RouteRec + top-2 baselines (SASRec, BSARec) × [KuaiRec, lastfm] with seeds:

| Seed | Description |
|------|-------------|
| 42   | primary (already in P0) |
| 123  | robustness check 1 |
| 456  | robustness check 2 |

Report mean ± std across 3 seeds. Use best hparams from P0 (no new tuning per seed).

## How to Run

**Pending** – will be implemented after P0 results.

Runner will re-use P0 best configs from `results/main_*_summary.csv` and launch with `++seed=123` / `++seed=456` overrides.

See `CIKM_roadmap.md` → Section P2-A for full plan.
