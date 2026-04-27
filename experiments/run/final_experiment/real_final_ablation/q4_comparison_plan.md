# Q4 Main-Text Redesign Plan

## Recommendation

Do not use efficiency as the main-text Q4.

Use a two-part RouteRec-specific question instead:

"Does RouteRec remain effective when only lighter, more portable cues are available, and does the trained model actually rely on those cues at evaluation time?"

This is closer to the paper's claim than a sparse-efficiency section because it tests both cue portability and cue utility.

## Final `1 + 3 + 3` design

Q4 is split into two panels.

### Panel A: reduced-cue availability

Retrain the model under four availability settings:

- `full`
- `remove_category`
- `remove_time`
- `portable_core`

Interpretation:

- `full`: all behavioral cues available
- `remove_category`: remove group or category-style cues
- `remove_time`: remove timing-style cues
- `portable_core`: keep only the most portable core subset

This panel answers whether RouteRec still holds up when the cue budget is reduced.

### Panel B: fixed-model cue efficacy

Take one trained `full` checkpoint per dataset and evaluate it under four cue conditions:

- `intact`
- `zero_all`
- `position_permute`
- `cross_sample_permute`

Interpretation:

- `intact`: reference evaluation
- `zero_all`: remove cue information entirely at evaluation time
- `position_permute`: keep cue values but break within-sequence alignment
- `cross_sample_permute`: keep the global cue distribution but attach cues to the wrong samples

This panel answers whether the trained model truly uses cue input at evaluation time rather than only benefiting from cue-assisted training.

## Why this is stronger than efficiency-first Q4

- It directly tests the paper's cue-portability claim.
- It is specific to RouteRec rather than generic sparse-MoE practicality.
- It avoids overclaiming from route visualizations when expert separation is not strong enough.
- It produces one training-time robustness result and one evaluation-time reliance result.

## Recommended plots

### Panel A

- grouped bars or lines
- x-axis: `Full`, `No Group`, `No Time`, `Portable Core`
- y-axis: seen-target `MRR@20`
- one series per dataset

### Panel B

- grouped drop bars or slope chart
- x-axis: `Intact`, `Zero All`, `Intra-Sequence Permute`, `Cross-Sample Permute`
- y-axis: delta in seen-target `MRR@20` relative to `Intact`
- one group per dataset

## Current implementation mapping

- Panel A runner: `q4_portability.py`
- Panel A settings: `q4_portability_settings()` in `common.py`
- Panel B runner: `q4_eval_feature_efficacy.py`
- Panel B settings: `q4_feature_efficacy_specs()` in `common.py`
- Export tables: `q4_portability_table.csv`, `q4_feature_efficacy.csv`

## Minimal paper claim

The minimum defensible claim for Q4 should be:

"RouteRec degrades gracefully when some cue families are removed during training, and the final model loses quality when cue input is corrupted at evaluation time."

That is enough for a main-text Q4 and aligns with the RouteRec narrative better than an efficiency table.
[- Q2: five-way routing-source comparison as a compact slope or grouped-bar figure]
[- Q3: two-panel architecture ablation figure already planned]
[- Q4: retention plot plus slice-sensitivity heatmap]
[- Q5: representative case visualizations with short qualitative reading]

[This makes each question visually distinct.]
- train time becomes less noisy than one-epoch timing
[The rhythm becomes:]
- inference time remains simple and reproducible
[- Q2: what should control routing?]
[- Q3: why this routed structure?]
[- Q4: does the idea survive under weaker observable signal?]
[- Q5: what does the router actually respond to?]
- active parameters remain the more robust fairness signal when runtime is sensitive to implementation details
[## What to report numerically]

[Avoid repeating the full nine-metric table in Q4.]

[Use one anchor metric only:]
## Recommended paper presentation
[- `MRR@20` as the main scalar]

[Then add one derived retention value:]
### Main table
[- `retention = reduced_cue_mrr20 / full_cue_mrr20`]

[This gives a more compact and less repetitive story than copying another metric table.]
Use KuaiRec as the primary dataset. Add Foursquare or LastFM as one supporting dataset if space allows.
[Suggested concise table if needed:]

[- Dataset]
[- Full]
[- No Category]
[- No Time]
[- Sequence Only]
[- Retention at best reduced setting]
Suggested columns:
[That is enough.]

[## Where slices should appear]
- Dataset
[Do not make slices a standalone big section.]
- Model / row type
[Use them only where they strengthen the semantic claim.]
- MRR@20
[Best use:]
- NDCG@10
[- in Q4 Panel B, to show family sensitivity under behavior-defined regimes]
[- in Q5, to pick representative cases or summarize route patterns]
- Total params (M)
[Bad use:]
- Active params (M)
[- large generic slice table with many bins and many metrics]
- Train x vs SASRec
[That would feel appendix-like and dilute the story.]
- Infer x vs SASRec
[## Strongest paper narrative after this change]

[If Q4 is redesigned this way, the main text says:]
Suggested row order:
[1. RouteRec improves ranking quality.]
[2. The gain comes from behavior-guided routing, not just adding experts.]
[3. The selected multi-stage sparse design is justified.]
[4. The method still works when behavior cues are partially missing or reduced.]
[5. The learned routes respond to behavior in interpretable ways.]

[That is a much tighter narrative than inserting a generic efficiency question between design justification and routing semantics.]
1. SASRec
[## What to move to the appendix]
2. FAME
[Move or keep in appendix:]
3. RouteRec dense reference
[- sparse efficiency summary]
[- active-parameter matching against SASRec/FAME]
[- runtime ratios]
[- larger structural sweeps]
4. RouteRec active-match to FAME or SASRec
[These are useful, but they should support the main RouteRec story rather than define it.]
5. RouteRec best screened point
[## Minimal experiment plan]

[If budget is tight, the cheapest convincing version is:]
If there is space, add the quality-retained low-active point as an extra row.
[1. Run reduced-cue Q4 on KuaiRec and Foursquare only.]
[2. Report `MRR@20` retention for `Full`, `No Category`, `No Time`, `Sequence Only`.]
[3. Reuse Q5 interventions to build one small slice-by-family sensitivity heatmap.]

[This is enough for a main-text Q4.]
### Figure
[## Best-case expanded version]

[If there is room and time, the strongest Q4 package is:]
The strongest supporting figure is a frontier scatter.
[1. Panel A: reduced-cue retention plot]
[2. Panel B: slice-by-family sensitivity heatmap]
[3. Appendix table: efficiency summary and active-budget matches]

[This gives portability, semantic specificity, and practicality without making the paper repetitive.]
Recommended plot:
[## Bottom line]

[The better main-text Q4 is not "Is RouteRec efficient?"]
- x-axis: active parameters
[It is:]
- y-axis: test MRR@20
["Does RouteRec remain effective when only weaker, more portable behavioral cues are available, and do the missing cues matter in the behavioral regimes where the method claims they should matter?"]
- point color: model family or match rule
[That question is more novel, more method-specific, and easier to present without repeating Q2/Q3.]
- point shape:
  - reference baselines
  - RouteRec screen points
  - selected matched points

Why this figure helps:

- it shows that the final chosen sparse point is not isolated
- it shows the local trade-off surface
- it makes the active-parameter fairness story visually obvious


## How to write the claim carefully

Preferred wording:

- RouteRec does not win by raw parameter pool alone; under the screened sparse portfolio, competitive quality is retained even when comparison is anchored by active parameter count.
- Although total parameter pool is larger, only a subset is active per decision, and the matched rows show that quality remains strong at comparable active capacity.
- Relative to FAME, the comparison is not only "MoE versus non-MoE": it also asks whether a behavior-guided sparse router reaches similar or better quality at a similar active budget, or reaches similar quality with a smaller executed budget.
- Runtime is reported as supporting practicality evidence, but the main fairness axis is active parameter count because wall-clock time can vary with implementation-level optimization.

Avoid claiming:

- RouteRec is universally faster.
- Sparse RouteRec always reduces runtime.

Those claims are harder to defend because kernel-level optimization and routing overhead can blur wall-clock wins.


## Recommended run order

Fast paper-focused order:

1. KuaiRec only, dry-run to inspect the sweep portfolio.
2. KuaiRec only, real run with `benchmark_epochs=3`.
3. Add one supporting dataset: `foursquare` or `lastfm0.03`.
4. Export notebook data and decide whether the paper needs only the main table or table plus frontier figure.

Example commands:

```bash
cd /workspace/FeaturedMoE
bash experiments/run/final_experiment/real_final_ablation/q4_efficiency.sh --dry-run
```

```bash
cd /workspace/FeaturedMoE
Q4_BENCHMARK_DATASETS=KuaiRecLargeStrictPosV2_0.2 \
Q4_MAX_ROUTE_SCREEN_RUNS=10 \
bash experiments/run/final_experiment/real_final_ablation/q4_efficiency.sh
```

```bash
cd /workspace/FeaturedMoE
Q4_BENCHMARK_DATASETS=KuaiRecLargeStrictPosV2_0.2,lastfm0.03 \
Q4_MAX_ROUTE_SCREEN_RUNS=12 \
bash experiments/run/final_experiment/real_final_ablation/q4_efficiency.sh
```


## Practical recommendation

For the main paper, the most convincing Q4 is not a broad tuning table.
It is:

- a small but diverse sparse portfolio
- a shared-path anchor (`SASRec`) and an MoE anchor (`FAME`)
- an active-parameter matched comparison row
- a quality-matched comparison row to the MoE baseline
- a best screened quality row
- one simple frontier figure in appendix or main text if space permits

That is enough to argue that RouteRec's advantage is not just coming from a larger dormant parameter pool, while still keeping the experiment fast enough to iterate.


## Keep or drop Q4?

Keep Q4 in the main text only if the confirmed rows say something RouteRec-specific:

- the gain is not explained only by a larger dormant expert pool
- compared with FAME, behavior-guided routing reaches similar quality at lower active budget, or better quality at similar active budget
- compared with SASRec, sparsity does not collapse the ranking gain

If the confirmed rows do not make that story clearly, Q4 should move to the appendix rather than stay as a weak main-paper efficiency section.