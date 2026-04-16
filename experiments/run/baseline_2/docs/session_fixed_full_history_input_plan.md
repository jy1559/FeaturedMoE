# session_fixed + full-history-input plan

## Goal

Keep the current `session_fixed` benchmark as the default task, but add an optional history-input mode that preserves session-level targets while allowing the model input to include the same user's earlier sessions.

The target behavior is:

- current mode
  - `[[1,2,3], [4,5,6], [7,8,9,10]]`
  - `[1,2] -> 3`, `[4,5] -> 6`, `[7,8,9] -> 10`
- proposed full-history-input mode
  - `[1,2] -> 3`, `[1,2,3,4,5] -> 6`, `[1,2,3,4,5,6,7,8,9] -> 10`
  - final input is still tail-cropped by `MAX_ITEM_LIST_LENGTH`

This should be implemented as an optional protocol on top of `session_fixed`, not as a replacement for the current main benchmark.

## Why this protocol

This protocol is a cleaner fairness check than changing `USER_ID_FIELD` from `session_id` to `user_id`.

- `USER_ID_FIELD=user_id` collapses the task into full-user-sequence recommendation and removes session-level target structure.
- `session_fixed + full-history-input` keeps the same session targets and split semantics, but relaxes the model input constraint.
- This directly tests whether some baselines are hurt by short session prefixes while preserving comparability with current RouteRec / FeaturedMoE runs.

## Recommended protocol

### Default

Keep current behavior as the default:

- `eval_mode=session_fixed`
- `history_input_mode=session_only`

### New optional mode

Add:

- `history_input_mode=full_history_session_targets`

Meaning:

- target unit remains a session
- evaluation split remains `train/valid/test` by session start time
- input sequence for a target event is built from the same user's causal history before that target time
- current session prefix is still included
- earlier sessions may serve as both previous prediction targets and later context, which is valid causal reuse

## Leakage-safe evaluation policy

This is the main design choice.

### Recommended main policy: `strict_train_prefix`

Use only train-split history as cross-session context for validation and test.

- train targets: may use earlier train interactions from the same user
- valid targets: may use train interactions only
- test targets: may use train interactions only

Pros:

- strongest protection against leakage
- easy to explain in paper/rebuttal
- keeps valid/test fully isolated from each other

Cons:

- underuses observed history for later splits
- more conservative than a live online setting

### Optional supplementary policy: `rolling_observed_prefix`

Allow later splits to use earlier observed held-out splits causally.

- train targets: earlier train interactions only
- valid targets: earlier train interactions only
- test targets: earlier train + valid interactions that are strictly earlier in time

Pros:

- closer to online deployment
- stronger long-history signal for test

Cons:

- harder to defend as the main benchmark
- valid becomes part of test context, which some reviewers may dislike

### Recommendation

Use `strict_train_prefix` for the first implementation and for any main comparison table under the new protocol. If needed later, add `rolling_observed_prefix` only as a secondary sensitivity variant.

## Do not switch `USER_ID_FIELD`

Current pipeline behavior depends on session-level target generation. The safer approach is:

- keep `USER_ID_FIELD=session_id`
- keep `SESSION_ID_FIELD=session_id`
- keep current evaluator / split interpretation
- add a separate history grouping field for sequence construction

Recommended new config keys:

```yaml
history_input_mode: session_only  # session_only | full_history_session_targets
history_group_field: user_id
target_group_field: session_id
history_eval_policy: strict_train_prefix  # strict_train_prefix | rolling_observed_prefix
history_include_current_session_prefix: true
history_require_past_session: false
```

This avoids broad breakage in RecBole assumptions and keeps the change localized to sequence construction.

## Data construction logic

### Current behavior

For `session_fixed`, the code reads:

- `{dataset}.train.inter`
- `{dataset}.valid.inter`
- `{dataset}.test.inter`

and converts each split independently into sequence samples.

Result:

- validation/test cannot see train-side earlier sessions from the same user as context

### Required new behavior

When `history_input_mode=full_history_session_targets`, do not convert each split independently.

Instead:

1. Load all three split files.
2. Attach split labels to every interaction.
3. Merge them into one interaction table.
4. Sort by:
   - `history_group_field` (`user_id`)
   - `timestamp`
   - stable tiebreaker using original file order if needed
5. Generate targets split by split, but build context from the merged table according to `history_eval_policy`.

This allows a validation/test target to use the same user's earlier train interactions as context.

## Sample generation rules

### Train

Keep current session-local augmentation target rule, but expand the prefix with earlier same-user history.

Example:

- user sessions: `[[1,2,3], [4,5,6]]`
- generated train targets:
  - `[1,2] -> 3`
  - `[1,2,3,4,5] -> 6`

If a session has multiple train targets under augmentation, each one uses all earlier causal interactions.

### Valid/Test

Keep current final-item target rule per session, but allow full causal history by user.

Example:

- valid session `[4,5,6]`
- target remains `6`
- input becomes earlier allowed user history plus current-session prefix `[4,5]`

### Tail crop

After building the full causal prefix, keep the most recent `MAX_ITEM_LIST_LENGTH` interactions.

This matches the current model interface and avoids unbounded sequence growth.

## Fairness interpretation

### Better than `USER_ID_FIELD=user_id`

`USER_ID_FIELD=user_id` answers a different question: how well does the model do on a user-level sequence benchmark?

The proposed mode answers the more relevant question:

- under the same session-level targets,
- how much does allowing prior-user-history context help each model?

### Better than changing baselines only

All models should receive the same item-history privilege under this protocol.

- baselines get longer raw context
- RouteRec / FeaturedMoE also get the same longer item history
- RouteRec remains differentiated by features and MoE routing, not by exclusive access to long-range context

This is the fairest version of the proposed idea.

## Implementation plan

### Phase 1: config plumbing

Add config keys with defaults preserving current behavior.

Files likely touched:

- `experiments/config.yaml`
- optionally a small doc note in `experiments/docs/MANUAL.md`

Requirements:

- default remains fully backward compatible
- no existing command should change behavior without explicit override

### Phase 2: merged split loader path for `session_fixed`

Inside the patched benchmark build path in `experiments/recbole_patch.py`:

- detect `history_input_mode=session_only`
  - keep current split-wise conversion path
- detect `history_input_mode=full_history_session_targets`
  - load all three split interactions
  - merge with split labels
  - build per-split target index sets
  - convert using merged causal history

### Phase 3: generalized sequence builder

Refactor `_convert_inter_to_sequence()` into a more general helper.

Suggested internal split:

- `_build_sequence_samples_session_only(...)`
- `_build_sequence_samples_full_history(...)`

Or a single generalized helper that accepts:

- ordered interaction table
- target rows to materialize
- allowed history mask per target row
- grouping field for history
- grouping field for target session

Key point:

- target selection and history selection must be separate concepts

### Phase 4: cache key update

Current session split cache keys are not enough once history mode is added.

Must include:

- `history_input_mode`
- `history_group_field`
- `target_group_field`
- `history_eval_policy`
- potentially `history_include_current_session_prefix`

Otherwise cached session-only splits could be incorrectly reused.

### Phase 5: diagnostics

Add logging for sanity checks:

- average context length by split
- average number of prior sessions used by split
- fraction of valid/test samples with non-empty cross-session context
- maximum observed history length before cropping

These logs will be important for verifying the mode is actually active.

## Validation checklist

Before any large run, verify the following on one small dataset such as `beauty`.

### Functional checks

- `session_only` reproduces current sample counts exactly
- `full_history_session_targets` leaves target counts unchanged
- valid/test targets remain identical to the current protocol
- only the input prefix changes

### Causality checks

- no sample includes future interactions from the same user
- under `strict_train_prefix`, valid/test never consume held-out context from their own split or later splits
- timestamps are strictly respected when sessions share the same user

### Debug example checks

For a manually chosen user with 3 sessions:

- inspect one train target
- inspect one valid target
- inspect one test target
- confirm the exact prefix content matches the intended logic

## First experiment matrix

Do not widen to all datasets/models immediately.

Recommended first pass:

- datasets:
  - `beauty`
  - `amazon_beauty`
  - `lastfm0.03`
- models:
  - `SASRec`
  - `BSARec`
  - `GRU4Rec`
  - `RouteRec` / `FeaturedMoE_N3`

Purpose:

- `beauty`, `amazon_beauty`: short-session stress test
- `lastfm0.03`: longer-session control
- `SASRec`, `BSARec`, `GRU4Rec`: different sequence-model sensitivities
- `RouteRec`: check whether gains remain after granting all models longer history

## Reporting recommendation

If results are eventually reported, the cleanest framing is:

1. Main benchmark: current `session_fixed`
2. Sensitivity benchmark: `session_fixed + full_history_session_targets`
3. Comparison focus: per-model delta, not only absolute score

Questions to answer:

- which baselines gain the most from longer user history?
- does RouteRec remain competitive after all models receive the same expanded item-history context?
- are gains concentrated on short-session datasets?

## Non-goals for the first version

To keep the first implementation controlled, do not add all of these immediately:

- session-boundary embedding features
- time-gap reset logic between sessions
- hybrid policies that partially mask same-session history
- online updating of validation context during hyperparameter search

These can be explored later if the basic protocol is useful.

## Final recommendation

The first implementation should be:

- optional
- backward compatible
- `session_fixed` target semantics preserved
- `strict_train_prefix` as the default leakage policy
- `history_group_field=user_id` for context only

This is the most defensible and experimentally useful version of full-history-input in the current codebase.