# Troubleshooting

## 1) Result JSON is missing
- Check root outputs:
	- experiments/run/artifacts/results/fmoe_n3
- Check mirrored paths:
	- experiments/run/artifacts/results/fmoe_n3/normal
	- experiments/run/artifacts/results/fmoe_n3/special
	- experiments/run/artifacts/results/fmoe_n3/diag

## 2) special/diag files are partially missing
- Some runs may keep summary JSON but miss diag compressed artifacts after cleanup.
- Use trial_summary.csv in diag folder as minimum available diagnostic evidence.

## 3) OOM or repeated failed trials
- Reduce train/eval batch size in run config.
- Start with dry-run and lower max-evals.
- Keep feature_ablation logging off during initial stabilization, then enable.

## 4) Baseline comparison mismatch
- Ensure SASRec baseline uses same dataset, phase, and budget envelope.
- Compare MRR@20 first, then inspect special slices for behavior differences.

## 5) Interpretation confusion
- special: robustness on difficult slices.
- diag: routing spread/collapse and balance stability signals.
- feature_ablation: dependence on feature channels.
