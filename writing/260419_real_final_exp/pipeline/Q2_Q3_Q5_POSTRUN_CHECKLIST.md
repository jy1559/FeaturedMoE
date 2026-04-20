# Q2/Q3/Q5 Post-Run Checklist

This checklist is for the moment when the currently running Q2~Q5 experiments finish and you want to turn the finished logs into notebook-ready CSVs, figure PDFs, and paper-ready TeX assets.

## Scope

Primary scope:
- Q2 routing control
- Q3 design justification
- Q5 behavior semantics

Optional scope:
- Q4 efficiency table

Main output locations:
- Logs: `experiments/run/artifacts/logs/real_final_ablation`
- CSV bundle: `writing/260419_real_final_exp/data`
- Figure PDFs: `writing/ACM_template/figures/appendix`
- Paper draft: `writing/ACM_template/sample-sigconf.tex`

## One-Command Path

Recommended command after runs finish:

```bash
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python writing/260419_real_final_exp/pipeline/postrun_q2_q5_pipeline.py
```

Recommended command with TeX compile included:

```bash
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python writing/260419_real_final_exp/pipeline/postrun_q2_q5_pipeline.py --compile-tex
```

## What The Pipeline Script Does

1. Checks that `q2`, `q3`, and `q5` summary files exist under `experiments/run/artifacts/logs/real_final_ablation`.
2. Checks that those summaries contain at least one `status=ok` row.
3. Runs `export_q2_q5_bundle.py` to refresh the notebook CSV bundle.
4. Rebuilds `q2_routing_profile.csv` from the saved result payload diagnostics.
5. Executes the Q2, Q3, and Q5 notebooks in place.
6. Verifies that the expected figure PDFs were created.
7. Optionally runs `latexmk` on `writing/ACM_template/sample-sigconf.tex`.

## Important Implementation Detail

`export_q2_q5_bundle.py` does not currently regenerate `q2_routing_profile.csv`.
The post-run script fills that gap by reading `diag_raw_test_json` from each finished Q2 result payload and rebuilding the profile CSV directly.

That means the correct refresh order is:

1. Run bundle export.
2. Rebuild Q2 routing profile.
3. Execute notebooks.
4. Check figure files.
5. Compile TeX if needed.

## Required Log Files

The script expects these files to exist after the experiment run is finished:

- `experiments/run/artifacts/logs/real_final_ablation/q2/summary.csv`
- `experiments/run/artifacts/logs/real_final_ablation/q3/summary.csv`
- `experiments/run/artifacts/logs/real_final_ablation/q5/summary.csv`

Optional for Q4:

- `experiments/run/artifacts/logs/real_final_ablation/q4/summary.csv`

## Expected CSV Outputs

The script refreshes or validates these files under `writing/260419_real_final_exp/data`:

- `q2_quality.csv`
- `q2_routing_profile.csv`
- `q3_temporal_decomp.csv`
- `q3_routing_org.csv`
- `q5_case_heatmap.csv`
- `q5_intervention_summary.csv`
- `q_suite_manifest.json`
- `q_suite_run_index.csv`

If Q4 is available, it also validates:
- `q4_efficiency_table.csv`

## Expected Figure Outputs

Q2:
- `writing/ACM_template/figures/appendix/a03_objective_variants_a.pdf`
- `writing/ACM_template/figures/appendix/a03_objective_variants_b.pdf`

Q3:
- `writing/ACM_template/figures/appendix/a02_structural_temporal.pdf`
- `writing/ACM_template/figures/appendix/a02_structural_cue_org.pdf`

Q5:
- `writing/ACM_template/figures/appendix/a05_routing_profiles_a.pdf`
- `writing/ACM_template/figures/appendix/a05_routing_profiles_b.pdf`
- `writing/ACM_template/figures/appendix/a05_routing_profiles_c.pdf`
- `writing/ACM_template/figures/appendix/a05_routing_profiles.pdf`

Q4 optional TeX snippet:
- `writing/ACM_template/figures/tab_q4_efficiency_main.tex`

## Paper Integration Status

The paper draft now expects these generated assets directly:

- Q2 uses two minipage subfigures.
- Q3 uses two minipage subfigures.
- Q5 uses three minipage subfigures in the main text and a combined panel for appendix usage.

That means once the notebooks run successfully, the generated PDFs are intended to be usable in `sample-sigconf.tex` without manual renaming.

## Manual Sanity Checks

After the script finishes, inspect these points before trusting the paper build:

1. `q2_quality.csv` has the expected four datasets for the current run plan.
2. `q3_temporal_decomp.csv` and `q3_routing_org.csv` are non-empty and use the expected variant labels.
3. `q5_case_heatmap.csv` contains the three main cases: `Repeat-heavy`, `Fast exploratory`, `Narrow-focus`.
4. `a03_objective_variants_a.pdf` and `a03_objective_variants_b.pdf` look balanced, with labels not clipped.
5. `a02_structural_temporal.pdf` and `a02_structural_cue_org.pdf` fit visually as half-column panels.
6. `a05_routing_profiles_a.pdf`, `b`, and `c` have readable annotations and titles.
7. If TeX was compiled, `sample-sigconf.pdf` has no overflow or missing-file warnings.

## If Something Fails

If CSV export fails:
- Check whether the corresponding `summary.csv` exists.
- Check whether the rows have `status=ok`.
- Check whether `result_path` inside the summary points to a real JSON result payload.

If Q2 routing profile rebuild fails:
- Check whether the Q2 result payload contains `diag_raw_test_json`.
- Check whether the referenced diagnostic JSON file exists.

If notebook execution fails:
- Re-run the specific notebook manually.
- Check whether `nbformat` and `nbclient` are available in `/venv/FMoE`.
- If they are missing, install them or run the notebook manually in VS Code.

If TeX compile fails:
- Confirm that all expected PDF assets exist under `writing/ACM_template/figures/appendix`.
- Confirm that `latexmk` is installed.

## Suggested Post-Run Sequence

1. Wait until tmux run is fully complete.
2. Run the pipeline script.
3. Inspect the generated CSV row counts.
4. Open Q2, Q3, and Q5 figures once to visually sanity-check them.
5. Compile `sample-sigconf.tex`.
6. Only then update manuscript wording or captions.
