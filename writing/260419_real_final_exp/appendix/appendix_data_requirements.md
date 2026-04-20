# Appendix Data Requirements

This document maps each appendix notebook cell to the CSV data it consumes, the figure or table it produces, and the real experiments needed to replace demo-dummy values.

---

## 1. CSV Files and Their Source Experiments

| CSV file | Notebook | What real experiment produces it |
|---|---|---|
| `appendix_dataset_stats.csv` | A01 | Static — computed from processed datasets (already real in tex) |
| `appendix_full_results_long.csv` | A01 | Baseline + RouteRec full eval runs across 6 datasets |
| `appendix_structural_variants.csv` | A02 | RouteRec structural ablation sweep (cue_org × temporal variants per dataset) |
| `appendix_sparse_tradeoff.csv` | A03 | RouteRec sparse-routing design sweep (dense / flat / top-k variants per dataset) |
| `appendix_sparse_diagnostics.csv` | A03 | RouteRec routing diagnostics: group entropy, n_eff, top1_frac per stage / variant |
| `appendix_objective_variants.csv` | A03 | RouteRec objective/regularization ablation (KNN / z-loss / balance variants) |
| `appendix_routing_diagnostics.csv` | A03 | Routing diagnostics per objective variant (same schema as sparse_diagnostics) |
| `appendix_special_bins.csv` | A04 | Session-length and target-frequency bin eval: RouteRec + SASRec + best baseline |
| `appendix_behavior_slice_quality.csv` | A04 | Behavior-slice MRR@20 + route_concentration per model per group |
| `appendix_behavior_slice_profiles.csv` | A04/A05 | Feature profile per behavioral group (feature_name, feature_value) |
| `appendix_intervention_summary.csv` | A05 | Routing intervention experiments: cue masking / group override per dataset |
| `appendix_case_routing_profile.csv` | A05 | Routing profile by stage for selected behavioral groups (heatmap data) |
| `appendix_diagnostic_case_profile.csv` | A05 | Diagnostic case profiles (same schema as case_routing_profile) |
| `appendix_transfer_summary.csv` | A06 | Cross-dataset / low-resource transfer experiments per data_fraction |
| `appendix_cost_summary.csv` | A06 | Param count + train/infer time ratios per model (can be measured on 1 dataset) |
| `appendix_selected_runs.csv` | A01/A06 | Best-run metadata (checkpoint path, result_json) per dataset × model |

---

## 2. Notebook → .tex Figure / Table Mapping

| Notebook | Cell | Produces | .tex label | .tex location |
|---|---|---|---|---|
| A01 | table cell | Table B1 dataset stats | `tab:appendix-datasets` | `app:data-details` — **already real** |
| A01 | table cell | Table B3 full results | `tab:appendix-full-seen` | `app:data-details` — **already real** |
| A02 | cell (a) | `fig_D_cue_org_variants.pdf` | `fig:appendix-stage-layout` panel (a) | `app:stage-layout` (`figure*`) |
| A02 | cell (b) | `fig_D_temporal_variants.pdf` | `fig:appendix-stage-layout` panel (b) | `app:stage-layout` (`figure*`) |
| A03 | Fig E1 cell | `fig_E1_sparse_variants.pdf` | `fig:appendix-sparse-design` | `app:sparse-design` (`figure*`) |
| A03 | Fig F1 cell | `fig_F1_obj_variants.pdf` | `fig:appendix-objective` | `app:objective` (`figure*`) |
| A03 | Fig E2(a) cell | entropy line plot | `fig:appendix-diagnostics` panel (b) | `app:diagnostics` |
| A03 | Fig E2(b)/H1 cell | n_eff lines + heatmaps | `fig:appendix-diagnostics` panels (b)(c) | `app:diagnostics` |
| A04 | Fig C1(a) cell | session-length bin lines | `fig:appendix-special-bins` panel (a) | `app:special-bins` |
| A04 | Fig C1(b) cell | freq bin lines | `fig:appendix-special-bins` panel (b) | `app:special-bins` |
| A04 | Fig I1(a) cell | behavior-slice bar+line | behavior-slice figure | `app:behavior-slices` |
| A04 | Fig I1(b) cell | relative gain bars | behavior-slice figure | `app:behavior-slices` |
| A05 | Fig J1(a) cell | score-drop bar chart | intervention figure (a) | `app:qualitative-cases` |
| A05 | Fig J1(b) cell | routing profile heatmaps | intervention figure (b) | `app:qualitative-cases` |
| A05 | Table J1 cell | routing family pivot table | Table J1 | `app:qualitative-cases` |
| A06 | Fig K1(a) cell | low-resource MRR@20 curves | transfer figure (a) | `app:transfer` |
| A06 | Fig K1(b) cell | relative gain over no-transfer | transfer figure (b) | `app:transfer` |
| A06 | Table K1 cell | full-data summary table | Table K1 | `app:transfer` |

---

## 3. Experiments Still Needed (all CSVs currently `data_status = demo_dummy`)

All CSVs were generated with synthetic values by `make_appendix_dummy.py`. The following real experiments are needed to fill them:

### Priority 1 — Main ablations (support Q3/Q4/Q5)
| Experiment | Datasets | Config changes | Output goes to |
|---|---|---|---|
| **Structural ablation** — cue_org variants | beauty, foursquare, KuaiRec, ML-1M | `variant_group=cue_org`: family_intact, fewer_groups, shuffled, flat_bag, random_group | `appendix_structural_variants.csv` (cue_org rows) |
| **Structural ablation** — temporal variants | beauty, foursquare, KuaiRec, ML-1M | macro_only, macro+mid, local_first, global_late, dup_mid | `appendix_structural_variants.csv` (temporal rows) |
| **Sparse routing design** | beauty, foursquare, KuaiRec, ML-1M | dense, flat_top6, 4gr2ex, 2gr4ex, 3gr2ex(main), 2gr1ex, 3gr3ex | `appendix_sparse_tradeoff.csv` |
| **Objective ablation** | beauty, foursquare, KuaiRec, ML-1M | no_aux, knn_only, zloss_only, balance_only, cons+zloss, full | `appendix_objective_variants.csv` |

### Priority 2 — Routing diagnostics (support Q5)
| Experiment | Notes | Output goes to |
|---|---|---|
| Routing stats per stage (entropy, n_eff, top1_frac) | Computed from saved model, no retraining | `appendix_sparse_diagnostics.csv`, `appendix_routing_diagnostics.csv` |
| Routing profile per behavioral group | Post-hoc from saved model | `appendix_case_routing_profile.csv`, `appendix_diagnostic_case_profile.csv` |

### Priority 3 — Bin / slice / intervention analysis
| Experiment | Notes | Output goes to |
|---|---|---|
| Session-length + freq bin eval | RouteRec vs. SASRec vs. best baseline per dataset | `appendix_special_bins.csv` |
| Behavioral slice eval | 4 slices: repeat-heavy, fast-tempo, narrow-focus, exploration-heavy | `appendix_behavior_slice_quality.csv` |
| Cue-masking interventions | Mask macro/mid/micro cues → re-eval, no retrain | `appendix_intervention_summary.csv` |

### Priority 4 — Optional / transfer
| Experiment | Notes | Output goes to |
|---|---|---|
| Low-resource transfer (data_fraction = 0.1/0.25/0.5/1.0) | Cross-dataset pretraining variants | `appendix_transfer_summary.csv` |
| Cost / efficiency measurement | Single dataset; measure params + wall-clock ratio | `appendix_cost_summary.csv` |

---

## 4. .tex Placeholders Remaining

| .tex label | Status | Placeholder type | Blocked on |
|---|---|---|---|
| `fig:appendix-special-bins` (a)(b) | `\placeholderpanel` | figure | `appendix_special_bins.csv` real data + A04 savefig |
| `fig:appendix-stage-layout` (a)(b) | `\includegraphics` pointing to `data/fig_D_*.pdf` | figure (PDFs) | A02 run with real data |
| `fig:appendix-sparse-design` | `\includegraphics` pointing to `data/fig_E1_sparse_variants.pdf` | figure (PDF) | A03 run with real data |
| `fig:appendix-objective` | `\includegraphics` pointing to `data/fig_F1_obj_variants.pdf` | figure (PDF) | A03 run with real data |
| `tab:appendix-efficiency` | `--` in table rows | LaTeX table | `appendix_cost_summary.csv` real data + A06 cell |
| `fig:appendix-diagnostics` (a)(b)(c) | `\placeholderpanel` | figure | `appendix_sparse_diagnostics.csv` real data + A03 savefig |
| behavior-slice figures | `\placeholderpanel` or missing | figure | `appendix_behavior_slice_quality.csv` real data + A04 savefig |
| intervention figures in `app:qualitative-cases` | `\placeholderpanel` or missing | figure | `appendix_intervention_summary.csv` real data + A05 savefig |
| transfer figures in `app:transfer` | not yet wired | figure | `appendix_transfer_summary.csv` real data + A06 savefig |

---

## 5. What Is Already Real in .tex

| .tex label | Content | Status |
|---|---|---|
| `tab:appendix-datasets` | Dataset statistics (Beauty/Foursquare/KuaiRec/LastFM/ML-1M/Retail Rocket) | ✅ Real values |
| `tab:appendix-full-seen` | Full 6×10×9 results table | ✅ Real values |
| `tab:appendix-main-setting` | Fixed config table | ✅ Static reference |
| `tab:appendix-implementation-spec` | Implementation spec | ✅ Static reference |
| `tab:appendix-feature-families` | Cue families reference | ✅ Static reference |
| `tab:appendix-final-grid` | Tuning grid | ✅ Static reference |
| `tab:appendix-special-bins` | Bin definitions | ✅ Static reference |
| `tab:appendix-slice-definitions` | Behavioral slice definitions | ✅ Static reference |
