# FMoE_N

Track directory for the lightweight `FeaturedMoE_N` line.

- model aliases: `FeaturedMoE_N`, `featured_moe_n`, `featuredmoe_n`
- intended artifact/log track name: `fmoe_n`
- scope: lightweight flat router baseline with shared feature bank and aggregated special metrics
- implementation notes: `IMPLEMENTATION_NOTES.md`
- next-phase roadmap: `ROADMAP_NEXT.md`
- common tune entrypoint: `tune_hparam.sh`
- phase summary updater: `update_phase_summary.py`
- state workspaces:
  - `S00_base_v1/`
    - baseline P0 / P0.5 rerun state
    - contains `phase_p0_anchor.sh`, `phase_p0_5_lr_narrow.sh`, `P0_P05_PLAN.md`
  - `S01_layout_lite_v1/`
    - lighter-layout architecture probe state
    - contains `phase_arch_probe.sh`, `ARCH_LITE_PLAN.md`
- live phase summary output: `experiments/run/artifacts/logs/fmoe_n/<AXIS>/<PHASE>/<DATASET>/<PHASE>_summary.csv`
- special metric mirrors:
  - normal result mirror: `experiments/run/artifacts/results/fmoe_n/normal/<AXIS>/<PHASE>/<DATASET>/<MODEL>/`
  - special result mirror: `experiments/run/artifacts/results/fmoe_n/special/<AXIS>/<PHASE>/<DATASET>/<MODEL>/`
  - special log-style mirror: `experiments/run/artifacts/logs/fmoe_n/special/<AXIS>/<PHASE>/<DATASET>/<MODEL>/`
- state-specific manifests:
  - P0: `experiments/run/artifacts/inventory/fmoe_n/p0_manifest_<STATE>_latest.json`
  - P0.5: `experiments/run/artifacts/inventory/fmoe_n/p05_manifest_<STATE>_latest.json`
- axis/state rule:
  - `tune_hparam.sh --state-tag S01_layout_lite_v1`를 주면 axis가 해당 state로 고정된다.
  - baseline anchor는 `S00_base_v1`, lite layout probe는 `S01_layout_lite_v1`를 기본값으로 둔다.
  - `S00_*`, `S01_*`, `S02_*`처럼 앞번호를 붙여 구조 진화 순서를 남긴다.
- recommended entrypoints:
  - `bash experiments/run/fmoe_n/S00_base_v1/phase_p0_anchor.sh --gpus 0,1,2,3`
  - `bash experiments/run/fmoe_n/S00_base_v1/phase_p0_5_lr_narrow.sh --gpus 0,1,2,3 --state-tag S00_base_v1`
  - `bash experiments/run/fmoe_n/S01_layout_lite_v1/phase_arch_probe.sh --gpus 0,1,2,3`
