# Config System - Implementation Summary

## Problem Solved
Config loading order와 priority가 혼란스러워서 모델별 하이퍼파라미터(예: Caser의 MAX_ITEM_LIST_LENGTH=10)가 제대로 적용되지 않고 있었습니다.

## Solution Implemented

### Config Loading Order (Priority: Low → High)
```
1. config.yaml (base)
   - Default values: epochs=100, learning_rate=0.001, MAX_ITEM_LIST_LENGTH not set
   - Placeholders: model: ???, dataset: ???

2. Hydra Defaults Groups (in order):
   a) eval_mode/{mode}.yaml (eval_mode=session by default)
   b) feature_mode/{mode}.yaml (feature_mode=full by default)
   c) model/{modelname}.yaml (model=??? → must specify)
      - e.g., model=caser → loads model/caser.yaml
      - Sets model="Caser" AND model-specific params like MAX_ITEM_LIST_LENGTH=10

3. CLI Overrides (highest priority)
   - python recbole_train.py model=caser epochs=50 gpu_id=5
   - Overrides everything above
```

### Key Config Files

**configs/config.yaml**
```yaml
defaults:
  - eval_mode: session       # Can be overridden
  - feature_mode: full       # Can be overridden
  - model: ???               # MUST specify via CLI
  - _self_
  
model: ???                   # Placeholder
dataset: ???                 # Placeholder
epochs: 100
learning_rate: 0.001
# No MAX_ITEM_LIST_LENGTH here - each model defines it
```

**configs/model/caser.yaml**
```yaml
# @package _global_

model: Caser
MAX_ITEM_LIST_LENGTH: 10     # Override default for Caser
num_filters: 16
filter_sizes: [2, 3, 4]
```

**configs/model/sasrec.yaml**
```yaml
# @package _global_

model: SASRec
MAX_ITEM_LIST_LENGTH: 50     # Standard value
num_heads: 4
inner_size: 256
```

## Critical Implementation Details

### 1. `@package _global_` Directive
- **All** model/*.yaml files must have `# @package _global_` at the top
- This tells Hydra to merge the YAML content into the global config, not into a `model:` subgroup
- Without this, model settings end up in `config.model.{...}` instead of `config.{...}`

### 2. CLI Override Names Must Match Config Keys
- Use lowercase for model names: `model=caser` (not `model=Caser`)
- Hydra file matching is case-sensitive: `model/caser.yaml` (not `model/Caser.yaml`)
- The actual model value in config is set by the YAML file: `model: Caser`

### 3. Model-Specific Parameters
Each model defines its own values:
```yaml
configs/model/
├── caser.yaml           → MAX_ITEM_LIST_LENGTH: 10
├── sasrec.yaml          → MAX_ITEM_LIST_LENGTH: 50
├── gru4rec.yaml         → MAX_ITEM_LIST_LENGTH: 50
└── ... (all others)     → MAX_ITEM_LIST_LENGTH: 50
```

## Verification

Run tests/test_config_load.py to verify all config combinations work correctly:
```bash
python tests/test_config_load.py
```

Expected output shows all 5 test cases pass with correct values.

## Usage Examples

### Example 1: Caser with short sequences
```bash
python recbole_train.py model=caser dataset=amazon_beauty
```
Expected:
- model="Caser"
- MAX_ITEM_LIST_LENGTH=10
- data_path="../Datasets/processed/feature_added"
- eval_mode="session"

### Example 2: SASRec with full features and 50 epochs
```bash
python recbole_train.py model=sasrec dataset=lastfm epochs=50
```
Expected:
- model="SASRec"
- MAX_ITEM_LIST_LENGTH=50
- epochs=50 (overridden)
- num_heads=4 (from model/sasrec.yaml)

### Example 3: GRU4Rec with interaction mode and custom GPU
```bash
python recbole_train.py model=gru4rec dataset=foursquare feature_mode=basic eval_mode=interaction gpu_id=5
```
Expected:
- model="GRU4Rec"
- data_path="../Datasets/processed/basic"
- eval_mode="interaction"
- gpu_id=5 (CLI override)

## Testing Results

✅ Config loading: All combinations work correctly
✅ SASRec: Trains successfully with correct config
✅ Caser: Trains successfully with MAX_ITEM_LIST_LENGTH=10 (1 epoch: 1.20s for SASRec, 12.90s for Caser)
✅ CLI overrides: epochs, learning_rate, gpu_id all work
✅ WandB naming: Model names show correctly ("SAS" for SASRec, "CAS" for Caser)

## Files Modified

1. **configs/config.yaml**
   - Simplified defaults, clear documentation
   - model and dataset remain as placeholders
   - No MAX_ITEM_LIST_LENGTH defined here

2. **configs/model/*.yaml** (all 17 files)
   - Added `# @package _global_` directive
   - Each model now defines MAX_ITEM_LIST_LENGTH
   - Caser has MAX_ITEM_LIST_LENGTH=10, others have 50

3. **hydra_utils.py**
   - Simplified load_hydra_config() - removed override conversion logic
   - Hydra handles config group selection directly

4. **recbole_train.py**
   - Removed excessive debug logging
   - Config values are now correct

5. **New: docs/CONFIG_GUIDE.md**
   - Complete documentation of config system
   - Examples and priority explanation
   - Debugging guide

6. **New: tests/test_config_load.py**
   - Automated tests for 5 common config combinations
   - Verifies model-specific settings are applied correctly
