# Config System Documentation

## Overview
Configuration is managed through **Hydra** with hierarchical composition:

```
config.yaml (base with defaults)
    ↓
Defaults groups loaded in order:
  1. eval_mode/session.yaml  (or eval_mode/interaction.yaml)
  2. feature_mode/full.yaml  (or feature_mode/basic.yaml)
  3. model/caser.yaml        (loaded based on model=caser CLI override)
    ↓
CLI overrides applied last (highest priority)
```

## Usage

### Basic run
```bash
python recbole_train.py model=caser dataset=amazon_beauty
```

### With different eval/feature modes
```bash
python recbole_train.py model=sasrec dataset=lastfm eval_mode=interaction feature_mode=basic
```

### With hyperparameter overrides
```bash
python recbole_train.py model=caser dataset=amazon_beauty epochs=50 learning_rate=0.0005 gpu_id=5
```

## Config Priority (lowest to highest)
1. **config.yaml** - Base defaults
   - model: ??? (placeholder)
   - dataset: ??? (placeholder)
   - epochs: 100, learning_rate: 0.001, etc.
   - loss_type: CE, train_neg_sample_args: null, etc.

2. **eval_mode/{mode}.yaml** - Session vs Interaction split configuration
   - eval_mode=session: 70/15/15 session split (default)
   - eval_mode=interaction: Within-session LOO split

3. **feature_mode/{mode}.yaml** - Feature set and data path
   - feature_mode=full: Full engineered features (~37GB)
   - feature_mode=basic: Core columns only (~2.7GB)

4. **model/{modelname}.yaml** - Model-specific hyperparameters
   - model=caser: MAX_ITEM_LIST_LENGTH=10, num_filters=16, filter_sizes=[2,3,4]
   - model=sasrec: num_heads=4, inner_size=256
   - model=gru4rec: hidden_size=128, num_layers=1
   - etc.

5. **CLI Overrides** - Command-line arguments (highest priority)
   - python recbole_train.py model=caser epochs=50 → overrides everything above
   - model=caser → loads model/caser.yaml AND sets model field to "Caser"

## Key Parameters

### Dataset Selection
```yaml
dataset: ???              # Must be specified: dataset=amazon_beauty|lastfm|kuairec|...
```

### Model Selection  
```yaml
model: ???               # Must be specified: model=caser|sasrec|gru4rec|...
                         # Loads from model/{modelname}.yaml
                         # Sets the "model" field in config
```

### Mode Selection
```yaml
eval_mode: session       # session (default) or interaction
feature_mode: full       # full (default) or basic
```

### Important Model-Specific Parameters
```yaml
MAX_ITEM_LIST_LENGTH: 50  # Sequence length
                          # Overridden per model (Caser=10, others=50)

loss_type: CE            # Cross-entropy with in-batch negatives
train_neg_sample_args: null  # No explicit sampling (in-batch sufficient)

num_filters: 16          # Caser-specific
hidden_size: 128         # General default
num_heads: 4             # Transformer models (SASRec, etc.)
```

## Examples with Expected Config Values

### Example 1: Caser with basic features
```bash
python recbole_train.py model=caser dataset=amazon_beauty feature_mode=basic
```
Expected config values:
- model: "Caser"
- dataset: "amazon_beauty"  
- MAX_ITEM_LIST_LENGTH: 10  (from model/caser.yaml)
- num_filters: 16  (from model/caser.yaml)
- data_path: "../Datasets/processed/basic"  (from feature_mode/basic.yaml)
- eval_mode: "session"  (default)

### Example 2: SASRec with interaction mode, 50 epochs
```bash
python recbole_train.py model=sasrec dataset=lastfm eval_mode=interaction epochs=50
```
Expected config values:
- model: "SASRec"
- dataset: "lastfm"
- MAX_ITEM_LIST_LENGTH: 50  (from model/sasrec.yaml)
- num_heads: 4  (from model/sasrec.yaml)
- epochs: 50  (CLI override)
- data_path: "../Datasets/processed/feature_added"  (from feature_mode/full.yaml - default)

### Example 3: GRU4Rec with all custom params
```bash
python recbole_train.py model=gru4rec dataset=foursquare feature_mode=basic eval_mode=interaction \
  epochs=30 learning_rate=0.0005 gpu_id=5
```
Expected config values:
- model: "GRU4Rec"
- dataset: "foursquare"
- MAX_ITEM_LIST_LENGTH: 50
- epochs: 30  (CLI override)
- learning_rate: 0.0005  (CLI override)
- gpu_id: 5  (CLI override)

## File Structure

```
configs/
├── config.yaml                 # Base config with defaults
├── eval_mode/
│   ├── session.yaml           # Session-level split (default)
│   └── interaction.yaml       # Interaction-level LOO split
├── feature_mode/
│   ├── full.yaml              # Full features (~37GB)
│   └── basic.yaml             # Core columns (~2.7GB)
└── model/
    ├── caser.yaml             # Caser: MAX_ITEM_LIST_LENGTH=10
    ├── sasrec.yaml            # SASRec: num_heads=4
    ├── gru4rec.yaml
    ├── bert4rec.yaml
    ├── narm.yaml
    ├── stamp.yaml
    ├── srgnn.yaml
    └── ... (more models)
```

## How Model/Dataset Config Works

### When you run:
```bash
python recbole_train.py model=caser dataset=amazon_beauty
```

### Step-by-step:
1. Load config.yaml (base)
   - model: ???
   - dataset: ???
   - eval_mode: session
   - feature_mode: full

2. Hydra composes defaults in order:
   - eval_mode: session → load eval_mode/session.yaml
   - feature_mode: full → load feature_mode/full.yaml (but overridden by feature_mode=full CLI)
   - model: ??? → **waits for CLI override**

3. CLI overrides applied:
   - model=caser → load model/caser.yaml, set model field to "Caser"
   - dataset=amazon_beauty → override dataset field
   - (Any other CLI params)

4. Final merged config:
   ```yaml
   model: Caser                    # From model/caser.yaml
   dataset: amazon_beauty          # From CLI
   MAX_ITEM_LIST_LENGTH: 10        # From model/caser.yaml
   num_filters: 16                 # From model/caser.yaml
   eval_mode: session              # From default
   feature_mode: full              # From default (or basic if CLI says so)
   data_path: ../Datasets/processed/feature_added
   loss_type: CE
   epochs: 100
   ... (all other settings)
   ```

## Important Notes

### CLI Override Priority
- model=caser does TWO things:
  1. Loads model/caser.yaml (Hydra group selection)
  2. Sets the "model" field value to "Caser" (config override)

- Both happen automatically - the model field in config.yaml acts as placeholder

### Model-Specific Overrides
Some models override defaults:
- **Caser**: MAX_ITEM_LIST_LENGTH=10 (shorter sequences for CNN efficiency)
- **SASRec**: num_heads=4, inner_size=256
- Others: Inherit defaults from config.yaml

### Dataset Path
- Automatically set based on feature_mode:
  - feature_mode=full → data_path: ../Datasets/processed/feature_added
  - feature_mode=basic → data_path: ../Datasets/processed/basic

## Debugging Config

To see the final merged config:
```python
# In recbole_train.py, the debug output shows:
[DEBUG] Loaded config keys: [...]
[DEBUG] model=Caser
[DEBUG] dataset=amazon_beauty
[DEBUG] MAX_ITEM_LIST_LENGTH=10
[DEBUG] loss_type=CE
```

If values are wrong, check:
1. CLI override syntax: `model=caser` (not `model=Caser`)
2. Model file exists: `configs/model/caser.yaml`
3. Priority: CLI overrides > model/*.yaml > feature_mode > eval_mode > config.yaml
