#!/usr/bin/env python3
"""Test Hydra config loading with multiple models."""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from hydra_utils import load_hydra_config

config_dir = ROOT_DIR / "configs"

def test_config(overrides, description):
    """Test a config combination."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Overrides: {overrides}")
    print(f"{'='*60}")
    try:
        cfg = load_hydra_config(
            config_dir=config_dir,
            config_name="config",
            overrides=overrides,
        )
        print(f"✓ model={cfg.get('model')}")
        print(f"✓ dataset={cfg.get('dataset')}")
        print(f"✓ MAX_ITEM_LIST_LENGTH={cfg.get('MAX_ITEM_LIST_LENGTH')}")
        print(f"✓ loss_type={cfg.get('loss_type')}")
        
        # Model-specific checks
        if cfg.get('model') == 'Caser':
            print(f"✓ num_filters={cfg.get('num_filters')} (Caser-specific)")
            assert cfg.get('MAX_ITEM_LIST_LENGTH') == 10, "Caser should have MAX_ITEM_LIST_LENGTH=10"
        elif cfg.get('model') == 'SASRec':
            print(f"✓ num_heads={cfg.get('num_heads')} (SASRec-specific)")
            print(f"✓ inner_size={cfg.get('inner_size')} (SASRec-specific)")
            assert cfg.get('MAX_ITEM_LIST_LENGTH') == 50, "SASRec should have MAX_ITEM_LIST_LENGTH=50"
        elif cfg.get('model') == 'FeaturedMoE':
            print(f"✓ d_feat_emb={cfg.get('d_feat_emb')} (FeaturedMoE-specific)")
            print(f"✓ expert_scale={cfg.get('expert_scale')} (FeaturedMoE-specific)")
            assert cfg.get('d_feat_emb') is not None, "FeaturedMoE should define d_feat_emb"
            assert cfg.get('expert_scale') in (1, 2, 3), "expert_scale should be 1/2/3"
        
        print(f"✓ feature_mode from data_path: {'basic' if 'basic' in cfg.get('data_path', '') else 'full'}")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test cases
tests = [
    (["model=caser", "dataset=amazon_beauty", "feature_mode=basic"], 
     "Caser + amazon_beauty + basic features"),
    
    (["model=caser", "dataset=amazon_beauty"],
     "Caser + amazon_beauty + full features (default)"),
    
    (["model=sasrec", "dataset=lastfm", "feature_mode=basic"],
     "SASRec + lastfm + basic features"),
    
    (["model=sasrec", "dataset=lastfm", "eval_mode=interaction"],
     "SASRec + lastfm + interaction mode"),
    
    (["model=caser", "dataset=foursquare", "epochs=50", "learning_rate=0.0005"],
     "Caser + foursquare + CLI overrides (epochs=50, lr=0.0005)"),

    (["model=featured_moe", "dataset=movielens1m", "feature_mode=full"],
     "FeaturedMoE + movielens1m + full features"),

    (["model=featured_moe", "dataset=movielens1m", "feature_mode=full", "expert_scale=2"],
     "FeaturedMoE + expert_scale override"),
]

print("\n" + "="*60)
print("CONFIG LOADING TESTS")
print("="*60)

results = []
for overrides, desc in tests:
    results.append(test_config(overrides, desc))

print("\n" + "="*60)
print(f"SUMMARY: {sum(results)}/{len(results)} tests passed")
print("="*60)
