"""
Precompute negative samples for efficient evaluation.

Instead of sampling negatives on-the-fly during each epoch,
we precompute and store a fixed set of negatives per dataset.

Negatives are sorted by popularity (most popular first), so when using
fewer negatives than precomputed, just take the first N.

Usage:
    # Precompute once per dataset
    python tools/precompute_neg_samples.py --dataset foursquare --n_neg 3000
    
    # Then in training, these are loaded and used for consistent evaluation
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import json


def precompute_negatives(
    dataset_path: Path,
    n_neg: int = 3000,
    seed: int = 42,
) -> dict:
    """
    Precompute negative samples for evaluation.
    
    Samples are sorted by popularity (most popular first).
    When using fewer than n_neg, just slice from the front.
    
    Args:
        dataset_path: Path to dataset directory (e.g., Datasets/processed/basic/foursquare)
        n_neg: Number of negative samples to precompute
        seed: Random seed for reproducibility (used for tie-breaking)
    
    Returns:
        Dictionary with:
        - 'neg_items': numpy array of item IDs sorted by popularity
        - 'item_popularity': {item_id: count}
        - 'metadata': {n_neg, seed, n_items, ...}
    """
    np.random.seed(seed)
    
    # Load interaction file
    inter_files = list(dataset_path.glob("*.inter"))
    if not inter_files:
        raise FileNotFoundError(f"No .inter file found in {dataset_path}")
    inter_file = inter_files[0]
    
    # Parse interactions to get item popularity
    item_counts = defaultdict(int)
    all_items = set()
    
    with open(inter_file, 'r') as f:
        header = f.readline().strip().split('\t')
        item_idx = header.index('item_id') if 'item_id' in header else 1
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > item_idx:
                try:
                    item_id = int(parts[item_idx])
                    item_counts[item_id] += 1
                    all_items.add(item_id)
                except ValueError:
                    continue
    
    n_items = len(all_items)
    max_item_id = max(all_items) + 1 if all_items else 0
    print(f"[INFO] Found {n_items} unique items, max_id={max_item_id-1}")
    
    # Sort items by popularity (descending)
    # Add small random noise for tie-breaking
    items_with_pop = [(item_id, item_counts[item_id] + np.random.random() * 0.001) 
                      for item_id in all_items]
    items_with_pop.sort(key=lambda x: x[1], reverse=True)
    
    # Take top n_neg items (or all items if n_neg > n_items)
    actual_n_neg = min(n_neg, n_items)
    neg_items = np.array([item_id for item_id, _ in items_with_pop[:actual_n_neg]], dtype=np.int64)
    
    print(f"[INFO] Precomputed {len(neg_items)} negative items (sorted by popularity)")
    print(f"       Top 5 popular items: {neg_items[:5].tolist()}")
    
    result = {
        'neg_items': neg_items,
        'n_neg': actual_n_neg,
        'n_items_total': n_items,
        'max_item_id': max_item_id,
        'seed': seed,
        'item_popularity': {int(k): int(v) for k, v in item_counts.items()},
    }
    
    return result


def save_negatives(neg_data: dict, output_path: Path):
    """Save precomputed negatives to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy for efficiency
    np.savez_compressed(
        output_path,
        neg_items=neg_data['neg_items'],
        n_neg=neg_data['n_neg'],
        n_items_total=neg_data['n_items_total'],
        max_item_id=neg_data['max_item_id'],
        seed=neg_data['seed'],
    )
    
    # Also save metadata as JSON for easy inspection
    meta_path = output_path.with_suffix('.json')
    meta = {k: v for k, v in neg_data.items() if k not in ['neg_items', 'item_popularity']}
    meta['neg_items_preview'] = neg_data['neg_items'][:20].tolist()
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"[INFO] Saved to {output_path} and {meta_path}")


def load_negatives(neg_file: Path) -> np.ndarray:
    """
    Load precomputed negatives from file.
    
    Returns:
        numpy array of negative item IDs (sorted by popularity)
    """
    data = np.load(neg_file, allow_pickle=True)
    return data['neg_items']


def get_negatives(neg_items: np.ndarray, n_sample: int, exclude_items: set = None) -> np.ndarray:
    """
    Get n_sample negative items from precomputed pool.
    
    Takes from the front (most popular first).
    Optionally excludes specific items.
    
    Args:
        neg_items: Precomputed negative items (sorted by popularity)
        n_sample: Number of negatives to return
        exclude_items: Set of item IDs to exclude (e.g., user's positive items)
    
    Returns:
        Array of negative item IDs
    """
    if exclude_items is None or len(exclude_items) == 0:
        return neg_items[:n_sample].copy()
    
    # Filter out excluded items, take first n_sample
    filtered = [item for item in neg_items if item not in exclude_items]
    return np.array(filtered[:n_sample], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Precompute negative samples for evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., foursquare)")
    parser.add_argument("--n_neg", type=int, default=3000, help="Number of negatives to precompute")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_mode", type=str, default="basic", help="Feature mode (basic/full)")
    args = parser.parse_args()
    
    # Find dataset path
    base_path = Path(__file__).parent.parent / "Datasets" / "processed" / args.feature_mode / args.dataset
    
    if not base_path.exists():
        print(f"[ERROR] Dataset path not found: {base_path}")
        return
    
    # Precompute
    neg_data = precompute_negatives(
        dataset_path=base_path,
        n_neg=args.n_neg,
        seed=args.seed,
    )
    
    # Save
    output_path = base_path / "neg_samples.npz"
    save_negatives(neg_data, output_path)
    
    print(f"\n[SUCCESS] Precomputed {neg_data['n_neg']} negatives for {args.dataset}")


if __name__ == "__main__":
    main()
