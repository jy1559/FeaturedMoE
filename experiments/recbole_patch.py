"""
RecBole 1.2.1 patches for FMoE experiments.
"""

# ============ Patch: Mock xgboost before RecBole imports ============
import sys
import pickle
import json
import hashlib
from pathlib import Path

# Create a complete mock for xgboost module
class MockBooster:
    def __init__(self, *args, **kwargs):
        pass

class MockXGBoost:
    Booster = MockBooster
    __version__ = "1.0.0"
    
sys.modules['xgboost'] = MockXGBoost()

import numpy as np
import torch
from copy import copy

from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType
import recbole.utils.utils as recbole_utils
import recbole.quick_start.quick_start as quick_start_module


# ============ Patch: get_model to support custom models ============
_original_get_model = recbole_utils.get_model


def _get_custom_model(model_name):
    """Try to load custom model."""
    try:
        # Import after patching is complete
        from models import (
            BiLSTM,
            CLRec,
            BSARec,
            FAME,
            SIGMA,
            DIFSR,
            MSSR,
            PAtt,
            FENRec,
            FeaturedMoE,
            FeaturedMoE_HiR,
            FeaturedMoE_HGR,
            FeaturedMoE_HGRv3,
            FeaturedMoE_HGRv4,
            FeaturedMoE_Individual,
            FeaturedMoE_HiR2,
            FeaturedMoE_ProtoX,
            FeaturedMoE_V2,
            FeaturedMoE_V2_HiR,
            FeaturedMoE_V3,
            FeaturedMoE_V4_Distillation,
        )
        custom_models = {
            'BiLSTM': BiLSTM, 
            'CLRec': CLRec, 
            'BSARec': BSARec, 
            'FAME': FAME,
            'SIGMA': SIGMA,
            'DIFSR': DIFSR,
            'MSSR': MSSR,
            'PAtt': PAtt,
            'FENRec': FENRec,
            'FeaturedMoE': FeaturedMoE,
            'FeaturedMoE_HiR': FeaturedMoE_HiR,
            'FeaturedMoE_HGR': FeaturedMoE_HGR,
            'featured_moe_hgr': FeaturedMoE_HGR,
            'featuredmoe_hgr': FeaturedMoE_HGR,
            'FeaturedMoE_HGRv3': FeaturedMoE_HGRv3,
            'featured_moe_hgr_v3': FeaturedMoE_HGRv3,
            'featuredmoe_hgr_v3': FeaturedMoE_HGRv3,
            'featuredmoe_hgrv3': FeaturedMoE_HGRv3,
            'FeaturedMoE_HGRv4': FeaturedMoE_HGRv4,
            'featured_moe_hgr_v4': FeaturedMoE_HGRv4,
            'featuredmoe_hgr_v4': FeaturedMoE_HGRv4,
            'featuredmoe_hgrv4': FeaturedMoE_HGRv4,
            'FeaturedMoE_Individual': FeaturedMoE_Individual,
            'featured_moe_individual': FeaturedMoE_Individual,
            'featuredmoe_individual': FeaturedMoE_Individual,
            'FeaturedMoE_HiR2': FeaturedMoE_HiR2,
            'featured_moe_hir2': FeaturedMoE_HiR2,
            'featuredmoe_hir2': FeaturedMoE_HiR2,
            'FeaturedMoE_ProtoX': FeaturedMoE_ProtoX,
            'featured_moe_protox': FeaturedMoE_ProtoX,
            'featuredmoe_protox': FeaturedMoE_ProtoX,
            'FeaturedMoE_v2': FeaturedMoE_V2,
            'FeaturedMoE_V2': FeaturedMoE_V2,
            'featured_moe_v2': FeaturedMoE_V2,
            'featuredmoe_v2': FeaturedMoE_V2,
            'FeaturedMoE_v2_HiR': FeaturedMoE_V2_HiR,
            'FeaturedMoE_V2_HiR': FeaturedMoE_V2_HiR,
            'featured_moe_v2_hir': FeaturedMoE_V2_HiR,
            'featuredmoe_v2_hir': FeaturedMoE_V2_HiR,
            'FeaturedMoE_v3': FeaturedMoE_V3,
            'FeaturedMoE_V3': FeaturedMoE_V3,
            'featured_moe_v3': FeaturedMoE_V3,
            'featuredmoe_v3': FeaturedMoE_V3,
            'FeaturedMoE_v4_Distillation': FeaturedMoE_V4_Distillation,
            'FeaturedMoE_V4_Distillation': FeaturedMoE_V4_Distillation,
            'featured_moe_v4_distillation': FeaturedMoE_V4_Distillation,
            'featuredmoe_v4_distillation': FeaturedMoE_V4_Distillation,
        }
        return custom_models.get(model_name)
    except:
        return None


def _patched_get_model(model_name):
    """Get model, trying custom models first."""
    # Try custom models first
    custom = _get_custom_model(model_name)
    if custom is not None:
        return custom
    # Fall back to RecBole
    try:
        return _original_get_model(model_name)
    except Exception as e:
        # If RecBole fails, still try custom
        custom = _get_custom_model(model_name)
        if custom is not None:
            return custom
        raise e


# Apply patch to recbole_utils
recbole_utils.get_model = _patched_get_model

# Also patch in configurator (it imported get_model directly)
from recbole.config import configurator
configurator.get_model = _patched_get_model


# ============ Patch: get_flops to handle SESSION mode ============
_original_get_flops = recbole_utils.get_flops


def _patched_get_flops(model, dataset, device, logger, transform=None, verbose=False):
    """Patched get_flops that handles SESSION mode."""
    # Check if dataset has item_id_list (sequence format)
    try:
        if hasattr(dataset, 'inter_feat'):
            inter = dataset.inter_feat
            # Check if it's Interaction with item_id_list or DataFrame without
            if hasattr(inter, 'interaction'):
                if 'item_id_list' not in inter.interaction:
                    logger.info("Skipping FLOPS calculation (SESSION mode - no sequence data in original dataset)")
                    return 0
            elif hasattr(inter, 'columns'):
                if 'item_id_list' not in inter.columns:
                    logger.info("Skipping FLOPS calculation (SESSION mode - no sequence data in original dataset)")
                    return 0
        return _original_get_flops(model, dataset, device, logger, transform, verbose)
    except Exception as e:
        logger.warning(f"FLOPS calculation failed: {e}")
        return 0


# Apply patch to both modules
recbole_utils.get_flops = _patched_get_flops
quick_start_module.get_flops = _patched_get_flops


# ============ Helper: Convert interaction format to sequence format ============
def _convert_inter_to_sequence(dataset, inter_feat, for_training=True):
    """
    Convert interaction-format data to sequence-format for sequential models.

    This optimized implementation avoids per-row tensor construction and
    builds history lists with chunked tensor gathers.

    Args:
        dataset: SequentialDataset instance
        inter_feat: Interaction with interaction-format data
        for_training: If True, generate augmented training samples

    Returns:
        Interaction object in sequence format
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    time_field = dataset.time_field
    max_len = int(dataset.config["MAX_ITEM_LIST_LENGTH"])
    list_suffix = dataset.config["LIST_SUFFIX"] if "LIST_SUFFIX" in dataset.config else "_list"
    item_list_length_field = (
        dataset.config["ITEM_LIST_LENGTH_FIELD"] if "ITEM_LIST_LENGTH_FIELD" in dataset.config else "item_length"
    )
    chunk_size = int(dataset.config["sequence_convert_chunk_size"]) if "sequence_convert_chunk_size" in dataset.config else 16384
    chunk_size = max(1024, chunk_size)
    model_name = str(dataset.config["model"]).lower() if "model" in dataset.config else ""
    default_fp16 = model_name in {
        "featured_moe",
        "featuredmoe",
        "featured_moe_hir",
        "featuredmoe_hir",
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_hgr_v3",
        "featuredmoe_hgr_v3",
        "featuredmoe_hgrv3",
        "featured_moe_individual",
        "featuredmoe_individual",
        "featured_moe_hir2",
        "featuredmoe_hir2",
        "featured_moe_protox",
        "featuredmoe_protox",
        "featured_moe_v2",
        "featuredmoe_v2",
    }
    use_fp16 = bool(dataset.config["fmoe_feature_fp16"]) if "fmoe_feature_fp16" in dataset.config else default_fp16

    def _is_feature_field(name: str) -> bool:
        return name.startswith("mac_") or name.startswith("mid_") or name.startswith("mic_")

    # Convert DataFrame-like input if needed (fallback path).
    if not hasattr(inter_feat, "interaction"):
        import pandas as pd

        if not isinstance(inter_feat, pd.DataFrame):
            return None
        inter_feat = Interaction(
            {
                col: torch.tensor(inter_feat[col].values)
                for col in inter_feat.columns
            }
        )

    fields = list(inter_feat.columns)
    if len(fields) == 0:
        return None

    # In FeaturedMoE mode, only keep required history lists:
    # item_id_list + engineered feature lists.
    history_fields = [f for f in fields if f != uid_field]
    if model_name in {
        "featured_moe",
        "featuredmoe",
        "featured_moe_hir",
        "featuredmoe_hir",
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_hgr_v3",
        "featuredmoe_hgr_v3",
        "featuredmoe_hgrv3",
        "featured_moe_individual",
        "featuredmoe_individual",
        "featured_moe_hir2",
        "featuredmoe_hir2",
        "featured_moe_protox",
        "featuredmoe_protox",
        "featured_moe_v2",
        "featuredmoe_v2",
    }:
        history_fields = [
            f for f in history_fields
            if (f == iid_field or _is_feature_field(f))
        ]

    # Ensure [uid, time]-sorted order.
    uid_np = inter_feat[uid_field].cpu().numpy()
    time_np = inter_feat[time_field].cpu().numpy()
    order_np = np.lexsort((time_np, uid_np))
    order_t = torch.from_numpy(order_np).long()
    uid_sorted = uid_np[order_np]

    # Build sample indices per user/session.
    changes = np.nonzero(uid_sorted[1:] != uid_sorted[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(uid_sorted)]))

    target_parts = []
    seq_start_parts = []
    seq_len_parts = []
    for start, end in zip(starts, ends):
        n = int(end - start)
        if n < 2:
            continue
        if for_training:
            rel_target = np.arange(1, n, dtype=np.int64)
            target = start + rel_target
            seq_len = np.minimum(rel_target, max_len)
            seq_start = target - seq_len
        else:
            target = np.array([end - 1], dtype=np.int64)
            seq_len_val = min(n - 1, max_len)
            seq_len = np.array([seq_len_val], dtype=np.int64)
            seq_start = target - seq_len

        target_parts.append(target)
        seq_start_parts.append(seq_start)
        seq_len_parts.append(seq_len)

    if not target_parts:
        return None

    target_idx_np = np.concatenate(target_parts)
    seq_start_np = np.concatenate(seq_start_parts)
    seq_len_np = np.concatenate(seq_len_parts)

    target_idx = torch.from_numpy(target_idx_np).long()
    seq_start = torch.from_numpy(seq_start_np).long()
    seq_len = torch.from_numpy(seq_len_np).long()
    new_length = int(target_idx.shape[0])

    new_dict = {item_list_length_field: seq_len}
    offsets = torch.arange(max_len, dtype=torch.long)

    for field in fields:
        values = inter_feat[field]
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        values = values[order_t]

        # Target-side scalar for each generated sample.
        new_dict[field] = values[target_idx]

        # Build history sequence field only when required.
        if field not in history_fields:
            continue

        list_field = field + list_suffix
        hist_values = values
        if use_fp16 and _is_feature_field(field) and hist_values.is_floating_point():
            hist_values = hist_values.to(torch.float16)

        out_shape = (new_length, max_len) + tuple(hist_values.shape[1:])
        out = torch.zeros(out_shape, dtype=hist_values.dtype)

        for s in range(0, new_length, chunk_size):
            e = min(new_length, s + chunk_size)
            chunk_start = seq_start[s:e]  # [B]
            chunk_len = seq_len[s:e]      # [B]
            idx = chunk_start.unsqueeze(1) + offsets.unsqueeze(0)  # [B, T]
            valid = offsets.unsqueeze(0) < chunk_len.unsqueeze(1)  # [B, T]
            idx = idx.clamp(max=hist_values.shape[0] - 1)
            gathered = hist_values[idx]
            if gathered.dim() == 2:
                gathered = gathered.masked_fill(~valid, 0)
            else:
                mask = ~valid
                for _ in range(gathered.dim() - 2):
                    mask = mask.unsqueeze(-1)
                gathered = gathered.masked_fill(mask, 0)
            out[s:e] = gathered

        new_dict[list_field] = out

    return Interaction(new_dict)


# ============ Patch: Handle benchmark mode with interaction-format files ============
_original_benchmark_presets = SequentialDataset._benchmark_presets


def _patched_benchmark_presets(self):
    """
    Patched version that handles interaction-format pre-split files.
    
    If benchmark files are in interaction format (no item_id_list), 
    we skip the original method since we'll convert later in build().
    """
    list_suffix = self.config['LIST_SUFFIX'] if 'LIST_SUFFIX' in self.config else '_list'
    item_list_field = self.iid_field + list_suffix
    
    # Initialize field names
    if not hasattr(self, 'item_id_list_field'):
        self.item_id_list_field = item_list_field
    
    # Check if data is already in sequence format
    if item_list_field in self.inter_feat.columns:
        # Already sequence format, use original method
        return _original_benchmark_presets(self)
    else:
        # Interaction format - mark for later conversion
        self._needs_sequence_conversion = True
        
        # Set up field properties for later
        self.item_list_length_field = self.config['ITEM_LIST_LENGTH_FIELD'] if 'ITEM_LIST_LENGTH_FIELD' in self.config else 'item_length'
        
        # Register list fields
        for field in self.inter_feat.columns:
            if field != self.uid_field:
                setattr(self, f'{field}_list_field', field + list_suffix)


# ============ Patch: Build method to handle SESSION mode ============
_original_build = SequentialDataset.build
_original_data_augmentation = SequentialDataset.data_augmentation


def _patched_data_augmentation(self):
    """Optimized sequential augmentation for interaction-format datasets.

    This replaces per-sample Python loops with chunked tensor gathers.
    """
    if "use_fast_data_augmentation" in self.config and not bool(self.config["use_fast_data_augmentation"]):
        return _original_data_augmentation(self)

    self.logger.debug("data_augmentation (optimized)")

    self._aug_presets()
    self._check_field("uid_field", "time_field")

    max_item_list_len = int(self.config["MAX_ITEM_LIST_LENGTH"])
    chunk_size = int(self.config["sequence_convert_chunk_size"]) if "sequence_convert_chunk_size" in self.config else 16384
    chunk_size = max(1024, chunk_size)

    model_name = str(self.config["model"]).lower() if "model" in self.config else ""
    default_fp16 = model_name in {
        "featured_moe",
        "featuredmoe",
        "featured_moe_hir",
        "featuredmoe_hir",
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_hgr_v3",
        "featuredmoe_hgr_v3",
        "featuredmoe_hgrv3",
        "featured_moe_individual",
        "featuredmoe_individual",
        "featured_moe_hir2",
        "featuredmoe_hir2",
        "featured_moe_protox",
        "featuredmoe_protox",
        "featured_moe_v2",
        "featuredmoe_v2",
    }
    use_fp16 = bool(self.config["fmoe_feature_fp16"]) if "fmoe_feature_fp16" in self.config else default_fp16

    def _is_feature_field(name: str) -> bool:
        return name.startswith("mac_") or name.startswith("mid_") or name.startswith("mic_")

    # Keep original order semantics: sorted by (uid, time).
    self.sort(by=[self.uid_field, self.time_field], ascending=True)

    uid_np = self.inter_feat[self.uid_field].numpy()
    if uid_np.size <= 1:
        self.inter_feat = self.inter_feat[:0]
        return

    # Build (target, seq_start, seq_len) for all valid next-item samples.
    changes = np.nonzero(uid_np[1:] != uid_np[:-1])[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(uid_np)]))

    target_parts = []
    seq_start_parts = []
    seq_len_parts = []
    for start, end in zip(starts, ends):
        n = int(end - start)
        if n < 2:
            continue
        rel_target = np.arange(1, n, dtype=np.int64)
        target = start + rel_target
        seq_len = np.minimum(rel_target, max_item_list_len)
        seq_start = target - seq_len
        target_parts.append(target)
        seq_start_parts.append(seq_start)
        seq_len_parts.append(seq_len)

    if not target_parts:
        self.inter_feat = self.inter_feat[:0]
        return

    target_idx_np = np.concatenate(target_parts)
    seq_start_np = np.concatenate(seq_start_parts)
    seq_len_np = np.concatenate(seq_len_parts)

    target_idx = torch.from_numpy(target_idx_np).long()
    seq_start = torch.from_numpy(seq_start_np).long()
    seq_len = torch.from_numpy(seq_len_np).long()
    new_length = int(target_idx.shape[0])

    new_data = self.inter_feat[target_idx]
    new_dict = {
        self.item_list_length_field: seq_len,
    }

    offsets = torch.arange(max_item_list_len, dtype=torch.long)

    for field in self.inter_feat:
        if field == self.uid_field:
            continue

        list_field = getattr(self, f"{field}_list_field")
        list_len = self.field2seqlen[list_field]
        shape = (
            (new_length, list_len)
            if isinstance(list_len, int)
            else (new_length,) + tuple(list_len)
        )

        value = self.inter_feat[field]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        hist_value = value
        if use_fp16 and _is_feature_field(field) and hist_value.is_floating_point():
            hist_value = hist_value.to(torch.float16)

        if (
            self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
            and field in self.config["numerical_features"]
        ):
            shape += (2,)

        out = torch.zeros(shape, dtype=hist_value.dtype)

        for s in range(0, new_length, chunk_size):
            e = min(new_length, s + chunk_size)
            chunk_start = seq_start[s:e]  # [B]
            chunk_len = seq_len[s:e]      # [B]
            idx = chunk_start.unsqueeze(1) + offsets.unsqueeze(0)  # [B, T]
            valid = offsets.unsqueeze(0) < chunk_len.unsqueeze(1)  # [B, T]
            idx = idx.clamp(max=hist_value.shape[0] - 1)
            gathered = hist_value[idx]
            if gathered.dim() == 2:
                gathered = gathered.masked_fill(~valid, 0)
            else:
                mask = ~valid
                for _ in range(gathered.dim() - 2):
                    mask = mask.unsqueeze(-1)
                gathered = gathered.masked_fill(mask, 0)

            # out shape can be [B,T] or [B,T,*]
            out[s:e] = gathered

        new_dict[list_field] = out

    new_data.update(Interaction(new_dict))
    self.inter_feat = new_data


def _patched_build(self):
    """
    Patched build that handles SESSION mode with interaction-format pre-split files.
    """
    # Check if we need to convert interaction -> sequence
    needs_conversion = getattr(self, '_needs_sequence_conversion', False)
    
    if self.benchmark_filename_list is not None and needs_conversion:
        # SESSION mode with pre-split interaction files
        self.logger.info("SESSION mode: Converting pre-split interaction files to sequence format")
        
        # Get config values
        list_suffix = self.config['LIST_SUFFIX'] if 'LIST_SUFFIX' in self.config else '_list'
        item_list_length_field = self.config['ITEM_LIST_LENGTH_FIELD'] if 'ITEM_LIST_LENGTH_FIELD' in self.config else 'item_length'
        max_len = self.config['MAX_ITEM_LIST_LENGTH']
        split_cache_enabled = True if "enable_session_split_cache" not in self.config else bool(self.config["enable_session_split_cache"])
        split_cache_file = None
        if split_cache_enabled and "checkpoint_dir" in self.config:
            try:
                ckpt_dir = Path(self.config["checkpoint_dir"])
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ds_name = getattr(self, "dataset_name", "dataset")
                split_cache_file = ckpt_dir / f"{ds_name}-session-split-len{max_len}.pth"
                if split_cache_file.exists():
                    with open(split_cache_file, "rb") as f:
                        cached_splits = pickle.load(f)
                    if isinstance(cached_splits, list) and len(cached_splits) == 3:
                        self.logger.info(f"SESSION mode: Loaded split cache from {split_cache_file}")
                        return cached_splits
            except Exception as e:
                self.logger.warning(f"SESSION mode: Failed to load split cache ({e})")
        
        # First, convert main dataset to Interaction format
        self._change_feat_format()
        
        # Get file boundaries
        cumsum = list(np.cumsum(self.file_size_list))
        boundaries = [0] + cumsum
        
        datasets = []
        split_names = ['train', 'valid', 'test']
        
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            # Get this split's data (now Interaction)
            split_inter = self.inter_feat[start:end]
            
            # Convert to sequence format
            is_train = (i == 0)  # First split is train
            converted = _convert_inter_to_sequence(self, split_inter, for_training=is_train)
            
            if converted is not None:
                # Create dataset copy with converted Interaction
                new_dataset = self.copy(converted)
                
                # Ensure inter_feat is Interaction
                if not isinstance(new_dataset.inter_feat, Interaction):
                    new_dataset.inter_feat = converted
                
                # Set up field properties
                new_dataset.item_id_list_field = self.iid_field + list_suffix
                new_dataset.item_list_length_field = item_list_length_field
                
                # Register list fields and set field properties
                for field in converted.columns:
                    if field.endswith(list_suffix):
                        base_field = field[:-len(list_suffix)]
                        setattr(new_dataset, f'{base_field}_list_field', field)
                        
                        # Determine feature type based on base field
                        if base_field == self.iid_field:
                            ftype = FeatureType.TOKEN_SEQ
                        else:
                            ftype = FeatureType.FLOAT_SEQ
                            
                        # Register field
                        if field not in new_dataset.field2type:
                            new_dataset.field2type[field] = ftype
                            new_dataset.field2seqlen[field] = max_len
                
                # Register item_length field
                if item_list_length_field not in new_dataset.field2type:
                    new_dataset.field2type[item_list_length_field] = FeatureType.TOKEN
                    new_dataset.field2seqlen[item_list_length_field] = 1
                
                datasets.append(new_dataset)
                try:
                    self.logger.info(
                        f"  {split_names[i]}: converted sequences={len(converted)} | rows={len(split_inter)} | max_len={max_len}"
                    )
                except Exception:
                    pass
            else:
                self.logger.warning(f"  {split_names[i]}: No valid sequences!")
                # Create empty dataset
                empty = self.copy(self.inter_feat[:0])
                datasets.append(empty)
        
        if split_cache_file is not None:
            try:
                with open(split_cache_file, "wb") as f:
                    pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"SESSION mode: Saved split cache to {split_cache_file}")
            except Exception as e:
                self.logger.warning(f"SESSION mode: Failed to save split cache ({e})")

        return datasets
    else:
        # Non-benchmark path: optionally cache built split datasets.
        split_cache_enabled = True if "enable_session_split_cache" not in self.config else bool(self.config["enable_session_split_cache"])
        if split_cache_enabled and "checkpoint_dir" in self.config:
            try:
                ckpt_dir = Path(self.config["checkpoint_dir"])
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ds_name = getattr(self, "dataset_name", "dataset")
                max_len = self.config['MAX_ITEM_LIST_LENGTH'] if 'MAX_ITEM_LIST_LENGTH' in self.config else "na"
                eval_args = self.config["eval_args"] if "eval_args" in self.config else {}
                eval_key = json.dumps(eval_args, sort_keys=True, ensure_ascii=True)
                eval_hash = hashlib.md5(eval_key.encode("utf-8")).hexdigest()[:10]
                split_cache_file = ckpt_dir / f"{ds_name}-split-cache-len{max_len}-{eval_hash}.pth"

                if split_cache_file.exists():
                    with open(split_cache_file, "rb") as f:
                        cached_splits = pickle.load(f)
                    if isinstance(cached_splits, list) and len(cached_splits) == 3:
                        self.logger.info(f"Loaded split cache from {split_cache_file}")
                        return cached_splits

                built = _original_build(self)
                with open(split_cache_file, "wb") as f:
                    pickle.dump(built, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f"Saved split cache to {split_cache_file}")
                return built
            except Exception as e:
                self.logger.warning(f"Split cache unavailable, fallback to original build ({e})")

        # Fallback: original RecBole build.
        return _original_build(self)


# Apply patches
SequentialDataset._benchmark_presets = _patched_benchmark_presets
SequentialDataset.build = _patched_build
SequentialDataset.data_augmentation = _patched_data_augmentation


# ============ Patch: Trainer to add epoch-wise wandb logging ============
from recbole.trainer.trainer import Trainer

_original_trainer_fit = Trainer.fit
_original_train_epoch = Trainer._train_epoch
_original_valid_epoch = Trainer._valid_epoch

def _patched_trainer_fit(self, *args, **kwargs):
    """Wrapper around original fit to preserve Trainer API (no behavior change here).
    Epoch-wise logging is handled by patched _train_epoch/_valid_epoch below.
    """
    return _original_trainer_fit(self, *args, **kwargs)

Trainer.fit = _patched_trainer_fit

# Patch: epoch start time tracking and per-epoch best logging
def _patched_train_epoch(self, *args, **kwargs):
    import time
    # mark epoch start to measure time
    if getattr(self, '_disable_patch_logging', False):
        return _original_train_epoch(self, *args, **kwargs)
    self._epoch_start_time = time.time()
    return _original_train_epoch(self, *args, **kwargs)


def _patched_valid_epoch(self, valid_data, show_progress=False):
    # run original validation
    if getattr(self, '_disable_patch_logging', False):
        return _original_valid_epoch(self, valid_data, show_progress)
    result = _original_valid_epoch(self, valid_data, show_progress)

    # compute epoch time if available
    import time
    epoch_time_sec = None
    try:
        if hasattr(self, '_epoch_start_time'):
            epoch_time_sec = time.time() - self._epoch_start_time
    except Exception:
        pass

    # track and log best-so-far based on valid_metric
    try:
        import wandb
        if wandb.run is not None and isinstance(result, dict):
            # maintain a local valid_step counter aligned with epochs
            if not hasattr(self, '_patched_valid_step'):
                self._patched_valid_step = 0
            # Determine improvement based on valid_metric
            valid_metric = str(self.config.get('valid_metric', 'NDCG@10')).lower()
            current = result.get(valid_metric, None)

            # Initialize store
            if not hasattr(self, '_best_valid_snapshot'):
                self._best_valid_snapshot = None
                self._best_valid_score = float('-inf')

            # Update best snapshot when improved
            if current is not None and current > self._best_valid_score:
                self._best_valid_score = current
                # snapshot the full metric dict
                self._best_valid_snapshot = {
                    'hit@5': result.get('hit@5', 0),
                    'hit@10': result.get('hit@10', 0),
                    'mrr@20': result.get('mrr@20', 0),
                    'ndcg@10': result.get('ndcg@10', 0),
                }

            # Log both current and best into valid section (lowercase names to match RecBole)
            payload = {}
            # Best (so far)
            if self._best_valid_snapshot:
                payload.update({
                    'valid/best_hit@5': self._best_valid_snapshot.get('hit@5', 0),
                    'valid/best_hit@10': self._best_valid_snapshot.get('hit@10', 0),
                    'valid/best_mrr@20': self._best_valid_snapshot.get('mrr@20', 0),
                    'valid/best_ndcg@10': self._best_valid_snapshot.get('ndcg@10', 0),
                })

            # Epoch timing
            if epoch_time_sec is not None:
                payload['etc/epoch_time_sec'] = epoch_time_sec

            # Log with valid_step to make panels appear under the valid group
            payload['valid_step'] = self._patched_valid_step

            if payload:
                # debug info to confirm patch runs
                try:
                    self.logger.info("Patched valid logging: emitting valid/best_* and etc/epoch_time_sec")
                except Exception:
                    pass
                wandb.log(payload, step=self._patched_valid_step)
                self._patched_valid_step += 1
    except Exception:
        # Avoid breaking training due to logging errors
        pass

    return result


Trainer._train_epoch = _patched_train_epoch
Trainer._valid_epoch = _patched_valid_epoch


# ============ Patch: Sampled Evaluation with Precomputed Negatives ============
# When n_items > threshold, use precomputed negatives instead of full ranking

_original_full_sort_batch_eval = Trainer._full_sort_batch_eval

def _patched_full_sort_batch_eval(self, batched_data):
    """
    Patched _full_sort_batch_eval that supports sampled evaluation.
    
    If self._sampled_eval_mask is set, applies it to scores after full_sort_predict.
    This masks out items not in the precomputed negative set, effectively doing
    sampled evaluation while keeping the same batch structure.
    """
    result = _original_full_sort_batch_eval(self, batched_data)
    
    # Check if sampled evaluation mask is set
    sampled_mask = getattr(self, '_sampled_eval_mask', None)
    if sampled_mask is not None:
        interaction, scores, positive_u, positive_i = result
        
        # Expand mask to batch size
        # scores shape: (batch_size, n_items)
        # sampled_mask shape: (n_items,) boolean
        
        # Set scores for non-candidate items to -inf
        # But keep positive items (they should always be evaluated)
        mask = sampled_mask.clone()
        mask[positive_i] = True  # Always include positive items
        
        # Apply mask: non-candidate items get -inf
        scores[:, ~mask] = float('-inf')
        
        return (interaction, scores, positive_u, positive_i)
    
    return result

Trainer._full_sort_batch_eval = _patched_full_sort_batch_eval


def setup_sampled_eval(trainer, neg_items: np.ndarray, n_neg: int, n_items: int, device):
    """
    Setup sampled evaluation for a trainer.
    
    Args:
        trainer: RecBole Trainer instance
        neg_items: Negative item IDs (should be RecBole internal IDs, sorted by priority)
        n_neg: Number of negatives to use
        n_items: Total number of items
        device: torch device
    """
    # Create mask: True for candidate items (first n_neg from neg_items)
    mask = torch.zeros(n_items, dtype=torch.bool, device=device)
    candidates = torch.LongTensor(neg_items[:n_neg]).to(device)
    mask[candidates] = True
    
    trainer._sampled_eval_mask = mask
    # Print message is handled by caller in recbole_train.py


def clear_sampled_eval(trainer):
    """Clear sampled evaluation mode."""
    if hasattr(trainer, '_sampled_eval_mask'):
        delattr(trainer, '_sampled_eval_mask')


# ============ Patch: Force leave=False on tqdm bars (clean output) ============
from tqdm import tqdm as tqdm_class
import tqdm as tqdm_module

# Store original __init__
_original_tqdm_init = tqdm_class.__init__

def _patched_tqdm_init(self, *args, **kwargs):
    """Force leave=False for all tqdm progress bars during training/eval."""
    # Only force leave=False if not explicitly set to True (allow overrides)
    if kwargs.get('leave', None) is None or kwargs.get('leave') == True:
        # Check if this looks like a batch/epoch progress bar (desc contains train/valid/eval)
        desc = kwargs.get('desc', '')
        if any(keyword in str(desc).lower() for keyword in ['train', 'valid', 'eval', 'evaluate']):
            kwargs['leave'] = False
    _original_tqdm_init(self, *args, **kwargs)

# Apply patch
tqdm_class.__init__ = _patched_tqdm_init
