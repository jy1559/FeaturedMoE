"""
RecBole 1.2.1 patches for FMoE experiments.
"""

# ============ Patch: Mock xgboost before RecBole imports ============
import sys
import pickle
import json
import hashlib
import importlib
import importlib.util
import time as _time
from pathlib import Path

_PATCH_DIR = Path(__file__).resolve().parent
# Ensure custom models under experiments/models are importable regardless of cwd.
if str(_PATCH_DIR) not in sys.path:
    sys.path.insert(0, str(_PATCH_DIR))


def _ensure_local_models_package() -> None:
    """Bind top-level 'models' to experiments/models to avoid name collisions."""
    models_dir = _PATCH_DIR / "models"
    init_file = models_dir / "__init__.py"
    if not init_file.exists():
        return

    loaded = sys.modules.get("models")
    loaded_file = str(getattr(loaded, "__file__", "") or "")
    if loaded is not None and loaded_file.startswith(str(models_dir)):
        return

    spec = importlib.util.spec_from_file_location(
        "models",
        str(init_file),
        submodule_search_locations=[str(models_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["models"] = module
    spec.loader.exec_module(module)


_ensure_local_models_package()

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
from recbole.utils.enum_type import FeatureType, ModelType
import recbole.utils.utils as recbole_utils
import recbole.quick_start.quick_start as quick_start_module
import recbole.data as recbole_data_module
import recbole.data.utils as recbole_data_utils


def _config_get(config_obj, key, default=None):
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    try:
        return config_obj[key]
    except Exception:
        return getattr(config_obj, key, default)


def _history_input_mode(config_obj) -> str:
    return str(_config_get(config_obj, "history_input_mode", "session_only") or "session_only").lower().strip()


def _history_full_mode_enabled(config_obj) -> bool:
    return _history_input_mode(config_obj) == "full_history_session_targets"


def _history_group_field(config_obj) -> str:
    return str(_config_get(config_obj, "history_group_field", "user_id") or "user_id").strip()


def _target_group_field(config_obj, default: str = "session_id") -> str:
    return str(_config_get(config_obj, "target_group_field", default) or default).strip()


def _history_eval_policy(config_obj) -> str:
    return str(_config_get(config_obj, "history_eval_policy", "strict_train_prefix") or "strict_train_prefix").lower().strip()


def _current_session_item_length_field(config_obj) -> str:
    return str(
        _config_get(config_obj, "current_session_item_length_field", "current_session_item_length")
        or "current_session_item_length"
    ).strip()


def _ensure_history_input_load_col(config_obj) -> None:
    if not _history_full_mode_enabled(config_obj):
        return

    time_field = str(_config_get(config_obj, "TIME_FIELD", "timestamp") or "timestamp").strip()
    required_fields = [
        _history_group_field(config_obj),
        _target_group_field(config_obj, default=str(_config_get(config_obj, "SESSION_ID_FIELD", "session_id") or "session_id")),
        time_field,
    ]

    load_col = _config_get(config_obj, "load_col", None)
    if not isinstance(load_col, dict):
        load_col = {}
    inter_cols = list(load_col.get("inter", []) or [])

    changed = False
    for field in required_fields:
        if field and field not in inter_cols:
            inter_cols.append(field)
            changed = True

    if not changed:
        return

    new_load_col = dict(load_col)
    new_load_col["inter"] = inter_cols
    try:
        config_obj["load_col"] = new_load_col
    except Exception:
        pass
    final_cfg = getattr(config_obj, "final_config_dict", None)
    if isinstance(final_cfg, dict):
        final_cfg["load_col"] = new_load_col


def _dataset_file_signature(config_obj, dataset_name: str) -> dict:
    """Build a lightweight signature for source .inter/.item files.

    This prevents stale split caches when data files are regenerated in-place.
    """
    data_path = str(_config_get(config_obj, "data_path", "") or "")
    ds_name = str(dataset_name or "").strip()
    sig = {"data_path": data_path, "dataset": ds_name}
    if not data_path or not ds_name:
        return sig

    root = Path(data_path) / ds_name
    for suffix in ("inter", "item"):
        fp = root / f"{ds_name}.{suffix}"
        key = f"{suffix}_file"
        if fp.exists():
            st = fp.stat()
            sig[key] = {
                "path": str(fp),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        else:
            sig[key] = None
    return sig


# ============ Patch: get_model to support custom models ============
_original_get_model = recbole_utils.get_model


_CUSTOM_MODEL_SPECS = {
    'BiLSTM': ('bilstm', 'BiLSTM'),
    'CLRec': ('clrec', 'CLRec'),
    'DuoRec': ('duorec', 'DuoRec'),
    'duorec': ('duorec', 'DuoRec'),
    'FEARec': ('fearec', 'FEARec'),
    'fearec': ('fearec', 'FEARec'),
    'FDSA': ('fdsa', 'FDSA'),
    'fdsa': ('fdsa', 'FDSA'),
    'TiSASRec': ('tisasrec', 'TiSASRec'),
    'tisasrec': ('tisasrec', 'TiSASRec'),
    'BSARec': ('bsarec', 'BSARec'),
    'FAME': ('fame', 'FAME'),
    'SIGMA': ('sigma', 'SIGMA'),
    'DIFSR': ('difsr', 'DIFSR'),
    'MSSR': ('mssr', 'MSSR'),
    'PAtt': ('patt', 'PAtt'),
    'FENRec': ('fenrec', 'FENRec'),
    'FeaturedMoE': ('FeaturedMoE', 'FeaturedMoE'),
    'FeaturedMoE_HGR': ('FeaturedMoE_HGR', 'FeaturedMoE_HGR'),
    'featured_moe_hgr': ('FeaturedMoE_HGR', 'FeaturedMoE_HGR'),
    'featuredmoe_hgr': ('FeaturedMoE_HGR', 'FeaturedMoE_HGR'),
    'FeaturedMoE_HGRv4': ('FeaturedMoE_HGRv4', 'FeaturedMoE_HGRv4'),
    'featured_moe_hgr_v4': ('FeaturedMoE_HGRv4', 'FeaturedMoE_HGRv4'),
    'featuredmoe_hgr_v4': ('FeaturedMoE_HGRv4', 'FeaturedMoE_HGRv4'),
    'featuredmoe_hgrv4': ('FeaturedMoE_HGRv4', 'FeaturedMoE_HGRv4'),
    'FeaturedMoE_v2': ('FeaturedMoE_v2', 'FeaturedMoE_V2'),
    'FeaturedMoE_V2': ('FeaturedMoE_v2', 'FeaturedMoE_V2'),
    'featured_moe_v2': ('FeaturedMoE_v2', 'FeaturedMoE_V2'),
    'featuredmoe_v2': ('FeaturedMoE_v2', 'FeaturedMoE_V2'),
    'FeaturedMoE_v3': ('FeaturedMoE_v3', 'FeaturedMoE_V3'),
    'FeaturedMoE_V3': ('FeaturedMoE_v3', 'FeaturedMoE_V3'),
    'featured_moe_v3': ('FeaturedMoE_v3', 'FeaturedMoE_V3'),
    'featuredmoe_v3': ('FeaturedMoE_v3', 'FeaturedMoE_V3'),
    'FeaturedMoE_v4_Distillation': ('FeaturedMoE_v4_Distillation', 'FeaturedMoE_V4_Distillation'),
    'FeaturedMoE_V4_Distillation': ('FeaturedMoE_v4_Distillation', 'FeaturedMoE_V4_Distillation'),
    'featured_moe_v4_distillation': ('FeaturedMoE_v4_Distillation', 'FeaturedMoE_V4_Distillation'),
    'featuredmoe_v4_distillation': ('FeaturedMoE_v4_Distillation', 'FeaturedMoE_V4_Distillation'),
    'FeaturedMoE_N': ('FeaturedMoE_N', 'FeaturedMoE_N'),
    'featured_moe_n': ('FeaturedMoE_N', 'FeaturedMoE_N'),
    'featuredmoe_n': ('FeaturedMoE_N', 'FeaturedMoE_N'),
    'FeaturedMoE_N2': ('FeaturedMoE_N2', 'FeaturedMoE_N2'),
    'featured_moe_n2': ('FeaturedMoE_N2', 'FeaturedMoE_N2'),
    'featuredmoe_n2': ('FeaturedMoE_N2', 'FeaturedMoE_N2'),
    'FeaturedMoE_N3': ('FeaturedMoE_N3', 'FeaturedMoE_N3'),
    'featured_moe_n3': ('FeaturedMoE_N3', 'FeaturedMoE_N3'),
    'featuredmoe_n3': ('FeaturedMoE_N3', 'FeaturedMoE_N3'),
}

_CUSTOM_MODEL_CACHE = {}


def _get_custom_model(model_name):
    """Try to load requested custom model only."""
    if model_name in _CUSTOM_MODEL_CACHE:
        return _CUSTOM_MODEL_CACHE[model_name]

    spec = _CUSTOM_MODEL_SPECS.get(model_name)
    if spec is None:
        return None

    module_name, class_name = spec
    try:
        module = importlib.import_module(f"models.{module_name}")
        cls = getattr(module, class_name)
        _CUSTOM_MODEL_CACHE[model_name] = cls
        return cls
    except Exception:
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


# ============ Patch: create_dataset cache auto-recovery ============
_original_create_dataset = recbole_data_utils.create_dataset


def _resolve_dataset_cache_file(config):
    """Resolve RecBole dataset cache file path used by create_dataset()."""
    dataset_module = importlib.import_module("recbole.data.dataset")
    model_name = str(config["model"])
    if hasattr(dataset_module, model_name + "Dataset"):
        dataset_class = getattr(dataset_module, model_name + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = Path(str(config["checkpoint_dir"])) / f'{config["dataset"]}-{dataset_class.__name__}.pth'
    configured = str(config["dataset_save_path"]) if "dataset_save_path" in config else ""
    return Path(configured) if configured else default_file


def _patched_create_dataset(config):
    """Recover from corrupted dataset cache files and rebuild safely."""
    _ensure_history_input_load_col(config)
    try:
        return _original_create_dataset(config)
    except Exception as exc:
        cache_file = None
        try:
            cache_file = _resolve_dataset_cache_file(config)
        except Exception:
            cache_file = None

        recoverable = isinstance(exc, (TypeError, EOFError, pickle.UnpicklingError, AttributeError, ValueError))
        if not recoverable:
            raise

        logger = recbole_data_utils.getLogger()
        if cache_file is not None and cache_file.exists():
            try:
                backup = cache_file.with_name(f"{cache_file.name}.corrupt_{int(_time.time())}")
                cache_file.rename(backup)
                logger.warning(f"[CacheRecover] moved corrupted dataset cache -> {backup}")
            except Exception:
                try:
                    cache_file.unlink()
                    logger.warning(f"[CacheRecover] removed corrupted dataset cache -> {cache_file}")
                except Exception:
                    logger.warning(f"[CacheRecover] failed to move/remove corrupted cache: {cache_file}")

        # Avoid reusing a shared cache path during this process after recovery.
        try:
            if "save_dataset" in config:
                config["save_dataset"] = False
            if "dataset_save_path" in config:
                config["dataset_save_path"] = ""
        except Exception:
            pass

        logger.warning(f"[CacheRecover] rebuilding dataset after cache load failure: {exc}")
        _ensure_history_input_load_col(config)
        return _original_create_dataset(config)


recbole_data_utils.create_dataset = _patched_create_dataset
if hasattr(recbole_data_module, "create_dataset"):
    recbole_data_module.create_dataset = _patched_create_dataset


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
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_v2",
        "featuredmoe_v2",
        "featured_moe_v3",
        "featuredmoe_v3",
        "featured_moe_v4_distillation",
        "featuredmoe_v4_distillation",
        "featured_moe_n",
        "featuredmoe_n",
        "featured_moe_n2",
        "featuredmoe_n2",
        "featured_moe_n3",
        "featuredmoe_n3",
    }
    use_fp16 = bool(dataset.config["fmoe_feature_fp16"]) if "fmoe_feature_fp16" in dataset.config else default_fp16

    def _is_feature_field(name: str) -> bool:
        return (
            name.startswith("mac_")
            or name.startswith("mac5_")
            or name.startswith("mac10_")
            or name.startswith("mid_")
            or name.startswith("mic_")
        )

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
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_v2",
        "featuredmoe_v2",
        "featured_moe_v3",
        "featuredmoe_v3",
        "featured_moe_v4_distillation",
        "featuredmoe_v4_distillation",
        "featured_moe_n",
        "featuredmoe_n",
        "featured_moe_n2",
        "featuredmoe_n2",
        "featured_moe_n3",
        "featuredmoe_n3",
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


def _select_sequence_history_fields(dataset, fields):
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    model_name = str(dataset.config["model"]).lower() if "model" in dataset.config else ""

    history_fields = [field for field in fields if field != uid_field]
    if model_name in {
        "featured_moe",
        "featuredmoe",
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_v2",
        "featuredmoe_v2",
        "featured_moe_v3",
        "featuredmoe_v3",
        "featured_moe_v4_distillation",
        "featuredmoe_v4_distillation",
        "featured_moe_n",
        "featuredmoe_n",
        "featured_moe_n2",
        "featuredmoe_n2",
        "featured_moe_n3",
        "featuredmoe_n3",
    }:
        history_fields = [
            field
            for field in history_fields
            if (
                field == iid_field
                or field.startswith("mac_")
                or field.startswith("mac5_")
                or field.startswith("mac10_")
                or field.startswith("mid_")
                or field.startswith("mic_")
            )
        ]
    return history_fields


def _build_sequence_interaction_from_history_indices(
    dataset,
    *,
    sorted_values,
    target_idx_np,
    history_idx_np,
    seq_len_np,
    current_session_len_np=None,
):
    if target_idx_np.size == 0:
        return None

    max_len = int(dataset.config["MAX_ITEM_LIST_LENGTH"])
    list_suffix = dataset.config["LIST_SUFFIX"] if "LIST_SUFFIX" in dataset.config else "_list"
    item_list_length_field = (
        dataset.config["ITEM_LIST_LENGTH_FIELD"] if "ITEM_LIST_LENGTH_FIELD" in dataset.config else "item_length"
    )
    current_session_len_field = _current_session_item_length_field(dataset.config)
    chunk_size = int(dataset.config["sequence_convert_chunk_size"]) if "sequence_convert_chunk_size" in dataset.config else 16384
    chunk_size = max(1024, chunk_size)
    model_name = str(dataset.config["model"]).lower() if "model" in dataset.config else ""
    default_fp16 = model_name in {
        "featured_moe",
        "featuredmoe",
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_v2",
        "featuredmoe_v2",
        "featured_moe_v3",
        "featuredmoe_v3",
        "featured_moe_v4_distillation",
        "featuredmoe_v4_distillation",
        "featured_moe_n",
        "featuredmoe_n",
        "featured_moe_n2",
        "featuredmoe_n2",
        "featured_moe_n3",
        "featuredmoe_n3",
    }
    use_fp16 = bool(dataset.config["fmoe_feature_fp16"]) if "fmoe_feature_fp16" in dataset.config else default_fp16

    def _is_feature_field(name: str) -> bool:
        return (
            name.startswith("mac_")
            or name.startswith("mac5_")
            or name.startswith("mac10_")
            or name.startswith("mid_")
            or name.startswith("mic_")
        )

    fields = list(sorted_values.keys())
    history_fields = _select_sequence_history_fields(dataset, fields)

    target_idx = torch.from_numpy(target_idx_np.astype(np.int64, copy=False)).long()
    seq_len = torch.from_numpy(seq_len_np.astype(np.int64, copy=False)).long()
    safe_history_idx = torch.from_numpy(np.maximum(history_idx_np, 0).astype(np.int64, copy=False)).long()
    valid_history_mask = torch.from_numpy(history_idx_np >= 0)

    new_dict = {
        item_list_length_field: seq_len,
    }
    if current_session_len_np is not None:
        new_dict[current_session_len_field] = torch.from_numpy(
            current_session_len_np.astype(np.int64, copy=False)
        ).long()

    new_length = int(target_idx.shape[0])
    for field in fields:
        values = sorted_values[field]
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)

        new_dict[field] = values[target_idx]
        if field not in history_fields:
            continue

        hist_values = values
        if use_fp16 and _is_feature_field(field) and hist_values.is_floating_point():
            hist_values = hist_values.to(torch.float16)

        out_shape = (new_length, max_len) + tuple(hist_values.shape[1:])
        out = torch.zeros(out_shape, dtype=hist_values.dtype)

        for start in range(0, new_length, chunk_size):
            end = min(new_length, start + chunk_size)
            idx = safe_history_idx[start:end]
            valid = valid_history_mask[start:end]
            gathered = hist_values[idx]
            if gathered.dim() == 2:
                gathered = gathered.masked_fill(~valid, 0)
            else:
                mask = ~valid
                for _ in range(gathered.dim() - 2):
                    mask = mask.unsqueeze(-1)
                gathered = gathered.masked_fill(mask, 0)
            out[start:end] = gathered

        new_dict[field + list_suffix] = out

    return Interaction(new_dict)


def _convert_split_interactions_to_full_history_sequences(dataset, split_interactions, split_names):
    max_len = int(dataset.config["MAX_ITEM_LIST_LENGTH"])
    history_group_field = _history_group_field(dataset.config)
    target_group_field = _target_group_field(
        dataset.config,
        default=str(_config_get(dataset.config, "SESSION_ID_FIELD", dataset.uid_field) or dataset.uid_field),
    )
    history_policy = _history_eval_policy(dataset.config)
    if history_policy not in {"strict_train_prefix", "rolling_observed_prefix"}:
        raise ValueError(
            "history_eval_policy must be one of ['strict_train_prefix','rolling_observed_prefix'], "
            f"got {history_policy}"
        )

    non_empty = [inter for inter in split_interactions if inter is not None]
    if not non_empty:
        return [None for _ in split_interactions]

    fields = list(non_empty[0].columns)
    if history_group_field not in fields:
        raise ValueError(
            f"history_input_mode=full_history_session_targets requires '{history_group_field}' in loaded inter columns. "
            "Add it to load_col.inter or use a feature_mode that loads it."
        )
    if target_group_field not in fields:
        raise ValueError(
            f"history_input_mode=full_history_session_targets requires '{target_group_field}' in loaded inter columns."
        )
    if dataset.time_field not in fields:
        raise ValueError(
            f"history_input_mode=full_history_session_targets requires time field '{dataset.time_field}'."
        )

    combined_values = {}
    split_id_parts = []
    total_rows = 0
    for split_idx, inter in enumerate(split_interactions):
        if inter is None:
            continue
        split_len = len(inter)
        total_rows += split_len
        split_id_parts.append(torch.full((split_len,), split_idx, dtype=torch.long))
        for field in fields:
            value = inter[field]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            combined_values.setdefault(field, []).append(value)

    if total_rows == 0:
        return [None for _ in split_interactions]

    merged_values = {field: torch.cat(parts, dim=0) for field, parts in combined_values.items()}
    split_id = torch.cat(split_id_parts, dim=0)
    original_idx = np.arange(total_rows, dtype=np.int64)

    history_group_np = merged_values[history_group_field].cpu().numpy()
    time_np = merged_values[dataset.time_field].cpu().numpy()
    order_np = np.lexsort((original_idx, time_np, history_group_np))
    order_t = torch.from_numpy(order_np.astype(np.int64, copy=False)).long()

    sorted_values = {field: value[order_t] for field, value in merged_values.items()}
    split_sorted_np = split_id[order_t].cpu().numpy()
    history_group_sorted_np = sorted_values[history_group_field].cpu().numpy()
    target_group_sorted_np = sorted_values[target_group_field].cpu().numpy()

    user_changes = np.nonzero(history_group_sorted_np[1:] != history_group_sorted_np[:-1])[0] + 1 if total_rows > 1 else np.array([], dtype=np.int64)
    user_starts = np.concatenate(([0], user_changes))
    user_ends = np.concatenate((user_changes, [total_rows]))

    target_idx_list = []
    seq_len_list = []
    current_session_len_list = []
    history_rows = []
    split_sample_ids = [[] for _ in split_interactions]

    for user_start, user_end in zip(user_starts, user_ends):
        user_positions = np.arange(user_start, user_end, dtype=np.int32)
        user_split = split_sorted_np[user_start:user_end]
        user_target_group = target_group_sorted_np[user_start:user_end]

        user_train_positions = user_positions[user_split == 0]
        user_valid_positions = user_positions[user_split == 1]

        if user_positions.size == 0:
            continue

        session_changes = (
            np.nonzero(user_target_group[1:] != user_target_group[:-1])[0] + 1
            if user_target_group.size > 1
            else np.array([], dtype=np.int64)
        )
        session_starts = np.concatenate(([0], session_changes))
        session_ends = np.concatenate((session_changes, [user_target_group.size]))

        for sess_start_rel, sess_end_rel in zip(session_starts, session_ends):
            sess_start = int(user_start + sess_start_rel)
            sess_end = int(user_start + sess_end_rel)
            session_len = sess_end - sess_start
            if session_len < 2:
                continue

            session_split = int(split_sorted_np[sess_start])
            if np.any(split_sorted_np[sess_start:sess_end] != session_split):
                raise ValueError("A target session spans multiple split files; expected one split per session.")

            train_cut = np.searchsorted(user_train_positions, sess_start, side="left")
            valid_cut = np.searchsorted(user_valid_positions, sess_start, side="left")
            cross_base = user_train_positions[:train_cut]
            if session_split == 2 and history_policy == "rolling_observed_prefix":
                cross_base = np.concatenate([cross_base, user_valid_positions[:valid_cut]], axis=0)

            if session_split == 0:
                rel_targets = np.arange(1, session_len, dtype=np.int32)
            else:
                rel_targets = np.array([session_len - 1], dtype=np.int32)

            for rel_target in rel_targets:
                target_pos = sess_start + int(rel_target)
                same_session_prefix = np.arange(sess_start, target_pos, dtype=np.int32)
                if cross_base.size == 0:
                    history_positions = same_session_prefix
                elif same_session_prefix.size == 0:
                    history_positions = cross_base
                else:
                    history_positions = np.concatenate([cross_base, same_session_prefix], axis=0)

                if history_positions.size == 0:
                    continue

                selected = history_positions[-max_len:]
                sample_idx = len(target_idx_list)
                target_idx_list.append(int(target_pos))
                seq_len_list.append(int(selected.size))
                current_session_len_list.append(int(min(int(rel_target), max_len)))
                history_rows.append(selected)
                split_sample_ids[session_split].append(sample_idx)

    if not target_idx_list:
        return [None for _ in split_interactions]

    n_samples = len(target_idx_list)
    history_idx_np = np.full((n_samples, max_len), -1, dtype=np.int32)
    for sample_idx, selected in enumerate(history_rows):
        history_idx_np[sample_idx, : selected.size] = selected

    target_idx_np = np.asarray(target_idx_list, dtype=np.int64)
    seq_len_np = np.asarray(seq_len_list, dtype=np.int64)
    current_session_len_np = np.asarray(current_session_len_list, dtype=np.int64)

    converted_splits = []
    for split_idx in range(len(split_interactions)):
        sample_ids = np.asarray(split_sample_ids[split_idx], dtype=np.int64)
        if sample_ids.size == 0:
            converted_splits.append(None)
            continue
        converted_splits.append(
            _build_sequence_interaction_from_history_indices(
                dataset,
                sorted_values=sorted_values,
                target_idx_np=target_idx_np[sample_ids],
                history_idx_np=history_idx_np[sample_ids],
                seq_len_np=seq_len_np[sample_ids],
                current_session_len_np=current_session_len_np[sample_ids],
            )
        )

    return converted_splits


def _make_converted_sequence_dataset(base_dataset, converted, *, list_suffix, item_list_length_field, max_len):
    new_dataset = base_dataset.copy(converted)
    if not isinstance(new_dataset.inter_feat, Interaction):
        new_dataset.inter_feat = converted

    new_dataset.item_id_list_field = base_dataset.iid_field + list_suffix
    new_dataset.item_list_length_field = item_list_length_field

    for field in converted.columns:
        if field.endswith(list_suffix):
            base_field = field[:-len(list_suffix)]
            setattr(new_dataset, f"{base_field}_list_field", field)
            if field not in new_dataset.field2type:
                new_dataset.field2type[field] = FeatureType.TOKEN_SEQ if base_field == base_dataset.iid_field else FeatureType.FLOAT_SEQ
                new_dataset.field2seqlen[field] = max_len

    if item_list_length_field not in new_dataset.field2type:
        new_dataset.field2type[item_list_length_field] = FeatureType.TOKEN
        new_dataset.field2seqlen[item_list_length_field] = 1

    current_session_len_field = _current_session_item_length_field(base_dataset.config)
    if current_session_len_field in converted.columns and current_session_len_field not in new_dataset.field2type:
        new_dataset.field2type[current_session_len_field] = FeatureType.TOKEN
        new_dataset.field2seqlen[current_session_len_field] = 1

    return new_dataset


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
        "featured_moe_hgr",
        "featuredmoe_hgr",
        "featured_moe_v2",
        "featuredmoe_v2",
        "featured_moe_v3",
        "featuredmoe_v3",
        "featured_moe_v4_distillation",
        "featuredmoe_v4_distillation",
        "featured_moe_n",
        "featuredmoe_n",
        "featured_moe_n2",
        "featuredmoe_n2",
        "featured_moe_n3",
        "featuredmoe_n3",
    }
    use_fp16 = bool(self.config["fmoe_feature_fp16"]) if "fmoe_feature_fp16" in self.config else default_fp16

    def _is_feature_field(name: str) -> bool:
        return (
            name.startswith("mac_")
            or name.startswith("mac5_")
            or name.startswith("mac10_")
            or name.startswith("mid_")
            or name.startswith("mic_")
        )

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
        history_mode = _history_input_mode(self.config)
        history_policy = _history_eval_policy(self.config)
        split_cache_enabled = True if "enable_session_split_cache" not in self.config else bool(self.config["enable_session_split_cache"])
        split_cache_file = None
        if split_cache_enabled and "checkpoint_dir" in self.config:
            try:
                ckpt_dir = Path(self.config["checkpoint_dir"])
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ds_name = getattr(self, "dataset_name", "dataset")
                data_path = str(self.config["data_path"]) if "data_path" in self.config else ""
                feature_mode = str(self.config["feature_mode"]) if "feature_mode" in self.config else ""
                load_col = self.config["load_col"] if "load_col" in self.config else {}
                data_sig = _dataset_file_signature(self.config, ds_name)
                cache_key = json.dumps(
                    {
                        "data_path": data_path,
                        "feature_mode": feature_mode,
                        "load_col": load_col,
                        "history_input_mode": history_mode,
                        "history_group_field": _history_group_field(self.config),
                        "target_group_field": _target_group_field(
                            self.config,
                            default=str(_config_get(self.config, "SESSION_ID_FIELD", self.uid_field) or self.uid_field),
                        ),
                        "history_eval_policy": history_policy,
                        "current_session_item_length_field": _current_session_item_length_field(self.config),
                        "data_sig": data_sig,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                cache_sig = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
                split_cache_file = ckpt_dir / f"{ds_name}-session-split-len{max_len}-{cache_sig}.pth"
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
        split_names = ['train', 'valid', 'test']
        split_interactions = [self.inter_feat[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]

        if history_mode == "full_history_session_targets":
            self.logger.info(
                f"SESSION mode: Using full-history session targets | history_group_field={_history_group_field(self.config)} "
                f"| target_group_field={_target_group_field(self.config, default=str(_config_get(self.config, 'SESSION_ID_FIELD', self.uid_field) or self.uid_field))} "
                f"| policy={history_policy}"
            )
            converted_splits = _convert_split_interactions_to_full_history_sequences(
                self,
                split_interactions=split_interactions,
                split_names=split_names,
            )
        else:
            converted_splits = []
            for i, split_inter in enumerate(split_interactions):
                converted_splits.append(_convert_inter_to_sequence(self, split_inter, for_training=(i == 0)))

        datasets = []
        for i, (split_inter, converted) in enumerate(zip(split_interactions, converted_splits)):
            if converted is not None:
                new_dataset = _make_converted_sequence_dataset(
                    self,
                    converted,
                    list_suffix=list_suffix,
                    item_list_length_field=item_list_length_field,
                    max_len=max_len,
                )
                datasets.append(new_dataset)
                try:
                    extra = ""
                    current_session_len_field = _current_session_item_length_field(self.config)
                    if current_session_len_field in converted.columns:
                        avg_current = float(converted[current_session_len_field].float().mean().item())
                        extra = f" | avg_current_session_len={avg_current:.2f}"
                    self.logger.info(
                        f"  {split_names[i]}: converted sequences={len(converted)} | rows={len(split_inter)} | max_len={max_len}{extra}"
                    )
                except Exception:
                    pass
            else:
                self.logger.warning(f"  {split_names[i]}: No valid sequences!")
                datasets.append(self.copy(self.inter_feat[:0]))
        
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
                data_path = str(self.config["data_path"]) if "data_path" in self.config else ""
                feature_mode = str(self.config["feature_mode"]) if "feature_mode" in self.config else ""
                load_col = self.config["load_col"] if "load_col" in self.config else {}
                data_sig = _dataset_file_signature(self.config, ds_name)
                eval_key = json.dumps(
                    {
                        "eval_args": eval_args,
                        "data_path": data_path,
                        "feature_mode": feature_mode,
                        "load_col": load_col,
                        "data_sig": data_sig,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
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
            valid_metric = str(_config_get(self.config, 'valid_metric', 'NDCG@10')).lower()
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


def _build_special_metric_item_counts(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None:
        return None
    inter_feat = getattr(dataset, "inter_feat", None)
    iid_field = getattr(dataset, "iid_field", None)
    n_items = int(getattr(dataset, "item_num", 0))
    if inter_feat is None or iid_field is None or n_items <= 0:
        return None
    item_ids = inter_feat[iid_field]
    if not torch.is_tensor(item_ids):
        item_ids = torch.as_tensor(item_ids)
    item_ids = item_ids.long().clamp(min=0)
    return torch.bincount(item_ids, minlength=n_items)[:n_items].cpu()


def _seen_target_mask_from_train_counts(trainer, positive_i: torch.Tensor, collector=None):
    """Return boolean mask for targets seen in train split (internal item ids)."""
    item_counts = getattr(trainer, "_fmoe_special_item_counts_train", None)
    if item_counts is None:
        item_counts = getattr(trainer, "_fmoe_special_item_counts", None)
    if item_counts is None and collector is not None:
        item_counts = getattr(collector, "item_counts", None)
    if item_counts is None:
        return None
    if not torch.is_tensor(item_counts):
        item_counts = torch.as_tensor(item_counts)
    item_counts = item_counts.long().cpu()
    if item_counts.numel() <= 0:
        return None
    pos_cpu = positive_i.detach().long().cpu().clamp(min=0, max=max(int(item_counts.numel()) - 1, 0))
    return item_counts.index_select(0, pos_cpu) > 0


def _update_main_eval_unseen_stats(trainer, split_name: str, *, total: int, seen: int, unseen: int, dropped_rows: int) -> None:
    stats = getattr(trainer, "_main_eval_unseen_filter_stats", None)
    if not isinstance(stats, dict):
        stats = {}
        trainer._main_eval_unseen_filter_stats = stats
    split = str(split_name or "unknown")
    rec = stats.get(split)
    if not isinstance(rec, dict):
        rec = {
            "total_targets": 0,
            "seen_targets": 0,
            "unseen_targets": 0,
            "dropped_eval_rows": 0,
            "enabled": False,
        }
    rec["total_targets"] = int(rec.get("total_targets", 0)) + int(total)
    rec["seen_targets"] = int(rec.get("seen_targets", 0)) + int(seen)
    rec["unseen_targets"] = int(rec.get("unseen_targets", 0)) + int(unseen)
    rec["dropped_eval_rows"] = int(rec.get("dropped_eval_rows", 0)) + int(dropped_rows)
    rec["enabled"] = True
    stats[split] = rec


def _empty_like_eval_result(scores: torch.Tensor, interaction):
    n_items = int(scores.size(1)) if scores.ndim >= 2 else 0
    empty_scores = scores.new_empty((0, n_items))
    try:
        empty_interaction = interaction[[]]
    except Exception:
        empty_interaction = interaction
    device = scores.device
    empty_idx = torch.empty((0,), dtype=torch.long, device=device)
    return empty_interaction, empty_scores, empty_idx, empty_idx


def begin_special_eval(trainer, data_loader, *, split_name: str):
    model_name = str(_config_get(trainer.config, "model", "")).lower()
    generic_enabled = bool(_config_get(trainer.config, "special_logging", False))
    fmoe_enabled = (
        (model_name.startswith("featured_moe") or model_name.startswith("featuredmoe"))
        and bool(_config_get(trainer.config, "fmoe_special_logging", True))
    )
    if not (generic_enabled or fmoe_enabled):
        return

    from models.FeaturedMoE.special_metrics import (
        SpecialMetricCollector,
        build_special_metric_config_snapshot,
        default_new_user_field,
    )

    item_count_source = "unknown"
    item_counts = getattr(trainer, "_fmoe_special_item_counts_train", None)
    if item_counts is not None:
        item_count_source = "train_split"
    if item_counts is None:
        item_counts = getattr(trainer, "_fmoe_special_item_counts", None)
        if item_counts is not None:
            item_count_source = "cached_loader"
    if item_counts is None:
        item_counts = _build_special_metric_item_counts(data_loader)
        trainer._fmoe_special_item_counts = item_counts
        if item_counts is not None:
            item_count_source = "eval_loader_fallback"
    if item_counts is None:
        return

    item_seq_len_field = getattr(getattr(trainer, "model", None), "ITEM_SEQ_LEN", None)
    if item_seq_len_field is None:
        item_seq_len_field = _config_get(trainer.config, "ITEM_LIST_LENGTH_FIELD", "item_length")
    new_user_field = default_new_user_field()
    new_user_base_field = new_user_field[:-5] if new_user_field.endswith("_list") else new_user_field
    new_user_available = False
    dataset_fields = set()
    dataset = getattr(data_loader, "dataset", None)
    inter_feat = getattr(dataset, "inter_feat", None)
    try:
        if inter_feat is not None:
            dataset_fields.update(getattr(inter_feat, "columns", []))
    except Exception:
        pass
    try:
        field2type = getattr(dataset, "field2type", None)
        if isinstance(field2type, dict):
            dataset_fields.update(field2type.keys())
    except Exception:
        pass
    if new_user_field in dataset_fields or new_user_base_field in dataset_fields:
        new_user_available = True
    if not new_user_available:
        new_user_field = None
    snapshot = build_special_metric_config_snapshot(
        feature_available=True,
        new_user_available=new_user_available,
    )
    snapshot["split"] = str(split_name)
    snapshot["item_seq_len_field"] = str(item_seq_len_field)
    snapshot["new_user_field"] = str(new_user_field or "")
    snapshot["item_count_source"] = str(item_count_source)

    trainer._fmoe_special_metric_collector = SpecialMetricCollector(
        split_name=split_name,
        item_counts=item_counts,
        item_seq_len_field=item_seq_len_field,
        new_user_field=new_user_field,
        config_snapshot=snapshot,
    )


def end_special_eval(trainer):
    collector = getattr(trainer, "_fmoe_special_metric_collector", None)
    if collector is None:
        return None
    summary = collector.finalize()
    trainer._fmoe_special_metric_collector = None
    return summary


def begin_diagnostic_eval(trainer, *, split_name: str):
    model = getattr(trainer, "model", None)
    if model is None or not hasattr(model, "begin_diagnostic_eval"):
        return
    try:
        model.begin_diagnostic_eval(split_name=split_name)
    except Exception:
        pass


def end_diagnostic_eval(trainer):
    model = getattr(trainer, "model", None)
    if model is None or not hasattr(model, "end_diagnostic_eval"):
        return None
    try:
        return model.end_diagnostic_eval()
    except Exception:
        return None


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
    interaction, scores, positive_u, positive_i = result
    if sampled_mask is not None:
        # Expand mask to batch size
        # scores shape: (batch_size, n_items)
        # sampled_mask shape: (n_items,) boolean

        # Set scores for non-candidate items to -inf
        # But keep positive items (they should always be evaluated)
        mask = sampled_mask.clone()
        mask[positive_i] = True  # Always include positive items

        # Apply mask: non-candidate items get -inf
        scores[:, ~mask] = float('-inf')

    collector = getattr(self, "_fmoe_special_metric_collector", None)
    if collector is not None:
        try:
            collector.update(
                interaction=interaction,
                scores=scores,
                positive_u=positive_u,
                positive_i=positive_i,
            )
        except Exception as e:
            try:
                self.logger.warning(f"special metric collection skipped for one batch: {e}")
            except Exception:
                pass

    # Optional: exclude unseen targets from main eval metrics while preserving
    # special collector stats (collector update happened before this filter).
    try:
        exclude_unseen = bool(_config_get(self.config, "exclude_unseen_target_from_main_eval", False))
    except Exception:
        exclude_unseen = False
    if exclude_unseen:
        split_name = "unknown"
        if collector is not None:
            split_name = str(getattr(collector, "split_name", "unknown"))
        seen_pos_mask = _seen_target_mask_from_train_counts(self, positive_i, collector=collector)
        if seen_pos_mask is not None:
            total_pos = int(positive_i.numel())
            seen_pos = int(seen_pos_mask.sum().item())
            unseen_pos = int(total_pos - seen_pos)
            if total_pos > 0 and unseen_pos > 0:
                keep_pos_idx = seen_pos_mask.nonzero(as_tuple=False).view(-1).to(device=positive_i.device)
                keep_positive_u_old = positive_u.index_select(0, keep_pos_idx).long()
                keep_positive_i = positive_i.index_select(0, keep_pos_idx).long()

                orig_rows = int(scores.size(0))
                row_keep = torch.zeros(orig_rows, dtype=torch.bool, device=scores.device)
                if keep_positive_u_old.numel() > 0:
                    row_keep[keep_positive_u_old] = True
                kept_rows_idx = row_keep.nonzero(as_tuple=False).view(-1)
                dropped_rows = int(orig_rows - kept_rows_idx.numel())

                if keep_positive_u_old.numel() <= 0 or kept_rows_idx.numel() <= 0:
                    _update_main_eval_unseen_stats(
                        self,
                        split_name,
                        total=total_pos,
                        seen=seen_pos,
                        unseen=unseen_pos,
                        dropped_rows=dropped_rows if dropped_rows > 0 else orig_rows,
                    )
                    return _empty_like_eval_result(scores, interaction)

                if kept_rows_idx.numel() < orig_rows:
                    scores = scores.index_select(0, kept_rows_idx)
                    interaction = interaction[kept_rows_idx.detach().cpu()]
                    remap = torch.full(
                        (orig_rows,),
                        fill_value=-1,
                        dtype=torch.long,
                        device=keep_positive_u_old.device,
                    )
                    remap[kept_rows_idx.to(device=keep_positive_u_old.device)] = torch.arange(
                        kept_rows_idx.numel(),
                        dtype=torch.long,
                        device=keep_positive_u_old.device,
                    )
                    positive_u = remap.index_select(0, keep_positive_u_old)
                    positive_i = keep_positive_i
                _update_main_eval_unseen_stats(
                    self,
                    split_name,
                    total=total_pos,
                    seen=seen_pos,
                    unseen=unseen_pos,
                    dropped_rows=dropped_rows,
                )
            elif total_pos > 0:
                _update_main_eval_unseen_stats(
                    self,
                    split_name,
                    total=total_pos,
                    seen=seen_pos,
                    unseen=unseen_pos,
                    dropped_rows=0,
                )

    return (interaction, scores, positive_u, positive_i)

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
