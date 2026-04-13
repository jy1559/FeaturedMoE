"""Custom sequential recommendation models for FMoE experiments.

Use lazy loading so a single optional dependency/import failure in one model
does not prevent all other custom models from being discoverable.
"""

from importlib import import_module

_MODEL_MODULES = {
	"BiLSTM": "bilstm",
	"CLRec": "clrec",
	"DuoRec": "duorec",
	"FEARec": "fearec",
	"FDSA": "fdsa",
	"TiSASRec": "tisasrec",
	"BSARec": "bsarec",
	"FAME": "fame",
	"SIGMA": "sigma",
	"DIFSR": "difsr",
	"MSSR": "mssr",
	"PAtt": "patt",
	"FENRec": "fenrec",
	"FeaturedMoE": "FeaturedMoE",
	"FeaturedMoE_HGR": "FeaturedMoE_HGR",
	"FeaturedMoE_HGRv4": "FeaturedMoE_HGRv4",
	"FeaturedMoE_V2": "FeaturedMoE_v2",
	"FeaturedMoE_V3": "FeaturedMoE_v3",
	"FeaturedMoE_V4_Distillation": "FeaturedMoE_v4_Distillation",
	"FeaturedMoE_N": "FeaturedMoE_N",
	"FeaturedMoE_N2": "FeaturedMoE_N2",
	"FeaturedMoE_N3": "FeaturedMoE_N3",
}

__all__ = list(_MODEL_MODULES.keys())


def __getattr__(name):
	module_name = _MODEL_MODULES.get(name)
	if module_name is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module = import_module(f".{module_name}", __name__)
	value = getattr(module, name)
	globals()[name] = value
	return value
