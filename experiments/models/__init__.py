"""
Custom sequential recommendation models for FMoE experiments.
All models inherit from RecBole's AbstractRecommender for compatibility.
"""

from .bilstm import BiLSTM
from .clrec import CLRec
from .bsarec import BSARec
from .fame import FAME
from .sigma import SIGMA
from .difsr import DIFSR
from .mssr import MSSR
from .patt import PAtt
from .fenrec import FENRec
from .FeaturedMoE import FeaturedMoE
from .FeaturedMoE_HiR import FeaturedMoE_HiR

__all__ = [
	"BiLSTM",
	"CLRec",
	"BSARec",
	"FAME",
	"SIGMA",
	"DIFSR",
	"MSSR",
	"PAtt",
	"FENRec",
	"FeaturedMoE",
	"FeaturedMoE_HiR",
]
