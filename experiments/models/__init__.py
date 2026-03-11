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
from .FeaturedMoE_HGR import FeaturedMoE_HGR
from .FeaturedMoE_HGRv4 import FeaturedMoE_HGRv4
from .FeaturedMoE_v2 import FeaturedMoE_V2
from .FeaturedMoE_v3 import FeaturedMoE_V3
from .FeaturedMoE_v4_Distillation import FeaturedMoE_V4_Distillation
from .FeaturedMoE_N import FeaturedMoE_N

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
	"FeaturedMoE_HGR",
	"FeaturedMoE_HGRv4",
	"FeaturedMoE_V2",
	"FeaturedMoE_V3",
	"FeaturedMoE_V4_Distillation",
	"FeaturedMoE_N",
]
