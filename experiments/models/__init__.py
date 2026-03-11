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
from .FeaturedMoE_HGR import FeaturedMoE_HGR
from .FeaturedMoE_HGRv3 import FeaturedMoE_HGRv3
from .FeaturedMoE_HGRv4 import FeaturedMoE_HGRv4
from .FeaturedMoE_Individual import FeaturedMoE_Individual
from .FeaturedMoE_HiR2 import FeaturedMoE_HiR2
from .FeaturedMoE_ProtoX import FeaturedMoE_ProtoX
from .FeaturedMoE_v2 import FeaturedMoE_V2
from .FeaturedMoE_v2_HiR import FeaturedMoE_V2_HiR
from .FeaturedMoE_v3 import FeaturedMoE_V3
from .FeaturedMoE_v4_Distillation import FeaturedMoE_V4_Distillation

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
	"FeaturedMoE_HGR",
	"FeaturedMoE_HGRv3",
	"FeaturedMoE_HGRv4",
	"FeaturedMoE_Individual",
	"FeaturedMoE_HiR2",
	"FeaturedMoE_ProtoX",
	"FeaturedMoE_V2",
	"FeaturedMoE_V2_HiR",
	"FeaturedMoE_V3",
	"FeaturedMoE_V4_Distillation",
]
