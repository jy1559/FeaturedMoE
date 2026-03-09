"""FeaturedMoE_v2 package.

Key upgrades over v1:
- Object-based layout schema with explicit pass/MoE boundaries.
- Unified execution modes: serial, parallel, parallel+repeat.
- Optional stage-merge auxiliary loss in parallel mode.
"""

from .featured_moe_v2 import FeaturedMoE_V2

__all__ = ["FeaturedMoE_V2"]
