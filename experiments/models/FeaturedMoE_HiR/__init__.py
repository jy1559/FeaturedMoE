"""
FeaturedMoE_HiR - Hierarchical-in-Stage routing variant for FeaturedMoE.

- Two-level routing inside each stage: bundle -> intra-bundle experts
- Experts are hidden-only FFNs
- Stage merge mode: serial (default) or parallel (stage-gate)
"""

from .featured_moe_hir import FeaturedMoE_HiR

__all__ = ["FeaturedMoE_HiR"]
