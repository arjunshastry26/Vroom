"""Module C: masks that exclude camera-body regions from analysis."""

from vroom_interlagos.masking.static_mask import StaticMaskConfig, estimate_static_mask

__all__ = ["StaticMaskConfig", "estimate_static_mask"]
