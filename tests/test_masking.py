from __future__ import annotations

import cv2
import numpy as np
import pytest

from vroom_interlagos.chunking.motion import DenseFlowConfig, dense_flow_stats
from vroom_interlagos.masking.static_mask import StaticMaskConfig, estimate_static_mask
from vroom_interlagos.synthetic import _texture


def _frames_with_static_block(n: int = 30) -> list[np.ndarray]:
    """Moving textured background with a static textured block (the 'car') in the lower half."""
    rng = np.random.default_rng(3)
    background = cv2.cvtColor(_texture(rng, 2000, 144, (1.0, 1.0, 1.0)), cv2.COLOR_BGR2GRAY)
    car = cv2.cvtColor(_texture(rng, 160, 60, (1.0, 1.0, 1.0)), cv2.COLOR_BGR2GRAY)
    frames = []
    for i in range(n):
        frame = background[:, i * 40 : i * 40 + 256].copy()
        frame[84:144, 48:208] = car
        frames.append(frame)
    return frames


def test_static_mask_finds_static_block() -> None:
    mask, std = estimate_static_mask(_frames_with_static_block(), StaticMaskConfig())
    assert mask.shape == (144, 256) and set(np.unique(mask)) <= {0, 255}
    block = mask[90:138, 54:202]  # interior of the static block
    assert (block == 0).mean() > 0.97
    outside = mask[:70]
    assert (outside == 255).mean() > 0.9
    assert std[100:130, 80:180].mean() < 2.0 < std[:70].mean()


def test_static_mask_needs_enough_frames() -> None:
    with pytest.raises(ValueError, match="at least"):
        estimate_static_mask(_frames_with_static_block(4), StaticMaskConfig())


def test_mask_restores_flow_statistics() -> None:
    """Without the mask the static block drags the median flow down; with it the true motion shows."""
    frames = _frames_with_static_block()
    a = frames[0]
    b = a.copy()
    b[:, :] = np.roll(a, -6, axis=1)
    b[84:144, 48:208] = a[84:144, 48:208]  # the block does not move
    mask = np.full(a.shape, 255, np.uint8)
    mask[84:144, 48:208] = 0
    cfg = DenseFlowConfig(scale=1.0)
    assert dense_flow_stats(a, b, cfg, mask)["flow_mag_median"] == pytest.approx(6.0, abs=1.0)
    mostly_car = np.zeros(a.shape, np.uint8)
    mostly_car[84:144, 48:208] = 255
    assert dense_flow_stats(a, b, cfg, mostly_car)["flow_mag_median"] < 1.5
