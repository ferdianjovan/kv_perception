"""Frozen startup environment colours and consecutive-frame confirmation.

Only colours inside the smoke/fog HSV palette participate in calibration. A
quantized BGR histogram keeps joint colour combinations, so unrelated channel
extrema cannot suppress a broad range of possible smoke colours.
"""

import math
from typing import Sequence

from kv_perception.smoke_detection import (
    _bounded_image,
    _color_bands,
    _integer_in_range,
    _palette_mask,
    _validate_image,
    ColorRange,
    DetectionResult,
)
import numpy as np


class BackgroundColorModel:
    """Learn persistent, sufficiently common palette colours, then freeze.

    Calibration must view a representative smoke-free environment. Each colour
    bin must occupy ``min_pixel_fraction`` of the entire processing image in at
    least ``min_frame_fraction`` of calibration frames. The exclusion buffer is
    a cube in quantized BGR space and applies globally, including after movement.
    """

    def __init__(
        self,
        color_ranges: Sequence[ColorRange],
        calibration_frames: int = 30,
        bin_size: int = 8,
        min_pixel_fraction: float = 0.001,
        min_frame_fraction: float = 0.6,
        color_margin: int = 12,
        max_processing_width: int = 640,
    ):
        self.color_ranges = tuple(color_ranges)
        self._bands = _color_bands(self.color_ranges)
        self.calibration_frames = _integer_in_range(
            'calibration_frames', calibration_frames, 1, 2**31 - 1,
        )
        self.bin_size = _integer_in_range('bin_size', bin_size, 1, 256)
        self.color_margin = _integer_in_range('color_margin', color_margin, 0, 255)
        self.max_processing_width = _integer_in_range(
            'max_processing_width', max_processing_width, 1, 2**31 - 1,
        )
        for name, value in (
            ('min_pixel_fraction', min_pixel_fraction),
            ('min_frame_fraction', min_frame_fraction),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f'{name} must be finite and in (0, 1]')
        self.min_pixel_fraction = float(min_pixel_fraction)
        self.min_frame_fraction = float(min_frame_fraction)
        self._bins = math.ceil(256 / self.bin_size)
        self.reset()

    @property
    def ready(self) -> bool:
        return self.frames_seen >= self.calibration_frames

    @property
    def progress(self) -> float:
        return min(1.0, self.frames_seen / self.calibration_frames)

    @property
    def excluded_bin_count(self) -> int:
        return int(np.count_nonzero(self._excluded))

    def reset(self) -> None:
        """Discard the frozen baseline and restart calibration."""
        self.frames_seen = 0
        self._support = np.zeros(self._bins**3, dtype=np.uint32)
        self._excluded = np.zeros((self._bins,) * 3, dtype=bool)

    def _indices(self, bgr_image: np.ndarray) -> np.ndarray:
        channels = bgr_image.astype(np.int32) // self.bin_size
        return (
            (channels[:, :, 0] * self._bins + channels[:, :, 1]) * self._bins
            + channels[:, :, 2]
        )

    def observe(self, bgr_image: np.ndarray) -> None:
        """Accumulate one valid startup frame; never adapt a frozen baseline."""
        image = _bounded_image(bgr_image, self.max_processing_width)
        if self.ready:
            return
        palette = _palette_mask(image, self._bands) != 0
        counts = np.bincount(self._indices(image)[palette], minlength=self._bins**3)
        required_pixels = max(1, math.ceil(palette.size * self.min_pixel_fraction))
        self._support += counts >= required_pixels
        self.frames_seen += 1
        if self.ready:
            required_frames = math.ceil(self.calibration_frames * self.min_frame_fraction)
            learned = (self._support >= required_frames).reshape((self._bins,) * 3)
            radius = min(self._bins - 1, math.ceil(self.color_margin / self.bin_size))
            self._excluded = self._buffer_colors(learned, radius)

    @staticmethod
    def _buffer_colors(learned: np.ndarray, radius: int) -> np.ndarray:
        """Dilate all three colour axes, without treating one axis as channels."""
        expanded = learned.copy()
        if radius == 0 or not np.any(expanded):
            return expanded
        # A sliding sum along each axis gives a separable 3-D cube dilation.
        # Padding with zeros keeps opposite ends of colour space separate.
        for axis in range(3):
            padding = [(0, 0)] * 3
            padding[axis] = (radius + 1, radius)
            summed = np.cumsum(np.pad(expanded, padding), axis=axis, dtype=np.int32)
            upper = [slice(None)] * 3
            lower = [slice(None)] * 3
            upper[axis] = slice(2 * radius + 1, None)
            lower[axis] = slice(None, -(2 * radius + 1))
            expanded = summed[tuple(upper)] - summed[tuple(lower)] > 0
        return expanded

    def exclusion_mask(self, bgr_image: np.ndarray) -> np.ndarray:
        """Return a bool mask at the supplied resolution; calibration never mutates it."""
        _validate_image(bgr_image)
        if not self.ready:
            return np.zeros(bgr_image.shape[:2], dtype=bool)
        return self._excluded.reshape(-1)[self._indices(bgr_image)]


class ConfirmationTracker:
    """Expose a blob only after a stable sequence of consecutive detections.

    Stability is a normalized centroid-distance threshold, not a probability.
    A missing frame result, time discontinuity, excessive gap, or a different
    target restarts confirmation. Quadrant boundaries do not break continuity.
    """

    def __init__(
        self,
        required_frames: int = 3,
        max_center_distance: float = 0.12,
        max_gap: float = 1.0,
    ):
        self.required_frames = _integer_in_range('required_frames', required_frames, 1, 2**31 - 1)
        if (
            not math.isfinite(max_center_distance)
            or not 0.0 <= max_center_distance <= math.sqrt(2)
        ):
            raise ValueError('max_center_distance must be finite and in [0, sqrt(2)]')
        if not math.isfinite(max_gap) or max_gap <= 0.0:
            raise ValueError('max_gap must be finite and positive')
        self.max_center_distance = float(max_center_distance)
        self.max_gap = float(max_gap)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self._previous_center = None
        self._previous_time = None

    def update(self, result: DetectionResult, now: float) -> DetectionResult:
        """Return no detection until the current candidate is confirmed."""
        if not math.isfinite(now):
            self.reset()
            raise ValueError('now must be finite')
        if not result.detected:
            self.reset()
            return DetectionResult()
        center = (result.center_x, result.center_y)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in center):
            self.reset()
            raise ValueError('Detection center must be finite and normalized to [0, 1]')
        continuous = (
            self._previous_time is not None
            and 0.0 <= now - self._previous_time <= self.max_gap
            and math.dist(center, self._previous_center) <= self.max_center_distance
        )
        self.count = min(self.required_frames, self.count + 1) if continuous else 1
        self._previous_time = float(now)
        self._previous_center = center
        return result if self.count >= self.required_frames else DetectionResult()
