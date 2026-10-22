"""Naive palette-range smoke/fog detection with quadrant-level guidance.

This is a colour heuristic, so similarly coloured scenery can trigger it. Camera
angles assume a forward-facing, level camera. An RGB image alone cannot measure
range: a metric height correction requires an explicitly assumed target range.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from kv_perception.smoke_background import BackgroundColorModel


NONE = 0
TOP_LEFT = 1
TOP_RIGHT = 2
BOTTOM_LEFT = 3
BOTTOM_RIGHT = 4
REGION_NAMES = ('NONE', 'TOP LEFT', 'TOP RIGHT', 'BOTTOM LEFT', 'BOTTOM RIGHT')


@dataclass(frozen=True)
class ColorRange:
    """One inclusive OpenCV HSV band (H: 0..179, S/V: 0..255)."""

    name: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    """Accepted blob coverage and the largest blob's internal centroid.

    ``region`` is the quadrant of the largest blob's centroid. ``regions`` also
    includes other quadrants meeting the occupancy threshold. Coverage values
    use all accepted blobs; quadrant coverage divides by that quadrant's area.
    Centers are normalized by (image dimension - 1), for internal guidance only.
    """

    detected: bool = False
    region: int = NONE
    regions: tuple[int, ...] = ()
    quadrant_coverage: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    coverage: float = 0.0
    center_x: float = 0.5
    center_y: float = 0.5


def _integer_in_range(name, value, minimum, maximum):
    if not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return int(value)


def _hue_bands(hues: np.ndarray, margin: int) -> list[tuple[int, int]]:
    """Cover the shortest circular hue arc, splitting at OpenCV's hue seam."""
    values = np.unique(hues).astype(int)
    gaps = np.diff(np.r_[values, values[0] + 180])
    largest_gap = int(np.argmax(gaps))
    start = int(values[(largest_gap + 1) % len(values)])
    end = int(values[largest_gap])
    span = (end - start) % 180
    if span + 2 * margin >= 179:
        return [(0, 179)]
    lower = (start - margin) % 180
    upper = (end + margin) % 180
    if lower <= upper:
        return [(lower, upper)]
    return [(0, upper), (lower, 179)]


def load_color_ranges(
    paths: Iterable[str | Path],
    min_alpha: int = 16,
    hue_margin: int = 6,
    saturation_margin: int = 65,
    value_margin: int = 45,
) -> list[ColorRange]:
    """Derive padded HSV bands from visible pixels in the supplied PNGs.

    Pixels with alpha <= ``min_alpha`` are excluded, including their hidden RGB
    values. Achromatic palettes accept any hue because hue is undefined at zero
    saturation. Margins allow some rendered lighting and alpha-blending changes.
    """
    min_alpha = _integer_in_range('min_alpha', min_alpha, 0, 255)
    hue_margin = _integer_in_range('hue_margin', hue_margin, 0, 179)
    saturation_margin = _integer_in_range('saturation_margin', saturation_margin, 0, 255)
    value_margin = _integer_in_range('value_margin', value_margin, 0, 255)
    ranges = []
    for entry in paths:
        path = Path(entry)
        palette = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if palette is None:
            raise ValueError(f'Unable to read colour palette: {path}')
        if palette.dtype != np.uint8:
            raise ValueError(f'Colour palette must use 8-bit channels: {path}')
        if palette.ndim == 2:
            bgr = cv2.cvtColor(palette, cv2.COLOR_GRAY2BGR)
            visible = np.ones(palette.shape, dtype=bool)
        elif palette.ndim == 3 and palette.shape[2] in (3, 4):
            bgr = palette[:, :, :3]
            visible = palette[:, :, 3] > min_alpha if palette.shape[2] == 4 else np.ones(
                palette.shape[:2], dtype=bool
            )
        else:
            raise ValueError(f'Unsupported colour palette shape: {path}')
        if not np.any(visible):
            raise ValueError(f'Colour palette contains no visible pixels: {path}')
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[visible]
        lower_s = max(0, int(hsv[:, 1].min()) - saturation_margin)
        upper_s = min(255, int(hsv[:, 1].max()) + saturation_margin)
        lower_v = max(0, int(hsv[:, 2].min()) - value_margin)
        upper_v = min(255, int(hsv[:, 2].max()) + value_margin)
        bands = [(0, 179)] if not np.any(hsv[:, 1]) else _hue_bands(hsv[:, 0], hue_margin)
        for lower_h, upper_h in bands:
            ranges.append(ColorRange(
                name=path.stem,
                lower=(lower_h, lower_s, lower_v),
                upper=(upper_h, upper_s, upper_v),
            ))
    if not ranges:
        raise ValueError('At least one colour palette is required')
    return ranges


def _validate_image(bgr_image: np.ndarray) -> None:
    if (
        not isinstance(bgr_image, np.ndarray)
        or bgr_image.dtype != np.uint8
        or bgr_image.ndim != 3
        or bgr_image.shape[2] != 3
        or min(bgr_image.shape[:2]) < 1
    ):
        raise ValueError('Expected a nonempty uint8 BGR image with three channels')


def _bounded_image(bgr_image: np.ndarray, max_width: int) -> np.ndarray:
    """Validate and resize camera input consistently for calibration/detection."""
    _validate_image(bgr_image)
    height, width = bgr_image.shape[:2]
    if width <= max_width:
        return bgr_image
    processing_height = max(1, round(height * max_width / width))
    return cv2.resize(
        bgr_image, (max_width, processing_height), interpolation=cv2.INTER_AREA,
    )


def _color_bands(color_ranges: Sequence[ColorRange]):
    if not color_ranges:
        raise ValueError('At least one HSV colour range is required')
    bands = []
    for band in color_ranges:
        if len(band.lower) != 3 or len(band.upper) != 3:
            raise ValueError('HSV colour ranges must have three channels')
        for index, maximum in enumerate((179, 255, 255)):
            lower = _integer_in_range('HSV lower bound', band.lower[index], 0, maximum)
            upper = _integer_in_range('HSV upper bound', band.upper[index], 0, maximum)
            if lower > upper:
                raise ValueError('HSV lower bounds must not exceed upper bounds')
        bands.append((np.array(band.lower, np.uint8), np.array(band.upper, np.uint8)))
    return tuple(bands)


def _palette_mask(bgr_image: np.ndarray, bands) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
    for lower, upper in bands:
        cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper), dst=mask)
    return mask


def _quadrants(width: int, height: int):
    middle_x, middle_y = width // 2, height // 2
    return (
        (slice(0, middle_y), slice(0, middle_x)),
        (slice(0, middle_y), slice(middle_x, width)),
        (slice(middle_y, height), slice(0, middle_x)),
        (slice(middle_y, height), slice(middle_x, width)),
    )


class SmokeBlobDetector:
    """HSV range matching, morphology, and connected-component filtering."""

    def __init__(
        self,
        color_ranges: Sequence[ColorRange],
        min_blob_area: int = 64,
        min_blob_fraction: float = 0.001,
        morphology_kernel: int = 5,
        min_quadrant_fraction: float = 0.001,
        max_processing_width: int = 640,
    ):
        self.min_blob_area = _integer_in_range('min_blob_area', min_blob_area, 1, 2**31 - 1)
        self.morphology_kernel = _integer_in_range('morphology_kernel', morphology_kernel, 1, 255)
        if self.morphology_kernel % 2 != 1:
            raise ValueError('morphology_kernel must be odd (1 disables morphology)')
        self.max_processing_width = _integer_in_range(
            'max_processing_width', max_processing_width, 1, 2**31 - 1
        )
        for name, value in (
            ('min_blob_fraction', min_blob_fraction),
            ('min_quadrant_fraction', min_quadrant_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be finite and in [0, 1]')
        self.min_blob_fraction = float(min_blob_fraction)
        self.min_quadrant_fraction = float(min_quadrant_fraction)
        self.color_ranges = tuple(color_ranges)
        self._bands = _color_bands(self.color_ranges)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morphology_kernel, self.morphology_kernel)
        )

    def detect(
        self,
        bgr_image: np.ndarray,
        background_model: 'BackgroundColorModel | None' = None,
    ) -> DetectionResult:
        """Detect palette blobs after excluding learned environment colours.

        A supplied background model must finish calibration before detections
        are exposed. Morphology cannot restore a rejected environment colour.
        """
        image = _bounded_image(bgr_image, self.max_processing_width)
        if background_model is not None and not background_model.ready:
            return DetectionResult()
        height, width = image.shape[:2]
        eligible_mask = _palette_mask(image, self._bands)
        if background_model is not None:
            eligible_mask[background_model.exclusion_mask(image)] = 0
        mask = eligible_mask.copy()
        if self.morphology_kernel > 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            cv2.bitwise_and(mask, eligible_mask, dst=mask)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        minimum_area = max(self.min_blob_area, math.ceil(self.min_blob_fraction * width * height))
        accepted = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum_area) + 1
        if count <= 1 or not accepted.size:
            return DetectionResult()
        keep = np.zeros(count, dtype=bool)
        keep[accepted] = True
        accepted_mask = keep[labels]
        largest = int(accepted[np.argmax(stats[accepted, cv2.CC_STAT_AREA])])
        center_x, center_y = centers[largest]
        region = 1 + int(center_x >= width // 2) + 2 * int(center_y >= height // 2)
        coverage = []
        for rows, columns in _quadrants(width, height):
            quadrant = accepted_mask[rows, columns]
            coverage.append(
                float(np.count_nonzero(quadrant) / quadrant.size) if quadrant.size else 0.0
            )
        regions = tuple(
            index + 1 for index, fraction in enumerate(coverage)
            if (fraction > 0 and fraction >= self.min_quadrant_fraction) or index + 1 == region
        )
        return DetectionResult(
            detected=True,
            region=region,
            regions=regions,
            quadrant_coverage=tuple(coverage),
            coverage=float(np.count_nonzero(accepted_mask) / accepted_mask.size),
            center_x=float(center_x / (width - 1)) if width > 1 else 0.5,
            center_y=float(center_y / (height - 1)) if height > 1 else 0.5,
        )


def guidance(
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    assumed_distance_m: float = 0.0,
) -> tuple[float, float, float]:
    """Return yaw-left, elevation-up (radians), and height-up (metres).

    The camera is assumed level and looking along the drone's forward axis. The
    distance, if supplied, is forward depth along the optical axis, rather than
    slant range. Without a positive depth the height correction is ``NaN``.
    """
    if width < 1 or height < 1:
        raise ValueError('Image dimensions must be positive')
    for name, value in (('center_x', center_x), ('center_y', center_y)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f'{name} must be finite and in [0, 1]')
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError('Camera intrinsics must be finite with positive focal lengths')
    if not math.isfinite(assumed_distance_m) or assumed_distance_m < 0:
        raise ValueError('assumed_distance_m must be finite and nonnegative')
    horizontal_offset = center_x * (width - 1) - cx
    vertical_offset = center_y * (height - 1) - cy
    yaw = -math.atan2(horizontal_offset, fx)
    elevation = -math.atan2(vertical_offset, fy)
    altitude = assumed_distance_m * math.tan(elevation) if assumed_distance_m > 0 else math.nan
    return yaw, elevation, altitude


def draw_overlay(
    bgr_image: np.ndarray,
    result: DetectionResult,
    yaw_rad: float,
    vertical_rad: float,
    altitude_m: float,
    assumed_distance_m: float = 0.0,
) -> np.ndarray:
    """Copy a BGR camera frame and mark quadrant occupancy without boxes/points."""
    _validate_image(bgr_image)
    output = bgr_image.copy()
    height, width = output.shape[:2]
    for region, (rows, columns) in enumerate(_quadrants(width, height), start=1):
        if region in result.regions:
            patch = output[rows, columns]
            if patch.size:
                tint = np.empty_like(patch)
                tint[:] = (0, 165, 255) if region == result.region else (0, 220, 220)
                alpha = 0.27 if region == result.region else 0.14
                cv2.addWeighted(patch, 1.0 - alpha, tint, alpha, 0, dst=patch)
    middle_x, middle_y = width // 2, height // 2
    cv2.line(output, (middle_x, 0), (middle_x, height - 1), (220, 220, 220), 1)
    cv2.line(output, (0, middle_y), (width - 1, middle_y), (220, 220, 220), 1)
    font_scale = max(0.28, min(0.6, width / 1100))
    line_height = max(13, int(25 * font_scale / 0.6))
    occupied = ', '.join(REGION_NAMES[region] for region in result.regions)
    text_lines = [f'SMOKE / FOG: {occupied}' if result.detected else 'SMOKE / FOG: none']
    if result.detected:
        text_lines.append(
            f'Yaw {math.degrees(yaw_rad):+.1f} deg (left +)  '
            f'Vertical {math.degrees(vertical_rad):+.1f} deg (up +)'
        )
        text_lines.append(
            f'Altitude {altitude_m:+.2f} m (up +; assumed depth {assumed_distance_m:.1f} m)'
            if math.isfinite(altitude_m) and assumed_distance_m > 0
            else 'Altitude unavailable: target depth required'
        )
    strip_height = min(height, len(text_lines) * line_height + 7)
    output[:strip_height] = (output[:strip_height].astype(np.float32) * 0.35).astype(np.uint8)
    for index, line in enumerate(text_lines):
        cv2.putText(
            output, line, (5, (index + 1) * line_height), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    for region, (rows, columns) in enumerate(_quadrants(width, height), start=1):
        label_y = min(
            height - 3,
            rows.start + max(
                line_height, strip_height + line_height if rows.start == 0 else line_height
            ),
        )
        coverage = result.quadrant_coverage[region - 1]
        label = f'{REGION_NAMES[region]} {coverage:.1%}'
        cv2.putText(
            output, label, (columns.start + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 255, 255) if region in result.regions else (230, 230, 230),
            1, cv2.LINE_AA,
        )
    return output
