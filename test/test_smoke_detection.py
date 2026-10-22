"""Verify palette matching, region selection, and camera guidance geometry."""

import math
from pathlib import Path

import cv2
from kv_perception.smoke_detection import (
    draw_overlay,
    guidance,
    load_color_ranges,
    SmokeBlobDetector,
)
import numpy as np
import pytest


TEXTURES = (
    Path(__file__).resolve().parents[2]
    / 'forest_map_generator' / 'models' / 'fog_generator' / 'materials' / 'textures'
)
GREEN = (0, 150, 0)


def scene(height=80, width=80):
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = GREEN
    return image


def hsv_is_matched(hsv, ranges):
    sample = np.asarray([[hsv]], dtype=np.uint8)
    return any(
        cv2.inRange(
            sample, np.asarray(color.lower, dtype=np.uint8),
            np.asarray(color.upper, dtype=np.uint8),
        )[0, 0] != 0
        for color in ranges
    )


def write_palette(path, hsv_colors, alpha):
    hsv = np.asarray([hsv_colors], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgra = np.dstack((bgr, np.asarray([alpha], dtype=np.uint8)))
    assert cv2.imwrite(str(path), bgra)


@pytest.fixture
def ranges():
    return load_color_ranges([
        TEXTURES / 'fogcolors.png', TEXTURES / 'smokecolors.png',
    ])


@pytest.fixture
def detector(ranges):
    return SmokeBlobDetector(
        ranges,
        min_blob_area=10,
        min_blob_fraction=0.0,
        morphology_kernel=1,
        min_quadrant_fraction=0.001,
    )


def texture_sample(name):
    image = cv2.imread(str(TEXTURES / name), cv2.IMREAD_UNCHANGED)
    assert image is not None
    opaque = image[:, :, :3][image[:, :, 3] >= 16]
    # Use a middle opaque texel rather than a hand-picked approximation.
    return opaque[len(opaque) // 2]


@pytest.mark.parametrize('name', ['fogcolors.png', 'smokecolors.png'])
def test_supplied_texture_colors_are_covered_by_derived_ranges(name):
    texture = cv2.imread(str(TEXTURES / name), cv2.IMREAD_UNCHANGED)
    assert texture is not None
    hsv = cv2.cvtColor(texture[:, :, :3], cv2.COLOR_BGR2HSV)
    opaque_hsv = hsv[texture[:, :, 3] >= 16]
    colors = load_color_ranges([TEXTURES / name])
    assert len(opaque_hsv) > 0
    assert all(hsv_is_matched(color, colors) for color in opaque_hsv)


def test_transparent_palette_colors_do_not_expand_the_match(tmp_path):
    palette = tmp_path / 'transparent.png'
    write_palette(palette, [(0, 0, 140), (60, 255, 220)], [255, 0])
    colors = load_color_ranges(
        [palette], hue_margin=0, saturation_margin=0, value_margin=0,
    )
    assert hsv_is_matched((0, 0, 140), colors)
    assert not hsv_is_matched((60, 255, 220), colors)


def test_palette_hue_range_wraps_across_red_without_including_green(tmp_path):
    palette = tmp_path / 'red_wrap.png'
    write_palette(palette, [(179, 230, 210), (1, 230, 210)], [255, 255])
    colors = load_color_ranges([palette], hue_margin=6)
    assert hsv_is_matched((0, 230, 210), colors)
    assert hsv_is_matched((177, 230, 210), colors)
    assert hsv_is_matched((3, 230, 210), colors)
    assert not hsv_is_matched((90, 230, 210), colors)


@pytest.mark.parametrize('name', ['fogcolors.png', 'smokecolors.png'])
@pytest.mark.parametrize('region, x, y', [
    (1, 10, 10), (2, 50, 10), (3, 10, 50), (4, 50, 50),
])
def test_opaque_texture_blob_reports_its_quadrant(detector, name, region, x, y):
    image = scene()
    image[y:y + 20, x:x + 20] = texture_sample(name)
    result = detector.detect(image)
    assert result.detected
    assert result.region == region
    assert tuple(result.regions) == (region,)
    assert result.coverage == pytest.approx(400 / 6400)
    assert result.quadrant_coverage[region - 1] == pytest.approx(400 / 1600)
    assert result.center_x == pytest.approx((x + 9.5) / 79)
    assert result.center_y == pytest.approx((y + 9.5) / 79)


def test_green_frame_has_no_detection(detector):
    result = detector.detect(scene())
    assert not result.detected
    assert result.region == 0
    assert not result.regions
    assert result.coverage == 0.0
    assert tuple(result.quadrant_coverage) == (0.0, 0.0, 0.0, 0.0)


def test_disconnected_subthreshold_spots_are_not_accumulated_as_one_blob(detector):
    image = scene()
    image[5:75:5, 5:75:5] = texture_sample('fogcolors.png')
    assert np.count_nonzero(np.all(image != GREEN, axis=2)) > 10
    result = detector.detect(image)
    assert not result.detected
    assert result.coverage == 0.0


def test_blob_spanning_image_center_reports_all_four_quadrants(detector):
    image = scene()
    image[30:50, 30:50] = texture_sample('fogcolors.png')
    result = detector.detect(image)
    assert result.detected
    assert set(result.regions) == {1, 2, 3, 4}
    assert tuple(result.quadrant_coverage) == pytest.approx((100 / 1600,) * 4)
    assert result.center_x == pytest.approx(0.5)
    assert result.center_y == pytest.approx(0.5)


def test_largest_blob_drives_guidance_but_other_regions_remain_visible(detector):
    image = scene()
    image[10:20, 10:20] = texture_sample('fogcolors.png')
    image[50:70, 50:70] = texture_sample('smokecolors.png')
    result = detector.detect(image)
    assert result.region == 4
    assert set(result.regions) == {1, 4}
    assert result.center_x == pytest.approx(59.5 / 79)
    assert result.center_y == pytest.approx(59.5 / 79)
    assert result.coverage == pytest.approx(500 / 6400)


def test_primary_region_is_kept_when_additional_quadrant_threshold_is_high(ranges):
    detector = SmokeBlobDetector(
        ranges, min_blob_area=10, min_blob_fraction=0.0,
        morphology_kernel=1, min_quadrant_fraction=0.9,
    )
    image = scene()
    image[10:30, 10:30] = texture_sample('fogcolors.png')
    result = detector.detect(image)
    assert result.detected
    assert result.region == 1
    assert tuple(result.regions) == (1,)


def test_relative_blob_area_rejects_small_component_on_large_image(ranges):
    detector = SmokeBlobDetector(
        ranges, min_blob_area=10, min_blob_fraction=0.1,
        morphology_kernel=1,
    )
    image = scene()
    image[10:30, 10:30] = texture_sample('fogcolors.png')
    assert not detector.detect(image).detected


def test_overlay_tints_detected_region_and_does_not_mutate_input(detector):
    image = scene()
    image[10:30, 10:30] = texture_sample('fogcolors.png')
    original = image.copy()
    result = detector.detect(image)
    overlay = draw_overlay(image, result, 0.1, 0.1, float('nan'))
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(overlay[35, 35], original[35, 35])


@pytest.mark.parametrize('center_x, center_y, yaw_sign, vertical_sign', [
    (0.25, 0.25, 1, 1),
    (0.75, 0.25, -1, 1),
    (0.25, 0.75, 1, -1),
    (0.75, 0.75, -1, -1),
])
def test_guidance_uses_left_positive_yaw_and_up_positive_vertical(
    center_x, center_y, yaw_sign, vertical_sign,
):
    yaw, vertical, altitude = guidance(
        center_x, center_y, 81, 81, 40.0, 40.0, 40.0, 40.0,
        assumed_distance_m=10.0,
    )
    assert yaw == pytest.approx(yaw_sign * math.atan(0.5))
    assert vertical == pytest.approx(vertical_sign * math.atan(0.5))
    assert altitude == pytest.approx(vertical_sign * 5.0)


def test_centered_target_requires_no_rotation_or_altitude_change():
    assert guidance(
        0.5, 0.5, 81, 81, 40.0, 40.0, 40.0, 40.0,
        assumed_distance_m=10.0,
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_metric_altitude_is_unknown_without_distance_assumption():
    yaw, vertical, altitude = guidance(
        0.25, 0.25, 81, 81, 40.0, 40.0, 40.0, 40.0,
    )
    assert yaw > 0
    assert vertical > 0
    assert math.isnan(altitude)


def test_guidance_respects_calibrated_offcenter_principal_point():
    yaw, vertical, altitude = guidance(
        0.5, 0.5, 81, 81, 40.0, 80.0, 30.0, 60.0,
        assumed_distance_m=12.0,
    )
    assert yaw == pytest.approx(-math.atan(10 / 40))
    assert vertical == pytest.approx(math.atan(20 / 80))
    assert altitude == pytest.approx(3.0)
