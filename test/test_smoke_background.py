"""Verify learned environment rejection and conservative candidate confirmation."""

from dataclasses import replace

from kv_perception.smoke_background import BackgroundColorModel, ConfirmationTracker
from kv_perception.smoke_detection import ColorRange, DetectionResult, SmokeBlobDetector
import numpy as np
import pytest


RANGES = [
    ColorRange('fog', (0, 0, 54), (179, 65, 255)),
    ColorRange('smoke', (1, 117, 23), (34, 255, 255)),
]
GREEN = (0, 150, 0)
SKY = (190, 190, 190)
FOG = (100, 100, 100)
BROWN = (12, 80, 160)


def scene(color=GREEN):
    return np.full((80, 80, 3), color, dtype=np.uint8)


def model(**options):
    parameters = {'calibration_frames': 3, 'max_processing_width': 80}
    parameters.update(options)
    return BackgroundColorModel(RANGES, **parameters)


def detector(**options):
    parameters = {
        'min_blob_area': 10, 'min_blob_fraction': 0.0,
        'morphology_kernel': 1, 'max_processing_width': 80,
    }
    parameters.update(options)
    return SmokeBlobDetector(RANGES, **parameters)


def learn(background, image, frames=3):
    for _ in range(frames):
        background.observe(image)
    assert background.ready
    assert background.progress == 1.0


def test_calibration_blocks_detections_even_when_palette_pixels_are_present():
    background = model()
    image = scene(SKY)
    assert detector().detect(image).detected
    for index in range(2):
        background.observe(image)
        assert background.frames_seen == index + 1
        assert background.progress == pytest.approx((index + 1) / 3)
        assert not background.ready
        assert not detector().detect(image, background_model=background).detected


def test_sky_gradient_is_rejected_when_it_moves_to_other_quadrants():
    background = model()
    image = scene()
    for index, value in enumerate((176, 184, 192, 200)):
        image[:40, index * 20:(index + 1) * 20] = value
    learn(background, image)
    shifted = np.roll(image, 40, axis=0)
    assert detector().detect(shifted).detected
    assert not detector().detect(shifted, background_model=background).detected


def test_buffer_rejects_small_sky_color_changes_but_retains_novel_smoke():
    background = model(color_margin=12)
    learn(background, scene(SKY))
    image = scene((198, 187, 181))
    assert not detector().detect(image, background_model=background).detected
    image[10:30, 10:30] = FOG
    result = detector().detect(image, background_model=background)
    assert result.detected
    assert result.region == 1
    assert result.regions == (1,)
    assert result.coverage == pytest.approx(400 / 6400)


@pytest.mark.parametrize('channel', [0, 1, 2])
def test_color_buffer_expands_along_every_bgr_axis(channel):
    background = model(color_margin=8)
    learn(background, scene((160, 160, 160)))
    adjacent = [160, 160, 160]
    adjacent[channel] = 168
    distant = [160, 160, 160]
    distant[channel] = 184
    assert background.exclusion_mask(scene(adjacent)).all()
    assert not background.exclusion_mask(scene(distant)).any()


def test_outside_palette_pixels_never_enter_the_environment_model():
    background = model()
    learn(background, scene())
    assert background.excluded_bin_count == 0
    assert not background.exclusion_mask(scene()).any()
    assert detector().detect(scene(FOG), background_model=background).detected


def test_transient_color_during_calibration_is_not_typical():
    background = model(calibration_frames=5)
    image = scene()
    image[:40] = SKY
    for index in range(5):
        sample = image.copy()
        if index == 0:
            sample[50:70, 10:30] = BROWN
        background.observe(sample)
    assert background.ready
    assert background.exclusion_mask(scene(SKY)).all()
    assert not background.exclusion_mask(scene(BROWN)).any()


def test_tiny_recurring_colors_are_not_typical_environment_colors():
    background = model(min_pixel_fraction=0.01)
    image = scene()
    image[:40] = SKY
    image[50:52, 10:12] = BROWN
    learn(background, image)
    assert not background.exclusion_mask(scene(BROWN)).any()


def test_frozen_background_does_not_learn_an_approaching_smoke_blob():
    background = model()
    learn(background, scene(SKY))
    for _ in range(10):
        background.observe(scene(FOG))
    assert background.frames_seen == 3
    assert detector().detect(scene(FOG), background_model=background).detected
    background.reset()
    assert not background.ready
    assert background.progress == 0.0
    assert background.frames_seen == 0
    assert not detector().detect(scene(FOG), background_model=background).detected


def test_closing_cannot_rejoin_smoke_blobs_through_excluded_sky():
    background = model(color_margin=0)
    learn(background, scene(SKY))
    image = scene()
    image[20:50, 10:40] = FOG
    image[20:50, 44:74] = FOG
    image[20:50, 40:44] = SKY
    result = detector(morphology_kernel=7).detect(image, background_model=background)
    assert result.detected
    # Two separate, equal blobs remain. Their merged centroid would be at x=41.5.
    assert result.center_x == pytest.approx(24.5 / 79)


def candidate(x=0.25, y=0.25, region=1):
    return DetectionResult(
        detected=True, region=region, regions=(region,),
        center_x=x, center_y=y, coverage=0.05,
    )


def test_only_consecutive_consistent_candidates_produce_guidance():
    tracker = ConfirmationTracker(required_frames=3)
    result = candidate()
    assert not tracker.update(result, 1.0).detected
    assert tracker.count == 1
    assert not tracker.update(result, 1.2).detected
    assert tracker.update(result, 1.4) == result
    assert tracker.count == 3
    assert tracker.update(result, 1.6) == result
    assert tracker.count == 3
    assert not tracker.update(DetectionResult(), 1.8).detected
    assert tracker.count == 0
    assert not tracker.update(result, 2.0).detected


@pytest.mark.parametrize('next_time,next_candidate', [
    (3.0, candidate()),  # Excessive frame gap.
    (0.5, candidate()),  # Clock moved backward.
    (1.4, candidate(x=0.8, y=0.8, region=4)),  # Different target.
])
def test_confirmation_restarts_after_gap_clock_jump_or_target_jump(
    next_time, next_candidate,
):
    tracker = ConfirmationTracker(required_frames=3, max_gap=1.0)
    tracker.update(candidate(), 1.0)
    tracker.update(candidate(), 1.2)
    assert not tracker.update(next_candidate, next_time).detected
    assert tracker.count == 1


def test_small_target_motion_across_quadrant_boundary_can_be_confirmed():
    tracker = ConfirmationTracker(required_frames=3)
    result = candidate(x=0.49)
    assert not tracker.update(result, 1.0).detected
    result = replace(result, center_x=0.51, region=2, regions=(2,))
    assert not tracker.update(result, 1.2).detected
    assert tracker.update(result, 1.4) == result
    tracker.reset()
    assert tracker.count == 0
    assert not tracker.update(result, 1.6).detected
