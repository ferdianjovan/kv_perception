"""Exercise image inputs and typed observations through an isolated ROS graph."""

import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
from cv_bridge import CvBridge
from kv_perception.msg import SmokeDetection
from kv_perception.smoke_detector import SmokeDetectorNode
import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

TEXTURES = (
    Path(__file__).resolve().parents[2]
    / 'forest_map_generator' / 'models' / 'fog_generator' / 'materials' / 'textures'
)


def wait_for(graph, predicate, description, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        graph.executor.spin_once(timeout_sec=0.02)
    pytest.fail(f'Timed out waiting for {description}')


@pytest.fixture
def graph(monkeypatch, tmp_path, request):
    monkeypatch.setenv('ROS_DOMAIN_ID', str(110 + os.getpid() % 50))
    monkeypatch.setenv('ROS_AUTOMATIC_DISCOVERY_RANGE', 'LOCALHOST')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    parameters = {
        'texture_paths': [
            str(TEXTURES / 'fogcolors.png'), str(TEXTURES / 'smokecolors.png'),
        ],
        'min_blob_area': 10,
        'min_blob_fraction': 0.0,
        'morphology_kernel': 1,
        'max_processing_rate': 100.0,
        'assumed_distance_m': 10.0,
        'background_calibration_frames': 1,
        'confirmation_frames': 1,
    }
    overrides = dict(getattr(request, 'param', {}))
    prime_background = overrides.pop('_prime_background', True)
    parameters.update(overrides)
    context = Context()
    rclpy.init(args=[], context=context)
    namespace = f'/smoke_detector_test_{os.getpid()}'
    detector = None
    driver = None
    executor = None
    try:
        detector = SmokeDetectorNode(
            context=context,
            namespace=namespace,
            parameter_overrides=[Parameter(name, value=value)
                                 for name, value in parameters.items()],
        )
        driver = Node('camera_test_driver', context=context, namespace=namespace)
        graph = SimpleNamespace(
            detector=detector, driver=driver, bridge=CvBridge(),
            detections=[], overlays=[],
        )
        graph.image_publisher = driver.create_publisher(
            Image, 'rgb_camera/image', qos_profile_sensor_data,
        )
        graph.camera_info_publisher = driver.create_publisher(
            CameraInfo, 'rgb_camera/camera_info', qos_profile_sensor_data,
        )
        graph.detection_subscription = driver.create_subscription(
            SmokeDetection, 'smoke/detection', graph.detections.append, 10,
        )
        graph.overlay_subscription = driver.create_subscription(
            Image, 'smoke/overlay', graph.overlays.append, 10,
        )
        executor = SingleThreadedExecutor(context=context)
        graph.executor = executor
        executor.add_node(detector)
        executor.add_node(driver)
        wait_for(
            graph,
            lambda: (
                graph.image_publisher.get_subscription_count() == 1
                and graph.camera_info_publisher.get_subscription_count() == 1
                and graph.detection_subscription.get_publisher_count() == 1
                and graph.overlay_subscription.get_publisher_count() == 1
            ),
            'camera and detector DDS discovery',
        )
        if prime_background:
            learning, _ = publish_frame(graph, clear_image(), -1)
            assert not learning.detected
            assert learning.status == SmokeDetection.CALIBRATING
            assert learning.background_ready
            graph.detections.clear()
            graph.overlays.clear()
        yield graph
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if detector is not None:
            detector.destroy_node()
        if driver is not None:
            driver.destroy_node()
        context.shutdown()


def calibrated_camera(graph, scale=1, frame_id='rgb_camera_optical_frame'):
    info = CameraInfo(
        header=Header(frame_id=frame_id), width=80 * scale, height=80 * scale,
    )
    info.k = [
        40.0 * scale, 0.0, 39.5 * scale,
        0.0, 40.0 * scale, 39.5 * scale,
        0.0, 0.0, 1.0,
    ]
    graph.camera_info_publisher.publish(info)
    wait_for(graph, lambda: graph.detector._camera_info is not None,
             'camera calibration delivery')


def image_with_smoke():
    image = clear_image()
    texture = cv2.imread(str(TEXTURES / 'fogcolors.png'), cv2.IMREAD_UNCHANGED)
    opaque = texture[:, :, :3][texture[:, :, 3] > 16]
    image[10:30, 10:30] = opaque[len(opaque) // 2]
    return image


def clear_image():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (0, 150, 0)
    return image


def smoke_free_sky():
    """Use a genuine fog-palette colour as the confusing background."""
    image = np.empty((80, 80, 3), dtype=np.uint8)
    image[:] = image_with_smoke()[15, 15]
    return image


def sky_with_distinct_smoke():
    image = smoke_free_sky()
    sky_colour = image[0, 0].astype(np.int16)
    palette = cv2.imread(str(TEXTURES / 'smokecolors.png'), cv2.IMREAD_UNCHANGED)
    visible = palette[:, :, :3][palette[:, :, 3] > 16]
    separation = np.max(np.abs(visible.astype(np.int16) - sky_colour), axis=1)
    colour = visible[np.argmax(separation)]
    assert np.max(np.abs(colour.astype(np.int16) - sky_colour)) > 40
    image[10:30, 10:30] = colour
    return image


def publish_frame(graph, image, stamp, encoding='bgr8'):
    # Respect the detector's configured 100 Hz processing throttle.
    time.sleep(0.02)
    message = graph.bridge.cv2_to_imgmsg(image, encoding=encoding)
    message.header.frame_id = 'rgb_camera_optical_frame'
    message.header.stamp.sec = stamp
    message.header.stamp.nanosec = 123456789
    graph.image_publisher.publish(message)
    wait_for(
        graph,
        lambda: (
            any(item.header == message.header for item in graph.detections)
            and any(item.header == message.header for item in graph.overlays)
        ),
        f'detection and overlay for frame {stamp}',
    )
    detection = next(item for item in graph.detections if item.header == message.header)
    overlay = next(item for item in graph.overlays if item.header == message.header)
    return detection, overlay


@pytest.mark.parametrize('calibration_scale', [1, 2])
def test_rgb_frame_publishes_typed_region_calibrated_guidance_and_overlay(
    graph, calibration_scale,
):
    calibrated_camera(graph, scale=calibration_scale)
    bgr_image = image_with_smoke()
    # A real RGB camera publishes rgb8. Conversion must preserve palette colours.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    detection, overlay = publish_frame(graph, rgb_image, 1, encoding='rgb8')
    assert isinstance(detection, SmokeDetection)
    assert detection.detected
    assert detection.status == SmokeDetection.CONFIRMED
    assert detection.background_ready
    assert detection.calibration_progress == 1.0
    assert detection.confirmation_frames == 1
    assert detection.region == SmokeDetection.TOP_LEFT
    assert list(detection.regions) == [SmokeDetection.TOP_LEFT]
    assert list(detection.quadrant_coverage) == pytest.approx([0.25, 0.0, 0.0, 0.0])
    assert detection.coverage == pytest.approx(400 / 6400)
    assert detection.camera_info_used
    assert detection.yaw_correction_rad == pytest.approx(math.atan(0.5))
    assert detection.vertical_correction_rad == pytest.approx(math.atan(0.5))
    assert detection.altitude_estimate_valid
    assert detection.assumed_distance_m == 10.0
    assert detection.altitude_correction_m == pytest.approx(5.0)
    assert overlay.encoding == 'bgr8'
    overlay_image = graph.bridge.imgmsg_to_cv2(overlay, desired_encoding='bgr8')
    assert overlay_image.shape == bgr_image.shape
    assert not np.array_equal(overlay_image[35, 35], bgr_image[35, 35])

    # Following a valid detection, a clear frame must explicitly erase the cue.
    clear_image = np.zeros_like(bgr_image)
    clear_image[:] = (0, 150, 0)
    cleared, cleared_overlay = publish_frame(graph, clear_image, 2)
    assert not cleared.detected
    assert cleared.status == SmokeDetection.SEARCHING
    assert cleared.confirmation_frames == 0
    assert cleared.region == SmokeDetection.NONE
    assert not cleared.regions
    assert list(cleared.quadrant_coverage) == [0.0] * 4
    assert cleared.coverage == 0.0
    assert cleared.yaw_correction_rad == 0.0
    assert cleared.vertical_correction_rad == 0.0
    assert not cleared.altitude_estimate_valid
    assert math.isnan(cleared.altitude_correction_m)
    assert cleared_overlay.header == cleared.header


@pytest.mark.parametrize('graph', [{'assumed_distance_m': 0.0}], indirect=True)
def test_without_calibration_or_range_outputs_fov_angles_and_unknown_metres(graph):
    detection, _ = publish_frame(graph, image_with_smoke(), 1)
    focal_length = 40.0 / math.tan(1.204 / 2.0)
    assert detection.detected
    assert not detection.camera_info_used
    assert detection.yaw_correction_rad == pytest.approx(math.atan(20.0 / focal_length))
    assert detection.vertical_correction_rad == pytest.approx(math.atan(20.0 / focal_length))
    assert not detection.altitude_estimate_valid
    assert detection.assumed_distance_m == 0.0
    assert math.isnan(detection.altitude_correction_m)


def test_other_camera_calibration_is_ignored(graph):
    calibrated_camera(graph, frame_id='unrelated_depth_camera_frame')
    detection, _ = publish_frame(graph, image_with_smoke(), 1)
    assert detection.detected
    assert not detection.camera_info_used
    focal_length = 40.0 / math.tan(1.204 / 2.0)
    assert detection.yaw_correction_rad == pytest.approx(math.atan(20.0 / focal_length))


def assert_no_guidance(message):
    assert not message.detected
    assert message.region == SmokeDetection.NONE
    assert not message.regions
    assert list(message.quadrant_coverage) == [0.0] * 4
    assert message.coverage == 0.0
    assert message.yaw_correction_rad == 0.0
    assert message.vertical_correction_rad == 0.0
    assert not message.altitude_estimate_valid
    assert math.isnan(message.altitude_correction_m)


@pytest.mark.parametrize('graph', [{
    '_prime_background': False,
    'background_calibration_frames': 3,
    'confirmation_frames': 3,
}], indirect=True)
def test_sky_is_learned_before_detection_and_novel_smoke_needs_confirmation(graph):
    sky = smoke_free_sky()
    for index in range(3):
        learning, overlay = publish_frame(graph, sky, index + 1)
        assert_no_guidance(learning)
        assert learning.status == SmokeDetection.CALIBRATING
        assert learning.calibration_progress == pytest.approx((index + 1) / 3)
        assert learning.background_ready == (index == 2)
        assert learning.confirmation_frames == 0
        assert overlay.header == learning.header

    # Rendering can slightly change the sky colour after takeoff. The padding
    # around learned colours must still exclude it, rather than create a blob.
    buffered_sky = np.clip(sky.astype(np.int16) - 5, 0, 255).astype(np.uint8)
    filtered, _ = publish_frame(graph, buffered_sky, 4)
    assert_no_guidance(filtered)
    assert filtered.status == SmokeDetection.SEARCHING
    assert graph.detector._background.excluded_bin_count > 0
    calibrated_count = graph.detector._background.frames_seen

    candidate = sky_with_distinct_smoke()
    for index in range(3):
        detection, _ = publish_frame(graph, candidate, 5 + index)
        assert detection.confirmation_frames == index + 1
        if index < 2:
            assert_no_guidance(detection)
            assert detection.status == SmokeDetection.CONFIRMING
        else:
            assert detection.detected
            assert detection.status == SmokeDetection.CONFIRMED
            assert detection.region == SmokeDetection.TOP_LEFT
            assert detection.yaw_correction_rad > 0
            assert detection.vertical_correction_rad > 0
    # Candidate smoke is never folded into the frozen environment baseline.
    assert graph.detector._background.frames_seen == calibrated_count


@pytest.mark.parametrize('graph', [{'confirmation_frames': 3}], indirect=True)
def test_missing_candidate_resets_confirmation_and_recalibration_clears_guidance(graph):
    candidate = image_with_smoke()
    pending, _ = publish_frame(graph, candidate, 1)
    assert_no_guidance(pending)
    assert pending.confirmation_frames == 1
    missing, _ = publish_frame(graph, clear_image(), 2)
    assert_no_guidance(missing)
    assert missing.status == SmokeDetection.SEARCHING
    assert missing.confirmation_frames == 0
    for index in range(3):
        detection, _ = publish_frame(graph, candidate, 3 + index)
        assert detection.confirmation_frames == index + 1
        assert detection.detected == (index == 2)

    service = graph.driver.create_client(Trigger, 'smoke/recalibrate')
    wait_for(graph, service.service_is_ready, 'recalibration service')
    previous_count = len(graph.detections)
    future = service.call_async(Trigger.Request())
    wait_for(
        graph, lambda: future.done() and len(graph.detections) > previous_count,
        'immediate clear observation from recalibration',
    )
    assert future.result().success
    assert 'calibration restarted' in future.result().message
    cleared = graph.detections[-1]
    assert_no_guidance(cleared)
    assert cleared.header == detection.header
    assert cleared.status == SmokeDetection.CALIBRATING
    assert not cleared.background_ready
    assert cleared.calibration_progress == 0.0
    assert cleared.confirmation_frames == 0
    # The next frame is a new background sample, even if it contains the
    # formerly confirmed blob, and cannot immediately reinstate old guidance.
    learning, _ = publish_frame(graph, candidate, 6)
    assert_no_guidance(learning)
    assert learning.status == SmokeDetection.CALIBRATING
    ignored, _ = publish_frame(graph, candidate, 7)
    assert_no_guidance(ignored)
    assert ignored.status == SmokeDetection.SEARCHING


@pytest.mark.parametrize('graph', [{'confirmation_frames': 2}], indirect=True)
@pytest.mark.parametrize('failure', ['decode', 'processing'])
def test_bad_frame_resets_pending_confirmation(graph, monkeypatch, failure):
    pending, _ = publish_frame(graph, image_with_smoke(), 1)
    assert pending.status == SmokeDetection.CONFIRMING
    if failure == 'decode':
        invalid = Image(
            header=Header(frame_id='rgb_camera_optical_frame'),
            width=80, height=80, encoding='unsupported_encoding', step=240,
        )
        invalid.header.stamp.sec = 2
        time.sleep(0.02)
        graph.image_publisher.publish(invalid)
        wait_for(
            graph, lambda: any(item.header == invalid.header for item in graph.detections),
            'clear observation after invalid image',
        )
        failed = next(item for item in graph.detections if item.header == invalid.header)
    else:
        def fail_processing(*args, **kwargs):
            raise ValueError('synthetic processing failure')

        with monkeypatch.context() as patch:
            patch.setattr(graph.detector._detector, 'detect', fail_processing)
            failed, _ = publish_frame(graph, image_with_smoke(), 2)
    assert_no_guidance(failed)
    assert failed.confirmation_frames == 0
    resumed, _ = publish_frame(graph, image_with_smoke(), 3)
    assert_no_guidance(resumed)
    assert resumed.status == SmokeDetection.CONFIRMING
    assert resumed.confirmation_frames == 1
