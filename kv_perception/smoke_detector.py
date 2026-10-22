"""ROS interface for the simulator's simple, colour based smoke detector.

This node only publishes observations.  The corrections assume a level,
forward facing camera and never command the vehicle.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from kv_perception.msg import SmokeDetection
from kv_perception.smoke_background import BackgroundColorModel, ConfirmationTracker
from kv_perception.smoke_detection import (
    DetectionResult,
    draw_overlay,
    guidance,
    load_color_ranges,
    SmokeBlobDetector,
)
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger


class SmokeDetectorNode(Node):
    """Detect texture-coloured blobs and publish quadrant and centering cues."""

    def __init__(self, **kwargs):
        super().__init__('smoke_detector', **kwargs)
        texture_directory = (
            Path(get_package_share_directory('forest_map_generator'))
            / 'models' / 'fog_generator' / 'materials' / 'textures'
        )
        defaults = {
            'image_topic': 'rgb_camera/image',
            'camera_info_topic': 'rgb_camera/camera_info',
            'texture_paths': [
                str(texture_directory / 'fogcolors.png'),
                str(texture_directory / 'smokecolors.png'),
            ],
            'min_texture_alpha': 16,
            'hue_margin': 6,
            'saturation_margin': 65,
            'value_margin': 45,
            'min_blob_area': 64,
            'min_blob_fraction': 0.001,
            'morphology_kernel': 5,
            'min_quadrant_fraction': 0.001,
            'max_processing_width': 640,
            'max_processing_rate': 5.0,
            'horizontal_fov': 1.204,
            'assumed_distance_m': 0.0,
            'background_calibration_frames': 30,
            'background_bin_size': 8,
            'background_min_pixel_fraction': 0.001,
            'background_min_frame_fraction': 0.6,
            'background_color_margin': 12,
            'confirmation_frames': 3,
            'confirmation_max_center_distance': 0.12,
            'confirmation_max_gap': 1.0,
        }
        parameters = {
            name: self.declare_parameter(name, default).value
            for name, default in defaults.items()
        }
        self._processing_rate = float(parameters['max_processing_rate'])
        self._horizontal_fov = float(parameters['horizontal_fov'])
        self._assumed_distance_m = float(parameters['assumed_distance_m'])
        if not math.isfinite(self._processing_rate) or self._processing_rate <= 0:
            raise ValueError('max_processing_rate must be finite and positive')
        if not 0 < self._horizontal_fov < math.pi:
            raise ValueError('horizontal_fov must be between 0 and pi radians')
        if not math.isfinite(self._assumed_distance_m) or self._assumed_distance_m < 0:
            raise ValueError('assumed_distance_m must be finite and nonnegative')
        colour_ranges = load_color_ranges(
            parameters['texture_paths'],
            min_alpha=parameters['min_texture_alpha'],
            hue_margin=parameters['hue_margin'],
            saturation_margin=parameters['saturation_margin'],
            value_margin=parameters['value_margin'],
        )
        self._detector = SmokeBlobDetector(
            colour_ranges,
            min_blob_area=parameters['min_blob_area'],
            min_blob_fraction=parameters['min_blob_fraction'],
            morphology_kernel=parameters['morphology_kernel'],
            min_quadrant_fraction=parameters['min_quadrant_fraction'],
            max_processing_width=parameters['max_processing_width'],
        )
        self._background = BackgroundColorModel(
            colour_ranges,
            calibration_frames=parameters['background_calibration_frames'],
            bin_size=parameters['background_bin_size'],
            min_pixel_fraction=parameters['background_min_pixel_fraction'],
            min_frame_fraction=parameters['background_min_frame_fraction'],
            color_margin=parameters['background_color_margin'],
            max_processing_width=parameters['max_processing_width'],
        )
        self._confirmation = ConfirmationTracker(
            required_frames=parameters['confirmation_frames'],
            max_center_distance=parameters['confirmation_max_center_distance'],
            max_gap=parameters['confirmation_max_gap'],
        )
        self._calibration_frames = parameters['background_calibration_frames']
        self._confirmation_frames = parameters['confirmation_frames']
        self._bridge = CvBridge()
        self._camera_info = None
        self._last_header = Header()
        self._last_image = None
        self._last_processed = -math.inf
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._detection_publisher = self.create_publisher(
            SmokeDetection, 'smoke/detection', output_qos
        )
        self._overlay_publisher = self.create_publisher(Image, 'smoke/overlay', output_qos)
        self._camera_info_subscription = self.create_subscription(
            CameraInfo, parameters['camera_info_topic'], self._on_camera_info, sensor_qos
        )
        self._image_subscription = self.create_subscription(
            Image, parameters['image_topic'], self._on_image, sensor_qos
        )
        self._recalibration_service = self.create_service(
            Trigger, 'smoke/recalibrate', self._on_recalibrate
        )
        self.get_logger().info(
            f'Learning smoke-free background from {self._calibration_frames} frames '
            f"in {parameters['image_topic']}; "
            'guidance starts after calibration and candidate confirmation'
        )

    def _on_camera_info(self, message: CameraInfo):
        self._camera_info = message

    def _intrinsics(self, width: int, height: int, frame_id: str):
        """Use the matching calibrated image dimensions, or the configured FOV."""
        info = self._camera_info
        if info is not None:
            same_frame = not (frame_id and info.header.frame_id) or (
                frame_id == info.header.frame_id
            )
            if (
                same_frame
                and info.width > 0
                and info.height > 0
                and len(info.k) == 9
                and all(math.isfinite(float(value)) for value in info.k)
                and info.k[0] > 0
                and info.k[4] > 0
            ):
                scale_x = width / info.width
                scale_y = height / info.height
                return (
                    float(info.k[0]) * scale_x,
                    float(info.k[4]) * scale_y,
                    float(info.k[2]) * scale_x,
                    float(info.k[5]) * scale_y,
                    True,
                )
        focal_length = (width / 2.0) / math.tan(self._horizontal_fov / 2.0)
        return focal_length, focal_length, (width - 1) / 2.0, (height - 1) / 2.0, False

    def _empty_message(self, header, status=None):
        message = SmokeDetection()
        message.header = header
        message.detected = False
        message.region = SmokeDetection.NONE
        message.regions = []
        message.quadrant_coverage = [0.0] * 4
        message.coverage = 0.0
        message.camera_info_used = False
        message.yaw_correction_rad = 0.0
        message.vertical_correction_rad = 0.0
        message.altitude_estimate_valid = False
        message.altitude_correction_m = math.nan
        message.assumed_distance_m = self._assumed_distance_m
        message.background_ready = bool(self._background.ready)
        message.calibration_progress = float(self._background.progress)
        message.confirmation_frames = 0
        message.status = status if status is not None else (
            SmokeDetection.SEARCHING if self._background.ready else SmokeDetection.CALIBRATING
        )
        return message

    def _on_recalibrate(self, request, response):
        """Discard the learned colours and immediately clear any guidance."""
        self._background.reset()
        self._confirmation.reset()
        self._last_processed = -math.inf
        message = self._empty_message(self._last_header)
        self._detection_publisher.publish(message)
        if self._last_image is not None:
            self._publish_overlay(
                self._last_header, self._last_image, DetectionResult(), message=message
            )
        response.success = True
        response.message = (
            f'Background calibration restarted: learning {self._calibration_frames} '
            'processed frames. Keep smoke out of the camera during calibration.'
        )
        return response

    def _on_image(self, image_message: Image):
        now = time.monotonic()
        if now - self._last_processed < 1.0 / self._processing_rate:
            return
        self._last_processed = now
        self._last_header = image_message.header
        message = self._empty_message(image_message.header)
        try:
            image = self._bridge.imgmsg_to_cv2(image_message, desired_encoding='bgr8')
        except Exception as error:
            self._confirmation.reset()
            self.get_logger().warning(
                f'Cannot decode RGB image: {error}', throttle_duration_sec=5.0
            )
            self._detection_publisher.publish(message)
            return
        try:
            self._last_image = image
            if not self._background.ready:
                self._background.observe(image)
                if self._background.ready:
                    self.get_logger().info(
                        'Background calibration complete: '
                        f'{self._background.excluded_bin_count} buffered color bins excluded; '
                        'awaiting stable smoke candidates'
                    )
                # Even the final learning frame is suppressed: it was used to
                # establish the background, rather than to test a candidate.
                message = self._empty_message(
                    image_message.header, status=SmokeDetection.CALIBRATING
                )
                self._detection_publisher.publish(message)
                self._publish_overlay(
                    image_message.header, image, DetectionResult(), message=message
                )
                return
            candidate = self._detector.detect(image, background_model=self._background)
            result = self._confirmation.update(candidate, now)
            message = self._empty_message(image_message.header)
            message.confirmation_frames = int(self._confirmation.count)
            if result.detected:
                message.status = SmokeDetection.CONFIRMED
            elif self._confirmation.count:
                message.status = SmokeDetection.CONFIRMING
            height, width = image.shape[:2]
            fx, fy, cx, cy, calibrated = self._intrinsics(
                width, height, image_message.header.frame_id
            )
            yaw, vertical, altitude = 0.0, 0.0, math.nan
            if result.detected:
                yaw, vertical, altitude = guidance(
                    result.center_x,
                    result.center_y,
                    width,
                    height,
                    fx,
                    fy,
                    cx,
                    cy,
                    assumed_distance_m=self._assumed_distance_m,
                )
            message.detected = bool(result.detected)
            message.region = int(result.region)
            message.regions = [int(region) for region in result.regions]
            message.quadrant_coverage = [float(value) for value in result.quadrant_coverage]
            message.coverage = float(result.coverage)
            message.camera_info_used = bool(result.detected and calibrated)
            message.yaw_correction_rad = float(yaw)
            message.vertical_correction_rad = float(vertical)
            message.altitude_estimate_valid = bool(
                result.detected and self._assumed_distance_m > 0 and math.isfinite(altitude)
            )
            message.altitude_correction_m = float(altitude)
        except Exception as error:
            self._confirmation.reset()
            self.get_logger().warning(
                f'Cannot process RGB image: {error}', throttle_duration_sec=5.0
            )
            message = self._empty_message(image_message.header)
            self._detection_publisher.publish(message)
            self._publish_overlay(
                image_message.header, image, DetectionResult(), message=message
            )
            return
        self._detection_publisher.publish(message)
        self._publish_overlay(
            image_message.header, image, result, yaw, vertical, altitude, message=message
        )

    def _publish_overlay(
        self, header, image, result, yaw=0.0, vertical=0.0, altitude=math.nan, message=None
    ):
        try:
            overlay = draw_overlay(
                image,
                result,
                yaw,
                vertical,
                altitude,
                assumed_distance_m=self._assumed_distance_m,
            )
            if message is not None:
                labels = {
                    SmokeDetection.CALIBRATING: (
                        f'CALIBRATING {self._background.frames_seen}/{self._calibration_frames}'
                    ),
                    SmokeDetection.SEARCHING: 'SEARCHING: background filtered',
                    SmokeDetection.CONFIRMING: (
                        f'CONFIRMING {message.confirmation_frames}/{self._confirmation_frames}'
                    ),
                    SmokeDetection.CONFIRMED: 'CONFIRMED SMOKE / FOG',
                }
                text = labels[message.status]
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_width = cv2.getTextSize(text, font, 1.0, 1)[0][0]
                scale = min(0.55, max(0.1, (overlay.shape[1] - 10) / text_width))
                line_height = max(14, cv2.getTextSize(text, font, scale, 1)[0][1] + 7)
                overlay[:line_height] = (25, 25, 25)
                cv2.putText(
                    overlay, text, (5, line_height - 4), font, scale,
                    (80, 255, 80) if result.detected else (255, 230, 120),
                    1, cv2.LINE_AA,
                )
            overlay_message = self._bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_message.header = header
            self._overlay_publisher.publish(overlay_message)
        except Exception as error:
            self.get_logger().warning(
                f'Cannot draw smoke overlay: {error}', throttle_duration_sec=5.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SmokeDetectorNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
