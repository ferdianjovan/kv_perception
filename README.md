# KV Perception

A simple RGB smoke/fog detector with startup environment calibration, HSV
color-range matching, connected blobs, and consecutive-frame confirmation.
It reports image quadrants and centering guidance for a later mission controller.

Build and launch alongside the existing simulator:

```bash
colcon build --packages-up-to kv_perception --symlink-install
source install/setup.bash
ros2 launch kv_perception smoke_detector.launch.py \
  drone_name:=kv_drone assumed_distance_m:=10.0
```

Use the same `drone_name` as `kv_world.launch.py`. The launch default is
`x500_depth`; the example above follows the `kv_drone` simulation setup.
For a real-time camera, add `use_sim_time:=false`. Camera topic launch arguments
can be relative to the drone namespace or absolute.

| Topic (with `drone_name:=kv_drone`) | Type | Purpose |
| --- | --- | --- |
| `/kv_drone/rgb_camera/image` | `sensor_msgs/Image` | RGB input |
| `/kv_drone/rgb_camera/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration input |
| `/kv_drone/smoke/overlay` | `sensor_msgs/Image` (`bgr8`) | RGB view with quadrant grid and detection tints |
| `/kv_drone/smoke/detection` | `kv_perception/SmokeDetection` | Regions, coverage, and centering guidance |

View the overlay with RViz's Image display or `rqt_image_view`. Inspect the
observations with:

```bash
ros2 topic echo /kv_drone/smoke/detection
```

At startup, show the camera a representative **smoke-free environment, including
the sky**. The first 30 processed frames (about six seconds at the default 5 Hz)
are used only for background calibration. No detections or guidance are emitted
during this phase; the overlay displays `CALIBRATING` and the frame count.

Only pixels already inside the fog/smoke palette ranges participate in learning.
Colors outside those ranges are ignored. The baseline retains multiple recurring
BGR color bins, preserving gradients and distinct sky/scenery colors instead of
using one average. By default, a color bin must cover at least 0.1% of an image
in at least 60% of calibration frames to count as typical environment color.

Learned colors are excluded everywhere in later images, with a 12-unit buffer
per 8-bit BGR channel, rounded up to the configured color-bin width. Colors are
removed before morphology and again afterward so closing cannot fill excluded
sky back into a blob. The baseline stays frozen during detection; approaching
smoke will not gradually be learned away.

After calibration, a candidate must persist for three spatially consistent
processed frames before `detected=true` and guidance become available. Missing
candidates, processing errors, a large centroid jump, or a gap above one second
restart confirmation. Small motion across a quadrant boundary is allowed. The
overlay shows `SEARCHING`, `CONFIRMING`, or `CONFIRMED` as appropriate. This is a
conservative heuristic, not a calibrated probability that smoke is present.

To relearn after lighting or surroundings change, first show a clear environment
and request recalibration:

```bash
ros2 service call /kv_drone/smoke/recalibrate std_srvs/srv/Trigger '{}'
```

Recalibration immediately clears existing guidance and restarts learning.
Smoke already visible during calibration may be learned as background. Smoke
whose color matches the buffered sky colors will also be suppressed; this
color-only method cannot reliably distinguish those cases.

Quadrant IDs are `NONE=0`, `TOP_LEFT=1`, `TOP_RIGHT=2`, `BOTTOM_LEFT=3`, and
`BOTTOM_RIGHT=4`. `regions` lists all occupied quadrants. `region` selects the
largest accepted blob's quadrant, and guidance refers to that blob. The overlay
tints occupied quadrants, emphasizes the selected quadrant, and displays the
corrections. It does not draw a precise smoke location or bounding box.

The message retains the source image timestamp and frame. `status` identifies
`CALIBRATING=0`, `SEARCHING=1`, `CONFIRMING=2`, or `CONFIRMED=3`.
`background_ready`, `calibration_progress`, and `confirmation_frames` expose
progress. Only confirmed detections provide regions or guidance; all other
processed frames publish `detected=false`, empty regions, zero angles, and an
unavailable altitude estimate. Consumers should also reject old observations
using the image timestamp if the camera stops publishing.

`yaw_correction_rad` is positive for a left/CCW turn and negative for a right/CW
turn. `vertical_correction_rad` is positive when the smoke is above image center.
`altitude_correction_m` is positive to climb and negative to descend. These signs
assume the simulator's forward-facing, level RGB camera; vehicle attitude and
other camera mounting rotations are not compensated.

RGB alone cannot measure smoke distance. The standard launch assumes a **10 m
forward distance** from the camera to the smoke plane, published explicitly as
`assumed_distance_m`. This is not a measured range or the slant distance to a blob.
Set `assumed_distance_m:=0.0` to publish angular guidance only; the metric field
will be NaN and `altitude_estimate_valid=false`. That validity flag indicates
that an estimate is available under the stated assumption, not that range has
been measured. Running the node directly without the launch/config defaults to
angles only.

For internal blob centroid `(u, v)` and camera intrinsics `(fx, fy, cx, cy)`:

```text
yaw correction      = -atan2(u - cx, fx)
vertical correction = -atan2(v - cy, fy)
altitude correction = assumed_distance_m * tan(vertical correction)
```

CameraInfo is preferred and scaled when image dimensions differ.
`camera_info_used` indicates whether calibration was usable. The fallback uses
square pixels and the OakD RGB model's horizontal FOV of 1.204 radians. The
centroid is internal and is not published as an image position.

Color bounds are derived at startup from visible pixels in the installed
`forest_map_generator/models/fog_generator/materials/textures/fogcolors.png`
and `smokecolors.png`. Samples with alpha at most 16 are excluded. The former
contains gray fog; the latter contains orange/brown smoke. With the default HSV
margins, the OpenCV ranges are:

| Palette | Lower HSV | Upper HSV |
| --- | --- | --- |
| Fog | `(0, 0, 54)` | `(179, 65, 255)` |
| Smoke | `(1, 117, 23)` | `(34, 255, 255)` |

OpenCV hue uses 0–179; saturation and value use 0–255. Padding accommodates some
lighting, the emitter's diffuse multiplier, and particle transparency. The learned
environment filter then removes typical matching colors. Unseen scenery can
still trigger this color-only heuristic. Coverage fields are pixel-area fractions,
not confidence scores.

Tune [config/smoke_detector.yaml](config/smoke_detector.yaml) or supply
`config:=/absolute/path/to/config.yaml`. HSV margins, minimum blob area/fraction,
morphology, and quadrant occupancy thresholds are configurable. Blob area is
measured after resizing to a maximum width of 640 pixels; processing is limited
to 5 Hz by default. The overlay keeps the original image resolution. The launch
arguments for topics, range, and simulation time override the parameter file.

Background and confirmation settings are in the same parameter file:

| Parameter | Default | Effect |
| --- | --- | --- |
| `background_calibration_frames` | `30` | Valid processed frames needed before detection |
| `background_bin_size` | `8` | BGR channel units per color bin |
| `background_min_pixel_fraction` | `0.001` | Per-frame area required for a typical color bin |
| `background_min_frame_fraction` | `0.6` | Calibration-frame recurrence required for a typical bin |
| `background_color_margin` | `12` | Extra exclusion buffer around learned colors |
| `confirmation_frames` | `3` | Consecutive stable candidates required |
| `confirmation_max_center_distance` | `0.12` | Maximum normalized centroid displacement between frames |
| `confirmation_max_gap` | `1.0` | Maximum time gap in seconds between confirming frames |

Increasing the color margin or confirmation count reduces sensitivity. If
changing processing rate, also choose a confirmation gap above the expected
interval between processed images.

Tests include the real PNG palettes, synthetic sky gradients and buffered color
changes, rare-color rejection during learning, frozen background behavior,
confirmation resets, quadrant overlays, guidance geometry, recalibration, and
real ROS image/message publication:

```bash
colcon test --packages-select kv_perception
colcon test-result --verbose
```

Rendered Gazebo smoke has not been evaluated for detection accuracy yet; tune
the thresholds against the actual camera stream before using it for navigation.
