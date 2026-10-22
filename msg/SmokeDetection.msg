# Source RGB image stamp and frame. A consumer should reject stale observations.
std_msgs/Header header

uint8 NONE=0
uint8 TOP_LEFT=1
uint8 TOP_RIGHT=2
uint8 BOTTOM_LEFT=3
uint8 BOTTOM_RIGHT=4

# Detection lifecycle. Guidance is published only in CONFIRMED status.
uint8 CALIBRATING=0
uint8 SEARCHING=1
uint8 CONFIRMING=2
uint8 CONFIRMED=3
uint8 status
bool background_ready
# Fraction of required startup calibration frames collected, 0..1.
float32 calibration_progress
# Consecutive spatially consistent candidate frames, capped at configured requirement.
uint32 confirmation_frames

# True only after background calibration and multi-frame confirmation.
bool detected
# Primary target: the quadrant of the largest accepted blob's centroid.
uint8 region
# All occupied quadrants; always includes region when detected=true.
uint8[] regions
# Accepted blob pixel fraction in each quadrant, ordered TL, TR, BL, BR.
float32[4] quadrant_coverage
# Accepted blob pixel fraction across the entire image; not a confidence score.
float32 coverage

# True when guidance uses CameraInfo; false when it uses configured FOV.
bool camera_info_used
# Relative yaw: positive turns left (CCW), negative turns right (CW), radians.
float32 yaw_correction_rad
# Positive means the target is above image center, radians.
float32 vertical_correction_rad

# Metres require a distance assumption; these are advisory estimates.
# For a forward-facing level camera, positive altitude means climb, negative descend.
bool altitude_estimate_valid
# NaN when no smoke is detected or assumed_distance_m is zero.
float32 altitude_correction_m
# Assumed FORWARD distance from the camera to the target plane (not measured range).
# Zero disables the metric estimate. Angles remain available for detections.
float32 assumed_distance_m
