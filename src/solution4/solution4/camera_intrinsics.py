"""Pinhole intrinsics for the head camera, taken from the robot description.

The obvious source is /camera_info, but the gz bridge publishes that only
around start-up and marks it VOLATILE, so a node that comes up later never
receives it - measured as zero messages over sixty seconds while the image
stream ran at 7 Hz. /robot_description is latched, so deriving the intrinsics
from the sensor's declared field of view always works and gives the same
numbers: 87 degrees across 640 px yields fx = 337.21, matching camera_info's
337.2096 to four decimals.
"""
import math
import re

_SENSOR = re.compile(
    r'<horizontal_fov>\s*([0-9.eE+-]+)\s*</horizontal_fov>.*?'
    r'<width>\s*(\d+)\s*</width>.*?<height>\s*(\d+)\s*</height>',
    re.S,
)


def intrinsics_from_urdf(urdf_text):
    """Return (fx, fy, cx, cy), or None if the description declares no camera."""
    match = _SENSOR.search(urdf_text)
    if not match:
        return None
    hfov = float(match.group(1))
    width = float(match.group(2))
    height = float(match.group(3))
    if hfov <= 0 or width <= 0 or height <= 0:
        return None
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    # Square pixels: the same focal length governs both axes.
    return (fx, fx, width / 2.0, height / 2.0)
