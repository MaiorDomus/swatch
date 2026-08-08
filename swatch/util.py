"""Utilities and convenience funs."""

import random
import string
from typing import Any

from colorthief import ColorThief
import cv2
import numpy as np

from swatch.config import ColorVariantConfig, ObjectConfig

### Image utils


def mask_image(crop: Any, color_variant: ColorVariantConfig) -> tuple[Any, int]:
    """Mask an image with color values"""
    color_lower = (
        [1, 1, 1]
        if color_variant.color_lower == "0, 0, 0"
        else color_variant.color_lower.split(", ")
    )
    color_upper = color_variant.color_upper.split(", ")

    # color_lower/color_upper are documented (and produced by
    # /api/colortest/values, which uses PIL under the hood) as R, G, B, but
    # crop comes from cv2.imread/imdecode, which is BGR -- reverse here so
    # each bound lines up with the channel it's actually meant to constrain,
    # instead of the R bound gating Blue and the B bound gating Red.
    lower: np.ndarray = np.array(
        [int(color_lower[2]), int(color_lower[1]), int(color_lower[0])],
        dtype="uint8",
    )
    upper: np.ndarray = np.array(
        [int(color_upper[2]), int(color_upper[1]), int(color_upper[0])],
        dtype="uint8",
    )

    mask = cv2.inRange(crop, lower, upper)
    output = cv2.bitwise_and(crop, crop, mask=mask)
    matches = int(np.count_nonzero(output))
    return (output, matches)


def compute_solidity(contour: Any) -> float:
    """How convex/smooth a contour's outline is: matched contour area over
    its convex hull area, in [0, 1]. A clean, filled oval or circular shape
    sits close to 1.0; an irregular or jagged blob (e.g. a diffuse
    reflection spread unevenly across a surface) is lower. This lets a real
    fixture's shape be told apart from a similarly-sized-and-shaped false
    positive elsewhere in the zone, which a bounding-box area/ratio check
    alone can't distinguish."""
    contour_area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))

    if hull_area == 0:
        return 0.0

    return float(contour_area / hull_area)


def detect_objects(mask: Any, obj: ObjectConfig) -> list[dict[str, Any]]:
    """Detect objects and return list of bounding boxes."""
    # get gray image
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # calculate contours
    _, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if obj.min_area < area < obj.max_area:
            if obj.min_ratio < (w / h) < obj.max_ratio:
                solidity = compute_solidity(contour)

                if obj.min_solidity < solidity < obj.max_solidity:
                    detected.append(
                        {
                            "box": [x, y, x + w, y + h],
                            "area": area,
                            "ratio": (w / h),
                            "solidity": solidity,
                        }
                    )

    return detected


def parse_colors_from_image(
    test_image: Any,
) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
    """Convenience fun to get colors from test image."""
    color_thief = ColorThief(test_image)
    main_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=3)
    return (main_color, palette)


### String utils


def get_random_suffix() -> str:
    """Returns 6 random character suffix string."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
