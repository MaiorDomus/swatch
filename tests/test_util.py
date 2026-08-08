"""Tests for swatch.util"""

import string
import unittest

import cv2
import numpy as np

from swatch.config import ColorVariantConfig, ObjectConfig
from swatch.util import compute_solidity, detect_objects, get_random_suffix, mask_image


class TestMaskImage(unittest.TestCase):
    """Testing mask_image color matching."""

    def setUp(self) -> None:
        """Build a 4x4 image: left half a matching color, right half not."""
        self.crop = np.zeros((4, 4, 3), dtype="uint8")
        self.crop[:, :2] = (5, 5, 5)
        self.crop[:, 2:] = (250, 250, 250)

    def test_mask_image_keeps_matching_pixels(self) -> None:
        variant = ColorVariantConfig(color_lower="4, 4, 4", color_upper="6, 6, 6")
        output, matches = mask_image(self.crop, variant)

        assert matches == 4 * 2 * 3
        assert (output[:, :2] == (5, 5, 5)).all()
        assert (output[:, 2:] == (0, 0, 0)).all()

    def test_mask_image_no_matches_returns_zero(self) -> None:
        variant = ColorVariantConfig(
            color_lower="100, 100, 100", color_upper="101, 101, 101"
        )
        output, matches = mask_image(self.crop, variant)

        assert matches == 0
        assert (output == 0).all()

    def test_mask_image_zero_lower_bound_is_bumped_to_one(self) -> None:
        """color_lower of "0, 0, 0" is special cased to [1, 1, 1] so pure
        black pixels in the source image are never treated as a match."""
        crop = np.zeros((2, 2, 3), dtype="uint8")
        variant = ColorVariantConfig(color_lower="0, 0, 0", color_upper="2, 2, 2")

        _, matches = mask_image(crop, variant)

        assert matches == 0

    def test_color_lower_upper_are_interpreted_as_rgb_not_bgr(self) -> None:
        """color_lower/color_upper are documented (and produced by
        /api/colortest/values) as R, G, B. crop arrays (from cv2.imread/
        imdecode) are BGR, so a pixel that's clearly red (high R, low G/B)
        must match an R,G,B config target of "red", not get compared against
        the wrong channels and match "blue" instead."""
        # BGR order: low blue, low green, high red -- i.e. actually red.
        crop = np.full((2, 2, 3), (10, 10, 200), dtype="uint8")
        red_variant = ColorVariantConfig(
            color_lower="180, 0, 0", color_upper="255, 50, 50"
        )
        blue_variant = ColorVariantConfig(
            color_lower="0, 0, 180", color_upper="50, 50, 255"
        )

        _, red_matches = mask_image(crop, red_variant)
        _, blue_matches = mask_image(crop, blue_variant)

        assert red_matches == 2 * 2 * 3
        assert blue_matches == 0


def _first_contour(mask: np.ndarray):
    """Extract the first external contour from a mask, for solidity tests."""
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


class TestComputeSolidity(unittest.TestCase):
    """Testing compute_solidity: how convex/smooth a shape is, used to tell
    a real light fixture apart from a similarly-sized-and-shaped but
    irregular false positive (e.g. a reflection)."""

    def test_filled_rectangle_is_fully_solid(self) -> None:
        mask = np.zeros((20, 20, 3), dtype="uint8")
        mask[2:6, 2:12] = (255, 255, 255)

        assert compute_solidity(_first_contour(mask)) == 1.0

    def test_jagged_shape_has_low_solidity(self) -> None:
        """An "E" shape: its bounding box/convex hull is mostly empty space
        between the three arms, unlike a filled blob."""
        mask = np.zeros((20, 20, 3), dtype="uint8")
        mask[2:16, 2:4] = (255, 255, 255)  # spine
        mask[2:4, 2:16] = (255, 255, 255)  # top arm
        mask[8:10, 2:12] = (255, 255, 255)  # middle arm
        mask[14:16, 2:16] = (255, 255, 255)  # bottom arm

        assert compute_solidity(_first_contour(mask)) < 0.5


class TestDetectObjects(unittest.TestCase):
    """Testing detect_objects bounding box filtering."""

    def setUp(self) -> None:
        """Build a mask with a single 10x4 (area 40) white blob."""
        self.mask = np.zeros((20, 20, 3), dtype="uint8")
        self.mask[2:6, 2:12] = (255, 255, 255)

    def test_detect_objects_finds_blob_within_thresholds(self) -> None:
        obj = ObjectConfig(min_area=10, max_area=100, min_ratio=0, max_ratio=10)
        detected = detect_objects(self.mask, obj)

        assert len(detected) == 1
        assert detected[0]["area"] == 40
        assert detected[0]["box"] == [2, 2, 12, 6]
        assert detected[0]["solidity"] == 1.0

    def test_detect_objects_filters_out_blob_below_min_solidity(self) -> None:
        """A jagged "E" shape should be rejected by a high min_solidity even
        though its bounding box area/ratio would otherwise pass -- this is
        what tells a real fixture apart from a similarly-shaped-and-sized
        but irregular false positive elsewhere in the frame."""
        jagged_mask = np.zeros((20, 20, 3), dtype="uint8")
        jagged_mask[2:16, 2:4] = (255, 255, 255)
        jagged_mask[2:4, 2:16] = (255, 255, 255)
        jagged_mask[8:10, 2:12] = (255, 255, 255)
        jagged_mask[14:16, 2:16] = (255, 255, 255)

        obj = ObjectConfig(min_area=0, max_area=1000, min_solidity=0.9)
        assert detect_objects(jagged_mask, obj) == []

    def test_detect_objects_min_solidity_allows_smooth_blob_through(self) -> None:
        obj = ObjectConfig(min_area=10, max_area=100, min_solidity=0.9)
        detected = detect_objects(self.mask, obj)

        assert len(detected) == 1

    def test_detect_objects_filters_out_blob_below_min_area(self) -> None:
        obj = ObjectConfig(min_area=1000, max_area=2000)
        assert detect_objects(self.mask, obj) == []

    def test_detect_objects_filters_out_blob_above_max_area(self) -> None:
        obj = ObjectConfig(min_area=0, max_area=10)
        assert detect_objects(self.mask, obj) == []

    def test_detect_objects_filters_out_blob_by_ratio(self) -> None:
        # blob is 10 wide x 4 tall -> ratio 2.5, exclude it with a tight max_ratio
        obj = ObjectConfig(min_area=0, max_area=1000, min_ratio=0, max_ratio=1)
        assert detect_objects(self.mask, obj) == []

    def test_detect_objects_empty_mask_returns_nothing(self) -> None:
        empty_mask = np.zeros((20, 20, 3), dtype="uint8")
        obj = ObjectConfig()
        assert detect_objects(empty_mask, obj) == []


class TestGetRandomSuffix(unittest.TestCase):
    """Testing get_random_suffix."""

    def test_length_is_six(self) -> None:
        assert len(get_random_suffix()) == 6

    def test_charset_is_lowercase_alnum(self) -> None:
        allowed = set(string.ascii_lowercase + string.digits)
        assert set(get_random_suffix()) <= allowed

    def test_suffixes_are_not_all_identical(self) -> None:
        # extremely unlikely to collide 20 times in a row if truly random
        suffixes = {get_random_suffix() for _ in range(20)}
        assert len(suffixes) > 1


if __name__ == "__main__":
    unittest.main()
