"""Tests for swatch.util"""

import string
import unittest

import numpy as np

from swatch.config import ColorVariantConfig, ObjectConfig
from swatch.util import detect_objects, get_random_suffix, mask_image


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
