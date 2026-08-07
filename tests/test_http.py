"""Tests for the swatch Flask HTTP API."""

import io
import json
import unittest

from PIL import Image

from swatch.config import SwatchConfig
from swatch.http import create_app
from swatch.image import ImageProcessor
from swatch.snapshot import SnapshotProcessor


def _jpeg_bytes(color=(10, 10, 10), size=(20, 20)) -> bytes:
    """Build an in-memory solid-color JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


class TestHttpApi(unittest.TestCase):
    """Testing the Flask routes."""

    def setUp(self) -> None:
        self.config_dict = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "1, 1, 1",
                            "color_upper": "2, 2, 2",
                        },
                    },
                },
            },
            "cameras": {
                "test_cam": {
                    "snapshot_config": {"url": "http://localhost/snap.jpg"},
                    "zones": {
                        "test_zone": {
                            "coordinates": "1, 2, 3, 4",
                            "objects": ["test_obj"],
                        },
                    },
                },
            },
        }
        self.swatch_config = SwatchConfig(**self.config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(self.swatch_config)
        image_processor = ImageProcessor(self.swatch_config, snapshot_processor)
        app = create_app(self.swatch_config, image_processor, snapshot_processor)
        app.testing = True
        self.client = app.test_client()

    def test_status(self) -> None:
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert b"running" in resp.data

    def test_get_config(self) -> None:
        resp = self.client.get("/config")
        assert resp.status_code == 200
        body = json.loads(resp.data)
        assert body["cameras"]["test_cam"]["name"] == "test_cam"
        assert "test_obj" in body["objects"]

    def test_get_config_schema(self) -> None:
        resp = self.client.get("/config/schema")
        assert resp.status_code == 200
        assert resp.content_type == "application/json"
        schema = json.loads(resp.data)
        assert "$defs" in schema
        assert schema["title"] == "SwatchConfig"

    def test_colortest_mask_requires_image(self) -> None:
        resp = self.client.post("/colortest/mask", data={})
        assert resp.status_code == 404
        body = json.loads(resp.data)
        assert body["success"] is False

    def test_colortest_mask_requires_colors(self) -> None:
        resp = self.client.post(
            "/colortest/mask",
            data={"test_image": (io.BytesIO(_jpeg_bytes()), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 404
        body = json.loads(resp.data)
        assert body["success"] is False

    def test_colortest_mask_returns_masked_image(self) -> None:
        resp = self.client.post(
            "/colortest/mask",
            data={
                "test_image": (io.BytesIO(_jpeg_bytes(color=(5, 5, 5))), "test.jpg"),
                "color_lower": "4, 4, 4",
                "color_upper": "6, 6, 6",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/jpg"
        assert len(resp.data) > 0

    def test_colortest_values_returns_dominant_color(self) -> None:
        resp = self.client.post(
            "/colortest/values",
            data={"test_image": (io.BytesIO(_jpeg_bytes()), "test.jpg")},
            content_type="multipart/form-data",
        )
        body = json.loads(resp.data)
        assert body["success"] is True
        assert "dominant color" in body["message"]

    def test_detect_camera_frame_unknown_camera(self) -> None:
        resp = self.client.post("/does-not-exist/detect")
        assert resp.status_code == 404
        body = json.loads(resp.data)
        assert body["success"] is False
        assert "not a camera" in body["message"]


if __name__ == "__main__":
    unittest.main()
