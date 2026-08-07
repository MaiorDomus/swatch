"""Tests for SwatchApp"""

import os
import shutil
import tempfile
from unittest import mock
import unittest

import yaml

from swatch.app import SwatchApp


class TestApp(unittest.TestCase):
    """Testing the configuration is parsed correctly."""

    def setUp(self) -> None:
        """Point CONFIG_FILE/DB_FILE/MEDIA_DIR at an isolated temp dir so this
        doesn't depend on files existing on the host (e.g. /config/config.yaml)."""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "databases")
        self.db_file = os.path.join(self.db_path, "swatch.db")
        self.media_dir = os.path.join(self.tmp_dir, "media")
        os.makedirs(self.media_dir)

        config_file = os.path.join(self.tmp_dir, "config.yaml")
        with open(config_file, "w") as f:
            yaml.safe_dump({"objects": {}, "cameras": {}}, f)

        self.env_patcher = mock.patch.dict(
            os.environ,
            {
                "CONFIG_FILE": config_file,
                "DB_FILE": self.db_file,
                "MEDIA_DIR": self.media_dir,
            },
        )
        self.env_patcher.start()
        self.app: SwatchApp | None = None

    def tearDown(self) -> None:
        if self.app is not None:
            self.app.stop()
        self.env_patcher.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_db_created(self) -> None:
        """Test that the db is created and path is as expected."""
        assert not os.path.exists(self.db_file)
        self.app = SwatchApp()
        assert os.path.exists(self.db_path)
        assert os.path.isfile(self.db_file)

    def test_stop_does_not_crash_with_camera_without_auto_detect(self) -> None:
        """Regression: stop() used to KeyError joining self.camera_processes
        for every configured camera, even cameras with auto_detect disabled
        (which never get an AutoDetector thread in the first place)."""
        config_file = os.environ["CONFIG_FILE"]
        with open(config_file, "w") as f:
            yaml.safe_dump(
                {
                    "objects": {},
                    "cameras": {"front": {"auto_detect": 0}},
                },
                f,
            )

        self.app = SwatchApp()

        self.app.stop()  # should not raise KeyError
        self.app = None  # already stopped, don't stop again in tearDown
