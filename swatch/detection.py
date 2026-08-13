"""For processing of images."""

import datetime
import logging
import multiprocessing
import threading
import time
from typing import Any

from swatch.config import CameraConfig, SwatchConfig
from swatch.debounce import TimeBasedSustainedStateTracker
from swatch.image import ImageProcessor
from swatch.models import Detection
from swatch.snapshot import SnapshotProcessor
from swatch.util import get_random_suffix

logger = logging.getLogger(__name__)


class AutoDetector(threading.Thread):
    """Handles the auto running of detection on cameras."""

    def __init__(
        self,
        image_processor: ImageProcessor,
        snap_processor: SnapshotProcessor,
        camera_config: CameraConfig,
        stop_event: multiprocessing.Event,
    ) -> None:
        threading.Thread.__init__(self)
        # AutoDetector is only ever constructed with cameras sourced from
        # SwatchConfig.runtime_config, which always stamps camera_config.name,
        # and only when snapshot_config.url is set (see SwatchApp.__init_processing__).
        assert camera_config.name is not None
        assert camera_config.snapshot_config.url is not None
        self.image_processor = image_processor
        self.snap_processor = snap_processor
        self.config = camera_config
        self.camera_name: str = camera_config.name
        self.snapshot_url: str = camera_config.snapshot_config.url
        self.stop_event = stop_event
        self.obj_data: dict[str, Any] = {}
        self.trackers: dict[str, TimeBasedSustainedStateTracker] = {}

    def __handle_db__(self, db_type: str, obj_id: str) -> None:
        """Handle the db transactions for detection."""
        now = datetime.datetime.now().timestamp()

        if db_type == "new":
            Detection.replace(
                id=self.obj_data[obj_id]["id"],
                label=self.obj_data[obj_id]["object_name"],
                camera=self.camera_name,
                zone=self.obj_data[obj_id]["zone_name"],
                color_variant=self.obj_data[obj_id]["variant"],
                start_time=now,
                top_area=self.obj_data[obj_id]["top_area"],
            ).execute()
        elif db_type == "update":
            Detection.update(
                color_variant=self.obj_data[obj_id]["variant"],
                top_area=self.obj_data[obj_id]["top_area"],
            ).where(Detection.id == self.obj_data[obj_id]["id"]).execute()
        elif db_type == "end":
            Detection.update(
                color_variant=self.obj_data[obj_id]["variant"],
                top_area=self.obj_data[obj_id]["top_area"],
                end_time=now,
            ).where(Detection.id == self.obj_data[obj_id]["id"]).execute()

    def __debounce_results__(
        self, detection_result: dict[str, Any], now: float | None = None
    ) -> None:
        """Debounce each zone/object's raw single-frame result in place, so a
        stray miss or false positive from one noisy frame doesn't flip the
        reported state. A single frame can fail area/ratio/solidity
        thresholds on a real match (JPEG artifacts, exposure flicker) even
        when the object is genuinely present, the same noisy-single-sample
        problem audio_monitors already solve with a sustained on/off window.

        Uses real elapsed time (TimeBasedSustainedStateTracker) rather than
        counting auto_detect ticks: each cycle also does an HTTP snapshot
        fetch, which routinely takes longer than auto_detect's configured
        interval, so counting ticks would silently stretch min_on_seconds/
        min_off_seconds well past what's configured."""
        if now is None:
            now = time.monotonic()

        for zone_name, objects in detection_result.items():
            for object_name, object_result in objects.items():
                tracker_key = f"{self.camera_name}.{zone_name}.{object_name}"

                if tracker_key not in self.trackers:
                    obj_config = self.image_processor.config.objects[object_name]
                    self.trackers[tracker_key] = TimeBasedSustainedStateTracker(
                        min_on_seconds=obj_config.min_on_seconds,
                        min_off_seconds=obj_config.min_off_seconds,
                    )

                object_result["result"] = self.trackers[tracker_key].update(
                    bool(object_result.get("result", False)), now
                )

    def __handle_detections__(self, detection_result: dict[str, Any]) -> None:
        """Run through map of detections for camera and add to the db."""
        cam_name = self.camera_name

        for zone_name, objects in detection_result.items():
            for object_name, object_result in objects.items():
                non_unique_id = f"{cam_name}.{zone_name}.{object_name}"

                if not self.obj_data.get(non_unique_id) and not object_result.get(
                    "result"
                ):
                    continue

                if not self.obj_data.get(non_unique_id):
                    self.obj_data[non_unique_id] = {}

                unique_id = (
                    f"{non_unique_id}.{get_random_suffix()}"
                    if not self.obj_data[non_unique_id].get("id")
                    else self.obj_data[non_unique_id]["id"]
                )

                self.obj_data[non_unique_id]["object_name"] = object_name
                self.obj_data[non_unique_id]["zone_name"] = zone_name
                self.obj_data[non_unique_id]["variant"] = object_result.get(
                    "variant", "default"
                )

                if object_result.get("objects"):
                    top_area = max([d["area"] for d in object_result["objects"]])
                    best_box = next(
                        d for d in object_result["objects"] if d["area"] == top_area
                    )["box"]

                    if top_area > self.obj_data[non_unique_id].get("top_area", 0):
                        self.obj_data[non_unique_id]["top_area"] = top_area

                        # save snapshot with best area
                        self.snap_processor.save_detection_snapshot(
                            cam_name,
                            zone_name,
                            unique_id,
                            best_box,
                        )

                if not self.obj_data[non_unique_id].get("id"):
                    self.obj_data[non_unique_id]["id"] = unique_id
                    self.__handle_db__("new", non_unique_id)
                else:
                    if object_result.get("result", False):
                        self.__handle_db__("update", non_unique_id)
                    else:
                        self.__handle_db__("end", non_unique_id)
                        del self.obj_data[non_unique_id]

    def run(self) -> None:
        # pylint: disable=singleton-comparison
        logger.info("Starting Auto Detection for %s", self.camera_name)

        while not self.stop_event.wait(self.config.auto_detect):
            try:
                result: dict[str, Any] = self.image_processor.detect(
                    self.camera_name, self.snapshot_url
                )
                self.__debounce_results__(result)
                self.__handle_detections__(result)
            except Exception:
                logger.exception(
                    "Unexpected error during detection for %s", self.camera_name
                )

        # ensure db doesn't contain bad data after shutdown
        Detection.update(end_time=datetime.datetime.now().timestamp()).where(
            Detection.end_time == None
        ).execute()
        logger.info("Stopping Auto Detection for %s", self.camera_name)


class DetectionCleanup(threading.Thread):
    """Handles the auto cleanup of detections."""

    def __init__(self, config: SwatchConfig, stop_event: multiprocessing.Event):
        threading.Thread.__init__(self)
        self.config: SwatchConfig = config
        self.stop_event: multiprocessing.Event = stop_event

    def __cleanup_db__(self) -> None:
        """Cleanup the old events in the db."""

        for cam_name, cam_config in self.config.cameras.items():
            expire_days = cam_config.snapshot_config.retain_days
            expire_after = (
                datetime.datetime.now() - datetime.timedelta(days=expire_days)
            ).timestamp()

            Detection.delete().where(
                Detection.camera == cam_name,
                Detection.start_time < expire_after,
            ).execute()

        # audio_monitors' Detection rows have no camera (see
        # AudioMonitor.__record_transition__), so they'd never match the
        # loop above and would accumulate forever -- prune by label instead.
        for monitor_name, monitor_config in self.config.audio_monitors.items():
            expire_after = (
                datetime.datetime.now()
                - datetime.timedelta(days=monitor_config.retain_days)
            ).timestamp()

            Detection.delete().where(
                Detection.label == monitor_name,
                Detection.start_time < expire_after,
            ).execute()

    def run(self) -> None:
        logger.info("Starting Detection Cleanup")

        while not self.stop_event.wait(3600):
            self.__cleanup_db__()

        logger.info("Stopping Detection Cleanup")
