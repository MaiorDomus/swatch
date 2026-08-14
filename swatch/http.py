"""Main http service that handles starting app modules."""

from functools import reduce
import json
import logging
import os
from typing import Any

import cv2
import numpy as np

from flask import (
    Blueprint,
    Flask,
    current_app,
    jsonify,
    make_response,
    request,
)

from peewee import DoesNotExist, operator
from playhouse.shortcuts import model_to_dict

from swatch.audio import AudioMonitor
from swatch.config import CameraConfig, ColorVariantConfig, SwatchConfig, ZoneConfig
from swatch.const import CONST_CONFIG_FILE, ENV_CONFIG
from swatch.image import ImageProcessor
from swatch.models import Detection
from swatch.snapshot import SnapshotProcessor
from swatch.util import mask_image, parse_colors_from_image

logger = logging.getLogger(__name__)
flask_logger = logging.getLogger("werkzeug")
bp = Blueprint("swatch", __name__)


def create_app(
    swatch_config: SwatchConfig,
    image_processor: ImageProcessor,
    snapshot_processor: SnapshotProcessor,
    audio_monitors: dict[str, AudioMonitor] | None = None,
) -> Flask:
    """Creates the Flask app to run the webserver."""
    app = Flask(__name__)
    disable_logs()
    app.register_blueprint(bp)
    app.swatch_config = swatch_config
    app.image_processor = image_processor
    app.snapshot_processor = snapshot_processor
    app.audio_monitors = audio_monitors or {}
    return app


### Basic / Frontend Routes


@bp.route("/")
def status() -> str:
    """Return Swatch stats."""
    return "Swatch is running."


### Config API Routes


@bp.route("/config", methods=["GET"])
def get_config() -> Any:
    """Get current config."""
    return make_response(jsonify(current_app.swatch_config.model_dump()), 200)


@bp.route("/config/schema", methods=["GET"])
def get_config_schema() -> Any:
    """Get schema for the swatch config.
    Which is useful for vscode or other code completion."""
    return current_app.response_class(
        json.dumps(current_app.swatch_config.model_json_schema()),
        mimetype="application/json",
    )


@bp.route("/config/raw", methods=["GET"])
def get_config_raw() -> Any:
    """Get the config.yaml file's contents exactly as the user wrote them
    (comments and formatting included), unlike /config which returns the
    parsed and re-serialized SwatchConfig model."""
    config_file = os.environ.get(ENV_CONFIG, CONST_CONFIG_FILE)

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            contents = file.read()
    except OSError:
        return make_response("", 404)

    return current_app.response_class(contents, mimetype="text/plain")


### Color Testing Routes


@bp.route("/colortest/values", methods=["POST"])
def test_colors() -> Any:
    """Test and get color values inside of test image."""
    test_image = request.files.get("test_image") if request.files else None

    if not test_image:
        return make_response(
            jsonify(
                {"success": False, "message": "An image needs to be sent as test_image"}
            ),
            404,
        )

    main_color, palette = parse_colors_from_image(test_image)

    return make_response(
        jsonify(
            {
                "success": True,
                "message": f"The dominant color is {main_color} with a mixed palette as {palette}",
            }
        ),
        200,
    )


@bp.route("/colortest/mask", methods=["POST"])
def test_mask() -> Any:
    """Test and get masked image for given lower and upper color values."""
    test_image = request.files.get("test_image") if request.files else None

    if not test_image:
        return make_response(
            jsonify(
                {"success": False, "message": "An image needs to be sent as test_image"}
            ),
            404,
        )

    color_lower = request.form.get("color_lower")
    color_upper = request.form.get("color_upper")

    if not color_lower or not color_upper:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": "color_lower and color_upper need to be provided",
                }
            ),
            404,
        )

    image_str = test_image.read()

    img = cv2.imdecode(np.frombuffer(image_str, np.uint8), -1)
    test_color_variant = ColorVariantConfig(
        color_lower=color_lower, color_upper=color_upper
    )
    img, _ = mask_image(img, test_color_variant)

    if img is None:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": "color_lower and color_upper need to be provided",
                }
            ),
            500,
        )

    _, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    response = make_response(jpg.tobytes())
    response.headers["Content-Type"] = "image/jpg"
    return response


### Detection API Routes


@bp.route("/detections", methods=["GET"])
def get_detections() -> Any:
    """Get detections from the db."""
    limit = request.args.get("limit", 100, type=int)
    camera = request.args.get("camera", "all")
    label = request.args.get("label", "all")
    zone = request.args.get("zone", "all")
    after = request.args.get("after", type=float)
    before = request.args.get("before", type=float)

    clauses: list[Any] = []
    excluded_fields: list[str] = []

    selected_columns = [
        Detection.id,
        Detection.camera,
        Detection.label,
        Detection.zone,
        Detection.top_area,
        Detection.color_variant,
        Detection.start_time,
        Detection.end_time,
    ]

    if camera != "all":
        clauses.append(Detection.camera == camera)

    if label != "all":
        clauses.append(Detection.label == label)

    if zone != "all":
        clauses.append(Detection.zone == zone)

    if after:
        clauses.append(Detection.start_time > after)

    if before:
        clauses.append(Detection.start_time < before)

    if len(clauses) == 0:
        clauses.append(True)

    detections = (
        Detection.select(*selected_columns)
        .where(reduce(operator.and_, clauses))
        .order_by(Detection.start_time.desc())
        .limit(limit)
    )

    return jsonify([model_to_dict(d, exclude=excluded_fields) for d in detections])


@bp.route("/detections/<detection_id>", methods=["GET"])
def get_detection(detection_id: str) -> Any:
    """Get specific detection."""
    try:
        return model_to_dict(Detection.get(Detection.id == detection_id))
    except DoesNotExist:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": f"Detection with id {detection_id} not found.",
                }
            ),
            404,
        )


@bp.route("/detections/<detection_id>", methods=["DELETE"])
def delete_detection(detection_id: str) -> Any:
    """Get specific detection."""
    try:
        Detection.delete().where(Detection.id == detection_id).execute()
        return make_response(
            jsonify(
                {
                    "success": True,
                    "message": "Deleted successfully.",
                }
            ),
            200,
        )
    except DoesNotExist:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": f"Detection with id {detection_id} not found.",
                }
            ),
            404,
        )


@bp.route("/<camera_name>/detect", methods=["POST"])
def detect_camera_frame(camera_name: str) -> Any:
    """Use camera frame to detect known objects."""
    if not camera_name:
        return make_response(
            jsonify({"success": False, "message": "camera_name must be set."}), 404
        )

    camera_config: CameraConfig = current_app.swatch_config.cameras.get(camera_name)

    if not camera_config:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": f"{camera_name} is not a camera in the config.",
                }
            ),
            404,
        )

    if (
        not (request.json and request.json.get("imageUrl"))
        and camera_config.snapshot_config.url
    ):
        image_url = camera_config.snapshot_config.url
    elif request.json:
        image_url = request.json.get("imageUrl")
    else:
        image_url = None

    if image_url:
        try:
            result: dict[str, Any] = current_app.image_processor.detect(
                camera_name, image_url
            )
        except Exception as _e:
            return make_response(
                jsonify(
                    {
                        "success": False,
                        "message": f"{image_url} is invalid or does not contain a valid image: {_e}.",
                    }
                ),
                404,
            )

        if result:
            return make_response(jsonify(result), 200)

        return make_response(
            jsonify({"success": False, "message": "Unknown error doing detection."}),
            500,
        )

    return make_response(
        jsonify(
            {
                "success": False,
                "message": "image url must be passed or set in the config.",
            }
        ),
        404,
    )


@bp.route("/<label>/latest", methods=["GET"])
def get_latest_result(label: str) -> Any:
    """Get the latest results for a label.

    Audio monitor state shares this namespace with image-detected object
    results, so a single "all" poll (used by the Home Assistant integration's
    update coordinator) picks up both without an extra request.
    """
    if not label:
        return make_response(
            jsonify({"success": False, "message": "Label needs to be provided"}), 404
        )

    audio_results = {
        name: {"result": monitor.is_on}
        for name, monitor in current_app.audio_monitors.items()
    }

    if label == "all":
        combined = {**current_app.image_processor.latest_results, **audio_results}
        return make_response(jsonify(combined), 200)

    if label in audio_results:
        return make_response(jsonify(audio_results[label]), 200)

    return current_app.image_processor.get_latest_result(label)

    ### Snapshot API Routes


@bp.route("/<camera_name>/snapshot.jpg", methods=["GET"])
def get_latest_camera_snapshot(camera_name: str) -> Any:
    """Get the latest snapshot for <camera_name>."""
    if not camera_name:
        return make_response(
            jsonify({"success": False, "message": "camera_name must be provided."}),
            404,
        )

    camera_config = current_app.swatch_config.cameras.get(camera_name)

    if not camera_config:
        return make_response(
            jsonify(
                {"success": False, "message": f"{camera_name} is not a valid camera."}
            ),
            404,
        )

    jpg_bytes = current_app.snapshot_processor.get_latest_camera_snapshot(camera_name)

    if not jpg_bytes:
        return make_response(
            jsonify({"success": False, "message": "Failed to load image from camera."}),
            500,
        )

    response = make_response(jpg_bytes)
    response.headers["Content-Type"] = "image/jpg"
    return response


@bp.route("/<camera_name>/<zone_name>/snapshot.jpg", methods=["GET"])
def get_latest_zone_snapshot(camera_name: str, zone_name: str) -> Any:
    """Get the latest snapshot for <camera_name>."""
    if not camera_name:
        return make_response(
            jsonify({"success": False, "message": "camera_name must be provided."}),
            404,
        )

    camera_config: CameraConfig() = current_app.swatch_config.cameras.get(camera_name)

    if not camera_config:
        return make_response(
            jsonify(
                {"success": False, "message": f"{camera_name} is not a valid camera."}
            ),
            404,
        )

    if not zone_name:
        return make_response(
            jsonify({"success": False, "message": "zone_name must be provided."}), 404
        )

    zone_config: ZoneConfig = camera_config.zones.get(zone_name)

    if not zone_config:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": f"{zone_name} is not a valid zone for {camera_name}.",
                }
            ),
            404,
        )

    jpg_bytes = current_app.snapshot_processor.get_latest_zone_snapshot(
        camera_name, zone_name
    )

    if not jpg_bytes:
        return make_response(
            jsonify({"success": False, "message": "Failed to load image from camera."}),
            500,
        )

    response = make_response(jpg_bytes)
    response.headers["Content-Type"] = "image/jpg"
    return response


@bp.route("/detections/<detection_id>/snapshot.jpg", methods=["GET"])
def get_detection_snapshot(detection_id: str) -> Any:
    """Get specific detection snapshot."""
    try:
        detection = Detection.get(Detection.id == detection_id)
        jpg_bytes = current_app.snapshot_processor.get_detection_snapshot(detection)

        if jpg_bytes:
            response = make_response(jpg_bytes)
            response.headers["Content-Type"] = "image/jpg"
            return response

        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": f"Error loading snapshot for {detection_id}.",
                }
            ),
            500,
        )

    except DoesNotExist:
        return make_response(
            jsonify(
                {
                    "success": False,
                    "message": f"Detection with id {detection_id} not found.",
                }
            ),
            404,
        )


@bp.route("/<camera_name>/detection.jpg", methods=["GET"])
def get_latest_detection(camera_name: str) -> Any:
    """Get the latest detection for <camera_name>."""
    if not camera_name:
        return make_response(
            jsonify({"success": False, "message": "camera_name must be provided."}),
            404,
        )

    camera_config = current_app.swatch_config.cameras.get(camera_name)

    if not camera_config:
        return make_response(
            jsonify(
                {"success": False, "message": f"{camera_name} is not a valid camera."}
            ),
            404,
        )

    jpg_bytes = current_app.snapshot_processor.get_latest_detection_snapshot(
        camera_name
    )

    response = make_response(jpg_bytes)
    response.headers["Content-Type"] = "image/jpg"
    return response


### Util Funs


def disable_logs():
    """Disable flask logs"""
    flask_logger.disabled = True
