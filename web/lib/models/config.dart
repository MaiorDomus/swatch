import 'package:swatch/models/camera.dart';
import 'package:swatch/models/detectable.dart';

class Config {

  Map<String, Camera> cameras = {};
  Map<String, Detectable> objects = {};
  List<String> audioMonitors = [];

  Config(final Map<String, dynamic> json) {
    json["cameras"].forEach((name, json) => {
      cameras[name] = Camera(json)
    });

    json["objects"].forEach((name, json) => {
      objects[name] = Detectable(json)
    });

    if (json["audio_monitors"] != null) {
      audioMonitors = List<String>.from(json["audio_monitors"].keys);
    }
  }

  Config.template();
}