// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter_test/flutter_test.dart';

import 'package:swatch/models/config.dart';

void main() {
  final config = {
    "objects": {
      "test_obj": {
        "color_variants": {
          "default": {
            "color_lower": "1, 1, 1",
            "color_upper": "2, 2, 2",
          },
        },
        "min_area": 0,
        "max_area": 100000,
      },
    },
    "cameras": {
      "test_cam": {
        "snapshot_config": {
          "url": "http://localhost/snap.jpg",
        },
        "zones": {
          "test_zone": {
            "coordinates": "1, 2, 3, 4",
            "objects": ["test_obj"],
          },
        },
      },
    },
  };

  final configWithAudioMonitors = {
    ...config,
    "audio_monitors": {
      "kitchen_hood": {
        "rtsp_url": "rtsps://192.168.1.1:7441/abc",
      },
    },
  };

  test("Config is parsed correctly", () {
    final swatchConfig = Config(config);
    assert(swatchConfig.cameras.isNotEmpty);
  });

  test("Config with no audio_monitors key parses to an empty list", () {
    final swatchConfig = Config(config);
    assert(swatchConfig.audioMonitors.isEmpty);
  });

  test("Config audio_monitors are parsed as a list of names", () {
    final swatchConfig = Config(configWithAudioMonitors);
    assert(swatchConfig.audioMonitors.length == 1);
    assert(swatchConfig.audioMonitors.first == "kitchen_hood");
  });
}
