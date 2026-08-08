import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:swatch/models/config.dart';
import 'package:swatch/models/detection_event.dart';
import 'package:window_location_href/window_location_href.dart';

class SwatchApi {
  static final SwatchApi _singleton = SwatchApi._internal();
  static String _swatchHost = "";
  // Matches whatever scheme the page itself was loaded with (e.g. behind a
  // reverse proxy forcing https) -- API calls previously always used
  // Uri.http() regardless, which browsers block as mixed content on an
  // https page even once the host is parsed correctly.
  static String _swatchScheme = "http";

  factory SwatchApi() {
    if (kDebugMode) {
      _swatchHost = "localhost:4500";
      _swatchScheme = "http";
    } else {
      // Uri.parse handles https (and ports, paths, etc.) correctly; the
      // previous approach (stripping the literal string "http://") left an
      // https:// href untouched, since that substring never occurs in one.
      final uri = Uri.tryParse(getHref() ?? "");

      if (uri != null && uri.host.isNotEmpty) {
        _swatchHost = uri.hasPort ? "${uri.host}:${uri.port}" : uri.host;
        _swatchScheme = uri.scheme.isNotEmpty ? uri.scheme : "http";
      }
    }

    return _singleton;
  }

  SwatchApi._internal();

  Uri _apiUri(final String path, [final Map<String, String>? params]) {
    return _swatchScheme == "https"
        ? Uri.https(_swatchHost, path, params)
        : Uri.http(_swatchHost, path, params);
  }

  String getHost() => _apiUri("").toString();

  /// Swatch API Funs

  Future<Config> getConfig() async {
    const base = "/api/config";
    final response = await http.get(_apiUri(base)).timeout(
          const Duration(seconds: 15),
        );

    if (response.statusCode == 200) {
      return Config(json.decode(response.body));
    } else {
      return Config.template();
    }
  }

  Future<List<DetectionEvent>> getDetections({
    final String? label,
    final int? limit,
  }) async {
    const base = "/api/detections";
    final params = <String, String>{
      if (label != null) "label": label,
      if (limit != null) "limit": limit.toString(),
    };
    final response = await http
        .get(_apiUri(base, params.isEmpty ? null : params))
        .timeout(const Duration(seconds: 15));

    if (response.statusCode == 200) {
      final parsed = json.decode(response.body);
      return List<DetectionEvent>.from(
        parsed.map((model) => DetectionEvent(model)),
      );
    } else {
      return [];
    }
  }

  Future<Config> getLatest() async {
    const base = "/api/all/latest";
    final response = await http.get(_apiUri(base)).timeout(
          const Duration(seconds: 15),
        );

    if (response.statusCode == 200) {
      return Config(json.decode(response.body));
    } else {
      return Config.template();
    }
  }

  Future<Map<String, dynamic>> getLatestForLabel(final String label) async {
    final base = "/api/$label/latest";
    final response = await http.get(_apiUri(base)).timeout(
          const Duration(seconds: 15),
        );

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      return {};
    }
  }

  Future<Uint8List> testImageMask(
    Uint8List image,
    String colorLower,
    String colorUpper,
  ) async {
    const base = "/api/colortest/mask";
    final request = http.MultipartRequest("POST", _apiUri(base));
    request.fields["color_lower"] = colorLower;
    request.fields["color_upper"] = colorUpper;
    request.files.add(
      http.MultipartFile.fromBytes("test_image", image,
          filename: 'test_image', contentType: MediaType("image", "jpg")),
    );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return response.bodyBytes;
    } else {
      if (kDebugMode) {
        print("testImageMask::${response.toString()}");
      }

      return image;
    }
  }

  // Detection Specific Funs

  Future<bool> deleteDetection(final String detectionId) async {
    final base = "/api/detections/$detectionId";
    final response = await http.delete(_apiUri(base)).timeout(
      const Duration(seconds: 15),
    );

    if (response.statusCode == 200) {
      return true;
    } else {
      return false;
    }
  }

  /// General Utility Funs

  Future<Uint8List> getImageBytes(final String imageSource) async {
    try {
      final http.Response r = await http.get(
        Uri.parse(imageSource),
      );
      return r.bodyBytes;
    } catch (e) {
      return Uint8List(0);
    }
  }
}
