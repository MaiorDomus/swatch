import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_layout_grid/flutter_layout_grid.dart';
import 'package:swatch/api/api.dart';
import 'package:swatch/components/component_audio_monitor.dart';
import 'package:swatch/components/component_camera.dart';
import 'package:swatch/const.dart';
import 'package:swatch/ext/extension_double.dart';
import 'package:swatch/models/config.dart';

import 'package:collapsible_sidebar/collapsible_sidebar.dart';
import 'package:swatch/theme/theme_helper.dart';

// Options shown in the refresh-interval dropdown; 0 means "Off".
const List<int> refreshIntervalOptions = [0, 2, 5, 10, 30, 60];
const int defaultRefreshSeconds = 5;

class DashboardRoute extends StatefulWidget {
  static const String route = '/dashboard';

  const DashboardRoute({Key? key}) : super(key: key);

  @override
  DashboardRouteState createState() => DashboardRouteState();
}

class DashboardRouteState extends State<DashboardRoute> {
  int _refreshSeconds = defaultRefreshSeconds;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Swatch"),
        centerTitle: false,
        backgroundColor: SwatchColors.getPrimaryColor(),
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  "Refresh:",
                  style: TextStyle(color: Colors.white),
                ),
                const SizedBox(width: 8.0),
                DropdownButton<int>(
                  value: _refreshSeconds,
                  dropdownColor: Colors.blueGrey[700],
                  underline: const SizedBox(),
                  iconEnabledColor: Colors.white,
                  items: refreshIntervalOptions
                      .map(
                        (seconds) => DropdownMenuItem(
                          value: seconds,
                          child: Text(
                            seconds == 0 ? "Off" : "${seconds}s",
                            style: const TextStyle(color: Colors.white),
                          ),
                        ),
                      )
                      .toList(),
                  onChanged: (value) {
                    if (value != null) {
                      setState(() => _refreshSeconds = value);
                    }
                  },
                ),
              ],
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: Stack(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: CollapsibleSidebar(
                isCollapsed: true,
                items: getSidebarRoutes(context, DashboardRoute.route),
                avatarImg: const NetworkImage(
                  "https://raw.githubusercontent.com/MaiorDomus/swatch/master/assets/swatch.png",
                ),
                body: _DashboardView(refreshSeconds: _refreshSeconds),
                backgroundColor: Colors.blueGrey[700]!,
                selectedTextColor: SwatchColors.getPrimaryColor(),
                iconSize: 24,
                borderRadius: 12,
                duration: const Duration(seconds: 0),
                sidebarBoxShadow: const [],
                title: "Swatch",
                textStyle: const TextStyle(
                  fontSize: 16,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DashboardView extends StatefulWidget {
  final int refreshSeconds;

  const _DashboardView({required this.refreshSeconds, Key? key})
      : super(key: key);

  @override
  State<_DashboardView> createState() => _DashboardViewState();
}

class _DashboardViewState extends State<_DashboardView> {
  final SwatchApi _api = SwatchApi();
  Timer? _timer;
  // Appended to camera/zone snapshot URLs so Image.network doesn't just
  // serve its cached copy of the same URL on every refresh tick.
  int _cacheBuster = DateTime.now().millisecondsSinceEpoch;

  @override
  void initState() {
    super.initState();
    _scheduleRefresh();
  }

  @override
  void didUpdateWidget(_DashboardView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.refreshSeconds != oldWidget.refreshSeconds) {
      _scheduleRefresh();
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _scheduleRefresh() {
    _timer?.cancel();

    if (widget.refreshSeconds <= 0) {
      return;
    }

    _timer = Timer.periodic(
      Duration(seconds: widget.refreshSeconds),
      (_) => setState(() {
        _cacheBuster = DateTime.now().millisecondsSinceEpoch;
      }),
    );
  }

  @override
  Widget build(BuildContext context) {
    final columnCount = MediaQuery.of(context).size.width.getColumnsForWidth();

    return Scaffold(
      body: FutureBuilder(
        future: _api.getConfig(),
        builder: (context, AsyncSnapshot<Config> config) {
          if (config.hasData) {
            return LayoutGrid(
              columnSizes: List.generate(columnCount, (index) => 1.fr),
              rowSizes: List.generate(columnCount, (index) => auto),
              children: [
                ..._getCameras(config.data!),
                ..._getAudioMonitors(config.data!),
              ],
            );
          } else {
            return Container();
          }
        },
      ),
    );
  }

  List<Widget> _getCameras(Config config) {
    final keys = config.cameras.keys.toList();
    return List.generate(
      config.cameras.length,
      (index) => CameraComponent(
        config.cameras[keys[index]]!,
        cacheBuster: _cacheBuster,
      ),
    );
  }

  List<Widget> _getAudioMonitors(Config config) {
    return List.generate(
      config.audioMonitors.length,
      (index) => AudioMonitorComponent(config.audioMonitors[index]),
    );
  }
}
