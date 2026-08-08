
import 'package:flutter/material.dart';
import 'package:swatch/api/api.dart';

import 'package:collapsible_sidebar/collapsible_sidebar.dart';
import 'package:swatch/theme/theme_helper.dart';
import 'package:swatch/const.dart';

class SettingsRoute extends StatefulWidget {
  static const String route = '/settings';

  const SettingsRoute({Key? key}) : super(key: key);

  @override
  SettingsRouteState createState() => SettingsRouteState();
}

class SettingsRouteState extends State<SettingsRoute> {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Swatch"),
        centerTitle: false,
        backgroundColor: SwatchColors.getPrimaryColor(),
      ),
      body: SafeArea(
        child: Stack(
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: CollapsibleSidebar(
                isCollapsed: true,
                items: getSidebarRoutes(context, SettingsRoute.route),
                avatarImg: const NetworkImage(
                  "https://raw.githubusercontent.com/MaiorDomus/swatch/master/assets/swatch.png",
                ),
                body: _SettingsView(),
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

class _SettingsView extends StatelessWidget {
  final SwatchApi _api = SwatchApi();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.only(bottom: 8.0),
              child: Text(
                "config.yaml",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
            ),
            const Padding(
              padding: EdgeInsets.only(bottom: 8.0),
              child: Text(
                "Read-only -- edit config.yaml directly to make changes.",
                style: TextStyle(color: Colors.grey, fontSize: 12),
              ),
            ),
            Expanded(
              child: FutureBuilder<String?>(
                future: _api.getRawConfig(),
                builder: (context, AsyncSnapshot<String?> snapshot) {
                  if (!snapshot.hasData) {
                    if (snapshot.connectionState == ConnectionState.done) {
                      return const Text("Unable to load config.yaml.");
                    }

                    return const Center(
                      child: SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    );
                  }

                  return Card(
                    shape: const RoundedRectangleBorder(
                      borderRadius: BorderRadius.all(Radius.circular(8.0)),
                    ),
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.all(12.0),
                      child: SelectableText(
                        snapshot.data!,
                        style: const TextStyle(
                          fontFamily: "monospace",
                          fontSize: 13,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
