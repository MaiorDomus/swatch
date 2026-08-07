import 'package:flutter/material.dart';
import 'package:swatch/api/api.dart';
import 'package:swatch/ext/extension_string.dart';

class AudioMonitorComponent extends StatelessWidget {
  final SwatchApi _api = SwatchApi();
  final String name;

  AudioMonitorComponent(
    this.name, {
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(
          Radius.circular(8.0),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.start,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              name.replaceAll('_', ' ').title(),
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Padding(
              padding: EdgeInsets.only(top: 4.0),
              child: Text(
                "Audio Monitor",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: FutureBuilder(
                future: _api.getLatestForLabel(name),
                builder: (
                  context,
                  AsyncSnapshot<Map<String, dynamic>> snapshot,
                ) {
                  final bool isOn =
                      snapshot.hasData && snapshot.data!["result"] == true;

                  return Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        isOn ? Icons.volume_up : Icons.volume_off,
                        color: isOn ? Colors.greenAccent : Colors.grey,
                      ),
                      const SizedBox(width: 8.0),
                      Text(
                        isOn ? "On" : "Off",
                        style: TextStyle(
                          color: isOn ? Colors.greenAccent : Colors.grey,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
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
