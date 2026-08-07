import 'package:flutter/material.dart';
import 'package:swatch/api/api.dart';
import 'package:swatch/ext/extension_string.dart';

class DetectionStatusIndicator extends StatelessWidget {
  final SwatchApi _api = SwatchApi();
  final String objectName;

  DetectionStatusIndicator(
    this.objectName, {
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      future: _api.getLatestForLabel(objectName),
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
              isOn ? Icons.check_circle : Icons.radio_button_unchecked,
              size: 16,
              color: isOn ? Colors.greenAccent : Colors.grey,
            ),
            const SizedBox(width: 4.0),
            Text(
              objectName.replaceAll('_', ' ').title(),
              style: TextStyle(
                fontSize: 12,
                color: isOn ? Colors.greenAccent : Colors.grey,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        );
      },
    );
  }
}
