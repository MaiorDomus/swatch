import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:swatch/api/api.dart';
import 'package:swatch/ext/extension_string.dart';
import 'package:swatch/models/config.dart';
import 'package:swatch/models/detection_event.dart';

class _TransitionEvent {
  final String label;
  final bool isOn;
  final double timestamp;

  _TransitionEvent(this.label, this.isOn, this.timestamp);
}

/// Shows the last N times any object or audio monitor turned on or off, in
/// one combined table -- each detection session (a start_time and, once
/// finished, an end_time) unpacks into up to two timeline entries.
class TransitionHistoryComponent extends StatelessWidget {
  final SwatchApi _api = SwatchApi();
  final Config config;
  final int limit;

  TransitionHistoryComponent(
    this.config, {
    this.limit = 5,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final labels = [...config.objects.keys, ...config.audioMonitors];

    return Card(
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(8.0)),
      ),
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Recent Activity",
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 8.0),
              child: FutureBuilder<List<_TransitionEvent>>(
                future: _loadEvents(labels),
                builder: (
                  context,
                  AsyncSnapshot<List<_TransitionEvent>> snapshot,
                ) {
                  if (!snapshot.hasData) {
                    return const SizedBox(
                      height: 40,
                      child: Center(
                        child: SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ),
                    );
                  }

                  final events = snapshot.data!;
                  if (events.isEmpty) {
                    return const Text("No activity recorded yet.");
                  }

                  return _buildTable(events);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTable(final List<_TransitionEvent> events) {
    const headerStyle = TextStyle(fontWeight: FontWeight.bold);
    const cellPadding = EdgeInsets.symmetric(vertical: 4.0, horizontal: 4.0);
    final format = DateFormat("MM/dd HH:mm:ss");

    return Table(
      columnWidths: const {
        0: FlexColumnWidth(2),
        1: FlexColumnWidth(2),
        2: FlexColumnWidth(1),
      },
      children: [
        const TableRow(
          children: [
            Padding(padding: cellPadding, child: Text("Time", style: headerStyle)),
            Padding(padding: cellPadding, child: Text("Sensor", style: headerStyle)),
            Padding(padding: cellPadding, child: Text("State", style: headerStyle)),
          ],
        ),
        ...events.map(
          (event) => TableRow(
            children: [
              Padding(
                padding: cellPadding,
                child: Text(
                  format.format(
                    DateTime.fromMillisecondsSinceEpoch(
                      (event.timestamp * 1000).round(),
                    ),
                  ),
                ),
              ),
              Padding(
                padding: cellPadding,
                child: Text(event.label.replaceAll('_', ' ').title()),
              ),
              Padding(
                padding: cellPadding,
                child: Text(
                  event.isOn ? "On" : "Off",
                  style: TextStyle(
                    color: event.isOn ? Colors.greenAccent : Colors.grey,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Future<List<_TransitionEvent>> _loadEvents(
    final List<String> labels,
  ) async {
    final results = await Future.wait(
      labels.map((label) => _api.getDetections(label: label, limit: limit * 2)),
    );

    final events = <_TransitionEvent>[];
    for (var i = 0; i < labels.length; i++) {
      for (final DetectionEvent session in results[i]) {
        events.add(_TransitionEvent(labels[i], true, session.startTime));

        if (session.endTime != -1) {
          events.add(_TransitionEvent(labels[i], false, session.endTime));
        }
      }
    }

    events.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return events.take(limit).toList();
  }
}
