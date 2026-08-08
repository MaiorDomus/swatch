import 'package:flutter/material.dart';
import 'package:swatch/theme/theme_helper.dart';

/// A small hand-rolled YAML highlighter (comments, keys, quoted strings,
/// numbers/booleans/null) for the read-only config.yaml viewer -- doesn't
/// pull in a full highlight.js-style package for one file type. Doesn't
/// attempt to split inline trailing comments (e.g. `key: value  # note`)
/// from the value, only whole-line comments (`# note`); config.yaml's own
/// convention throughout this project is whole-line comments, and trying
/// to detect an inline `#` risks misfiring on a value that legitimately
/// contains one (e.g. a "#ffffff" color string).
const _commentColor = Color(0xFF9E9E9E);
final _keyColor = SwatchColors.getPrimaryColor();
const _stringColor = Colors.orangeAccent;
const _scalarColor = Colors.lightBlueAccent;
const _plainColor = Colors.white70;

final _fullLineCommentPattern = RegExp(r'^(\s*)#(.*)$');
final _keyValuePattern = RegExp(r'^(\s*(?:-\s+)?)([\w.\-]+)(\s*:\s*)(.*)$');
final _listItemPattern = RegExp(r'^(\s*-\s+)(.*)$');
final _scalarValuePattern = RegExp(
  r'^-?\d+(\.\d+)?$|^(true|false|null|~)$',
  caseSensitive: false,
);

/// One TextSpan per source line, rather than a single combined tree for
/// the whole file: SelectableText.rich on a very large multi-line span
/// tree hits a known class of Flutter web selection/focus crash on the
/// old Flutter version this project is pinned to (pubspec.yaml's Dart SDK
/// constraint rules out anything newer) -- confirmed against a real
/// deployment (selecting the highlighted text threw "can't access
/// property 'focus', this.a.c is null"), and reproduced only after
/// switching from a single plain SelectableText to SelectableText.rich
/// with a large tree. Rendering one small SelectableText.rich per line
/// keeps each span tree tiny and avoids it, at the cost of selection no
/// longer dragging across line boundaries.
List<TextSpan> highlightYamlLines(final String source) {
  return source
      .split('\n')
      .map((line) => TextSpan(
            children: _highlightLine(line),
            style: const TextStyle(color: _plainColor),
          ))
      .toList();
}

List<TextSpan> _highlightLine(final String line) {
  final commentMatch = _fullLineCommentPattern.firstMatch(line);
  if (commentMatch != null) {
    return [
      TextSpan(text: commentMatch.group(1)),
      TextSpan(
        text: '#${commentMatch.group(2)}',
        style: const TextStyle(
          color: _commentColor,
          fontStyle: FontStyle.italic,
        ),
      ),
    ];
  }

  final keyValueMatch = _keyValuePattern.firstMatch(line);
  if (keyValueMatch != null) {
    final prefix = keyValueMatch.group(1)!;
    final key = keyValueMatch.group(2)!;
    final colon = keyValueMatch.group(3)!;
    final value = keyValueMatch.group(4)!;

    return [
      TextSpan(text: prefix),
      TextSpan(
        text: key,
        style: TextStyle(color: _keyColor, fontWeight: FontWeight.bold),
      ),
      TextSpan(text: colon),
      ..._highlightValue(value),
    ];
  }

  final listItemMatch = _listItemPattern.firstMatch(line);
  if (listItemMatch != null) {
    return [
      TextSpan(text: listItemMatch.group(1)),
      ..._highlightValue(listItemMatch.group(2)!),
    ];
  }

  return [TextSpan(text: line)];
}

List<TextSpan> _highlightValue(final String value) {
  if (value.isEmpty) {
    return [TextSpan(text: value)];
  }

  final isQuoted = (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"));

  if (isQuoted) {
    return [TextSpan(text: value, style: const TextStyle(color: _stringColor))];
  }

  if (_scalarValuePattern.hasMatch(value.trim())) {
    return [TextSpan(text: value, style: const TextStyle(color: _scalarColor))];
  }

  return [TextSpan(text: value)];
}
