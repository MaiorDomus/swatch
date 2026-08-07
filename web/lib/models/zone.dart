class Zone {

  String coordinates = "";
  String name = "";
  List<String> objects = [];

  Zone(final Map<String, dynamic> json) {
    coordinates = json["coordinates"];
    objects = List<String>.from(json["objects"]);
  }
}