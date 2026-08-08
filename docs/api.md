# HTTP API

## `/api/config`

Returns JSON config

## `/api/colortest/values`

```json
{ "test_image": test_image.jpg } // multipart file
```

Upload image to get info about colors detected in the image.
This should make it easier to add objects with known colors as
the program sees them.

## `/api/colortest/mask`

```json
{ "test_image": test_image.jpg } // multipart file
{
    "color_lower": "0, 0, 0",
    "color_upper": "255, 255, 255"
} // fields
```

Upload image to get info about colors detected in the image.
This should make it easier to add objects with known colors as
the program sees them.

## `/api/<camera_name>/detect`

```json
{
    "imageUrl": "http://some_camera_image.jpg"
} // json
```

Take the `camera_name` config and `imageUrl` to run detection and see which objects are detected.

## `/api/<label>/latest`

Returns the latest results for the given label. `all` can be passed to get a result for
all labels. A label can be either an object name (from `objects`/zone `objects`) or an
`audio_monitors` name — both share this endpoint and are merged together under `all`.

Object detection result:

```json
{
    "trash_can":{
        "area":2818,
        "camera_name":"front_doorbell_cam",
        "result":true,
        "solidity":0.87,
        "variant":"overcast"
    }
} // json
```

Audio monitor result:

```json
{
    "kitchen_hood":{
        "result":true
    }
} // json
```

## `/api/detections`

Returns on/off history (each row spans a `start_time` to an `end_time`, or `end_time: null`
if still ongoing). Query params, all optional: `label` (object or `audio_monitors` name --
both share this history the same way they share `/<label>/latest`), `camera`, `zone`,
`limit` (default 100), `after`/`before` (unix timestamps, filters on `start_time`).

```json
[
    {
        "id":"kitchen_hood.ab12cd",
        "label":"kitchen_hood",
        "camera":"",
        "zone":"",
        "color_variant":"audio",
        "top_area":0,
        "start_time":1786183359.308,
        "end_time":1786183400.512
    }
] // json
```

Object detection rows populate `camera`/`zone`/`top_area`/`color_variant` from the match;
audio monitor rows leave those blank/zero since there's no equivalent.

## `/api/<camera_name>/snapshot.jpg`

Returns a snapshot of the latest image for the <camera_name>.

## `/api/<camera_name>/<zone_name>/snapshot.jpg`

Returns a snapshot of the latest snapshot for the <zone_name> of the <camera_name>.

## `/api/<camera_name>/detection.jpg`

Returns a snapshot of the latest detection for the <camera_name>.
