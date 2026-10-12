# Bundled wake word models

`stop.tflite` / `stop.json` are the microWakeWord "stop" model by Kevin Ahrendt,
as shipped by the Open Home Foundation's
[linux-voice-assistant](https://github.com/OHF-Voice/linux-voice-assistant)
(Apache-2.0). The other built-in wake words (okay_nabu, hey_jarvis, hey_mycroft,
alexa) come with the `pymicro-wakeword` package.

`melspectrogram.onnx` and `embedding_model.onnx` are openWakeWord's shared feature
models, from [openWakeWord](https://github.com/dscripka/openWakeWord) v0.6.0
(Apache-2.0; the embedding model is Google's
[speech_embedding](https://www.kaggle.com/models/google/speech-embedding), Apache-2.0).
Every openWakeWord wake word model runs on top of them.
