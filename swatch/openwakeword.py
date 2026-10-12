"""openWakeWord models for voice satellites.

openWakeWord (https://github.com/dscripka/openWakeWord) is the other common
wake word engine next to microWakeWord: custom models trained with its tools
(or with forks such as EchoMuse's forge) are small ONNX classifiers that sit
on top of two shared feature models, a mel spectrogram and Google's
speech_embedding. Both feature models are bundled in voice_models/ (see
NOTICE.md there); the per-wake-word classifier comes from wake_word_dir.

This is a small streaming reimplementation of openwakeword's own
AudioFeatures/Model.predict on onnxruntime, so swatch doesn't pull in the
openwakeword package (whose tflite-runtime dependency has no wheels for
current Python versions). It mirrors upstream's numbers: 80 ms (1280 sample)
steps, each adding 8 mel frames and one 96-value embedding of the last 76
mel frames, and the classifier scoring the newest n embeddings (n from its
input shape, usually 16 = 1.28 s).
"""

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

BUILTIN_MODELS_DIR = Path(__file__).parent / "voice_models"
MELSPEC_MODEL = BUILTIN_MODELS_DIR / "melspectrogram.onnx"
EMBEDDING_MODEL = BUILTIN_MODELS_DIR / "embedding_model.onnx"

SAMPLES_PER_STEP = 1280  # 80 ms at 16 kHz
PHASE_SAMPLES = 160  # 10 ms, one mel frame
# The mel spectrogram of a step also needs the 3 frames (160 samples each)
# before it, so its windows line up with the previous step's.
MELSPEC_CONTEXT_SAMPLES = 160 * 3
EMBEDDING_WINDOW = 76  # mel frames per embedding
MEL_BINS = 32
# Like upstream, ignore the first few scores after a (re)start: the
# classifier's window still holds embeddings from before the stream.
WARMUP_STEPS = 5


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Several satellites (and audio monitors) share the machine; one thread
    # per model is plenty for 12.5 steps a second.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class _FeaturePipeline:
    """One stream of speech_embedding features, its 80 ms steps starting
    offset samples into the audio."""

    def __init__(self, features: "OpenWakeWordFeatures", offset: int) -> None:
        self._features = features
        self._offset = offset
        self.reset()

    def reset(self) -> None:
        self._samples = np.zeros(0, dtype=np.int16)
        self._skip = self._offset
        # Upstream starts the mel buffer as ones, so the first embeddings
        # are computed against that rather than against nothing.
        self._mels = np.ones((EMBEDDING_WINDOW, MEL_BINS), dtype=np.float32)

    def process(self, samples: np.ndarray) -> list[np.ndarray]:
        if self._skip:
            skipped = samples[: self._skip]
            samples = samples[self._skip :]
            self._skip -= len(skipped)

        self._samples = np.concatenate((self._samples, samples))
        embeddings: list[np.ndarray] = []

        while len(self._samples) >= MELSPEC_CONTEXT_SAMPLES + SAMPLES_PER_STEP:
            chunk = self._samples[: MELSPEC_CONTEXT_SAMPLES + SAMPLES_PER_STEP]
            self._samples = self._samples[SAMPLES_PER_STEP:]
            mels = self._features.melspectrogram(chunk)
            self._mels = np.vstack((self._mels, mels))[-EMBEDDING_WINDOW:]
            embeddings.append(self._features.embedding(self._mels))

        return embeddings


class OpenWakeWordFeatures:
    """Turns 16 kHz, 16-bit mono PCM into speech_embedding features, one
    96-value vector per 80 ms per phase.

    A model's score depends a lot on where the 80 ms steps fall relative to
    the wake word: on real camera recordings the same utterance scored 0.01
    with one alignment and 0.87 with another 20 ms later. So the features
    are computed for several alignments (phases), PHASE_SAMPLES apart, and a
    model triggers on whichever scores highest. Four phases 10 ms apart
    caught every alignment of every test recording; one caught 79%.
    """

    def __init__(self, phases: int = 4) -> None:
        self._melspec = _session(MELSPEC_MODEL)
        self._melspec_input = self._melspec.get_inputs()[0].name
        self._embedding = _session(EMBEDDING_MODEL)
        self._embedding_input = self._embedding.get_inputs()[0].name
        self._pipelines = [
            _FeaturePipeline(self, phase * PHASE_SAMPLES) for phase in range(phases)
        ]

    def reset(self) -> None:
        for pipeline in self._pipelines:
            pipeline.reset()

    def melspectrogram(self, chunk: np.ndarray) -> np.ndarray:
        mels = self._melspec.run(
            None, {self._melspec_input: chunk.astype(np.float32)[None, :]}
        )[0]
        # Upstream's transform, bringing the ONNX mel spectrogram close to
        # the TensorFlow one speech_embedding was trained on.
        return np.asarray(np.squeeze(mels) / 10 + 2)

    def embedding(self, mels: np.ndarray) -> np.ndarray:
        embedding = self._embedding.run(
            None, {self._embedding_input: mels[None, :, :, None]}
        )[0]
        return np.asarray(embedding).reshape(-1)

    def process_streaming(self, raw: bytes) -> list[tuple[int, np.ndarray]]:
        """Add audio; returns (phase, embedding) for every completed step."""
        samples = np.frombuffer(raw, dtype=np.int16)
        return [
            (phase, embedding)
            for phase, pipeline in enumerate(self._pipelines)
            for embedding in pipeline.process(samples)
        ]


class OpenWakeWord:
    """One openWakeWord classifier, scoring the newest embeddings of each
    phase. Has the same shape as pymicro_wakeword's MicroWakeWord where
    VoiceSatellite uses it (id, wake_word, probability_cutoff,
    process_streaming, reset)."""

    def __init__(
        self,
        id: str,  # pylint: disable=redefined-builtin
        wake_word: str,
        model_path: Path,
        probability_cutoff: float = 0.5,
        trained_languages: list[str] | None = None,
    ) -> None:
        self.id = id
        self.wake_word = wake_word
        self.probability_cutoff = probability_cutoff
        self.trained_languages = trained_languages or []
        self._session = _session(model_path)
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self.window = int(model_input.shape[1])
        self.last_probability = 0.0
        self.reset()

    def reset(self) -> None:
        self._embeddings: dict[int, deque[np.ndarray]] = {}
        self._steps: dict[int, int] = {}

    def process_streaming(self, feature: tuple[int, np.ndarray]) -> bool:
        """Add one (phase, embedding); True when the wake word was heard."""
        phase, embedding = feature
        embeddings = self._embeddings.setdefault(phase, deque(maxlen=self.window))
        embeddings.append(embedding)
        steps = self._steps[phase] = self._steps.get(phase, 0) + 1

        if len(embeddings) < self.window or steps <= WARMUP_STEPS:
            self.last_probability = 0.0
            return False

        window = np.stack(embeddings)[None, :, :].astype(np.float32)
        self.last_probability = float(
            np.asarray(self._session.run(None, {self._input_name: window})[0]).reshape(
                -1
            )[0]
        )
        return self.last_probability >= self.probability_cutoff
