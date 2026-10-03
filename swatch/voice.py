"""Turns a camera's microphone (and, through UniFi Protect talkback, its
speaker) into a Home Assistant Assist satellite.

Each configured voice satellite listens to the PCM an AudioSource decodes
from the camera's RTSP stream -- the same source an audio monitor on that
camera uses, so both share one RTSP connection -- and runs wake word
detection (microWakeWord) on it locally. Home Assistant talks to the
satellite over the ESPHome native API (plaintext), exactly as it would to an
ESPHome voice device: it is added through the ESPHome integration, gets an
assist_satellite entity, and picks the pipeline and wake words in its own UI.

Once a wake word is heard the satellite streams the audio to Home Assistant
(API audio), which runs speech-to-text, the conversation agent and
text-to-speech. The reply's TTS url is then played on the camera's speaker
through UniFi Protect's talkback. Wake words are ignored while a pipeline
runs or a reply plays, so the camera can't wake itself up; only the "stop"
word is listened for then.

Protocol handling follows the Open Home Foundation's linux-voice-assistant
(Apache-2.0), which serves the same API from a Linux box.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# pylint: disable=no-name-in-module
from aioesphomeapi._frame_helper.packets import make_plain_text_packets
from aioesphomeapi.api_pb2 import (
    AuthenticationRequest,
    AuthenticationResponse,
    DeviceInfoRequest,
    DeviceInfoResponse,
    DisconnectRequest,
    DisconnectResponse,
    HelloRequest,
    HelloResponse,
    ListEntitiesDoneResponse,
    ListEntitiesRequest,
    PingRequest,
    PingResponse,
    SubscribeVoiceAssistantRequest,
    VoiceAssistantAnnounceFinished,
    VoiceAssistantAnnounceRequest,
    VoiceAssistantAudio,
    VoiceAssistantConfigurationRequest,
    VoiceAssistantConfigurationResponse,
    VoiceAssistantEventResponse,
    VoiceAssistantRequest,
    VoiceAssistantResponse,
    VoiceAssistantSetConfiguration,
    VoiceAssistantWakeWord,
)
from aioesphomeapi.core import MESSAGE_TYPE_TO_PROTO
from aioesphomeapi.model import VoiceAssistantEventType, VoiceAssistantFeature
from google.protobuf import message
import pymicro_wakeword
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model

from swatch.config import ProtectConfig, VoiceSatelliteConfig

logger = logging.getLogger(__name__)

PROTO_TO_MESSAGE_TYPE = {v: k for k, v in MESSAGE_TYPE_TO_PROTO.items()}

# The api version linux-voice-assistant reports; new enough for every voice
# feature Home Assistant uses (API audio, announcements, start conversation).
API_VERSION_MAJOR = 1
API_VERSION_MINOR = 10

MAX_ACTIVE_WAKE_WORDS = 2
# A run that hears nothing back from Home Assistant for this long (no
# pipeline events) is given up on, so the satellite can't get stuck with
# wake words disabled.
RUN_TIMEOUT_SECONDS = 60.0
# Upper bound on playing one url, so a talkback stream that never finishes
# can't leave the satellite busy (Home Assistant refuses new announcements
# until the previous one reports finished).
PLAY_TIMEOUT_SECONDS = 180.0
DOWNLOAD_TIMEOUT_SECONDS = 30.0
# Most mic audio held back while Home Assistant hasn't confirmed a run yet
# (10 s at 16 kHz, 16-bit mono); anything older is dropped.
MAX_PENDING_MIC_BYTES = 16000 * 2 * 10
BUILTIN_MODELS_DIR = Path(__file__).parent / "voice_models"

SUPPORTED_FEATURES = (
    VoiceAssistantFeature.VOICE_ASSISTANT
    | VoiceAssistantFeature.API_AUDIO
    | VoiceAssistantFeature.ANNOUNCE
    | VoiceAssistantFeature.START_CONVERSATION
)


def _esphome_version() -> str:
    try:
        return version("aioesphomeapi")
    except PackageNotFoundError:
        return "unknown"


# Two-note cues: rising when the satellite starts listening, falling when
# it stops. Generated rather than shipped, short and quiet on purpose.
WAKE_CUE_NOTES = (880.0, 1320.0)
DONE_CUE_NOTES = (1320.0, 880.0)
CUE_SAMPLE_RATE = 24000
CUE_AMPLITUDE = 0.15
CUE_NOTE_SECONDS = 0.09
CUE_GAP_SECONDS = 0.03
CUE_FADE_SECONDS = 0.012


def write_cue(path: str, notes: Iterable[float]) -> None:
    """Write a short, soft sequence of sine notes (with fades, so they
    don't click) to a 16-bit mono WAV file."""
    rate = CUE_SAMPLE_RATE
    note_len = int(CUE_NOTE_SECONDS * rate)
    fade_len = int(CUE_FADE_SECONDS * rate)
    gap = b"\x00\x00" * int(CUE_GAP_SECONDS * rate)
    frames = bytearray()

    for freq in notes:
        for i in range(note_len):
            edge = min(i, note_len - 1 - i)
            envelope = (
                0.5 - 0.5 * math.cos(math.pi * edge / fade_len)
                if edge < fade_len
                else 1.0
            )
            value = CUE_AMPLITUDE * envelope * math.sin(2 * math.pi * freq * i / rate)
            frames += int(value * 32767).to_bytes(2, "little", signed=True)

        frames += gap

    with wave.open(path, "wb") as cue:
        cue.setnchannels(1)
        cue.setsampwidth(2)
        cue.setframerate(rate)
        cue.writeframes(bytes(frames))


def satellite_mac_address(name: str) -> str:
    """A stable, locally-administered MAC for the satellite: Home Assistant's
    ESPHome integration keys the device on it, so it must survive restarts
    but needn't be a real interface's address."""
    digest = bytearray(hashlib.sha256(f"swatch-voice-{name}".encode()).digest()[:6])
    # Set the locally administered bit, clear the multicast bit.
    digest[0] = (digest[0] | 0x02) & 0xFE
    return ":".join(f"{b:02X}" for b in digest)


# -----------------------------------------------------------------------------
# Wake words


@dataclass
class AvailableWakeWord:
    """A microWakeWord model that can be activated."""

    id: str
    wake_word: str
    trained_languages: list[str]
    config_path: Path

    def load(self) -> MicroWakeWord:
        model = MicroWakeWord.from_config(self.config_path)
        # from_config derives the id from the .tflite name; keep ours.
        model.id = self.id
        return model


def find_wake_words(extra_dir: str | None = None) -> dict[str, AvailableWakeWord]:
    """All wake words this satellite can offer Home Assistant: the models
    bundled with pymicro-wakeword, plus any microWakeWord configs in
    extra_dir (which can override a builtin with the same id)."""
    found: dict[str, AvailableWakeWord] = {}
    builtin_dir = Path(pymicro_wakeword.__file__).parent / "models"
    config_paths = [builtin_dir / f"{model.value}.json" for model in Model]

    if extra_dir:
        config_paths += sorted(Path(extra_dir).glob("*.json"))

    for config_path in config_paths:
        try:
            with open(config_path, encoding="utf-8") as config_file:
                config = json.load(config_file)

            if config.get("type") != "micro":
                logger.warning(
                    "Skipping wake word %s: only microWakeWord models are supported",
                    config_path,
                )
                continue

            ww_id = Path(config["model"]).stem
            found[ww_id] = AvailableWakeWord(
                id=ww_id,
                wake_word=config["wake_word"],
                trained_languages=config.get("trained_languages", []),
                config_path=config_path,
            )
        except (OSError, ValueError, KeyError):
            logger.exception("Could not read wake word config %s", config_path)

    return found


# -----------------------------------------------------------------------------
# Replies on the camera speaker


class ProtectSpeaker:
    """Plays audio urls through a UniFi Protect camera's speaker."""

    def __init__(self, config: ProtectConfig) -> None:
        self.config = config
        self._client: Any = None
        self._camera: Any = None
        # The camera plays one thing at a time: cues and replies queue up.
        self._play_lock = asyncio.Lock()

    async def _get_camera(self) -> Any:
        if self._camera is not None:
            return self._camera

        # Imported lazily: uiprotect is a large dependency only voice
        # satellites with a speaker need.
        from uiprotect import (
            ProtectApiClient,
        )  # pylint: disable=import-outside-toplevel

        if self._client is None:
            self._client = ProtectApiClient(
                self.config.host,
                self.config.port,
                self.config.username,
                self.config.password,
                api_key=self.config.api_key,
                verify_ssl=self.config.verify_ssl,
                # Don't cache login sessions to disk inside the container.
                store_sessions=False,
            )

        bootstrap = await self._client.update()

        for camera in bootstrap.cameras.values():
            if self.config.camera in (camera.id, camera.name):
                if not camera.feature_flags.has_speaker:
                    raise ValueError(f"Camera {camera.name} has no speaker")

                volume = self.config.speaker_volume
                if (
                    volume is not None
                    and camera.speaker_settings.speaker_volume != volume
                ):
                    await camera.set_speaker_volume(volume)
                    logger.info("Set %s speaker volume to %s", camera.name, volume)

                self._camera = camera
                return camera

        raise ValueError(f"No UniFi Protect camera named {self.config.camera}")

    async def connect(self) -> None:
        """Log in and find the camera ahead of the first reply, so that one
        isn't delayed (or held up) by the login and the uiprotect import."""
        try:
            camera = await self._get_camera()
            logger.info("UniFi Protect connected, replies play on %s", camera.name)
        except Exception:
            logger.exception("Could not connect to UniFi Protect, will retry on reply")

    async def _download(self, url: str) -> str:
        """Fetch url to a temporary file and return its path.

        Home Assistant serves TTS replies chunked with no length, and the
        ffmpeg/PyAV http reader talkback uses raises "Input/output error" at
        the end of such a response -- after the whole reply has already been
        decoded, so the reply plays but is reported as failed (and the
        Protect session is dropped). Reading a complete local file avoids it.
        """
        import aiohttp  # pylint: disable=import-outside-toplevel

        suffix = Path(url.split("?", 1)[0]).suffix or ".audio"
        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.read()

        fd, path = tempfile.mkstemp(prefix="swatch-reply-", suffix=suffix)
        with os.fdopen(fd, "wb") as audio_file:
            audio_file.write(data)

        return path

    async def play(self, url: str) -> None:
        """Play url on the camera's speaker, returning once it has finished."""
        path = await self._download(url) if url.startswith("http") else url

        try:
            async with self._play_lock:
                await self._play_file(path)
        finally:
            if path != url:
                os.unlink(path)

    async def _play_file(self, path: str) -> None:
        try:
            camera = await self._get_camera()
            playback = camera.play_audio(
                path, blocking=True, use_public_api=self.config.api_key is not None
            )

            try:
                await asyncio.wait_for(playback, PLAY_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning("Talkback on %s didn't finish, stopping it", camera.name)
                await self.stop()
                raise
        except Exception:
            # A stale session or a camera that went away: start over next time.
            self._camera = None
            raise

    async def stop(self) -> None:
        camera = self._camera
        stream = getattr(camera, "talkback_stream", None) if camera else None

        if stream is not None and stream.is_running:
            await camera.stop_audio()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close_session()
            self._client = None
            self._camera = None


# -----------------------------------------------------------------------------
# ESPHome native API (plaintext) framing


class EspHomeConnection(asyncio.Protocol):
    """One Home Assistant connection to a satellite's API port: decodes
    plaintext ESPHome frames and hands the messages to the satellite."""

    def __init__(self, satellite: VoiceSatellite) -> None:
        self.satellite = satellite
        self._buffer = bytearray()
        self._transport: asyncio.Transport | None = None

    # -- framing ------------------------------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.Transport)
        self._transport = transport
        logger.debug(
            "Voice satellite %s: connection from %s",
            self.satellite.name,
            transport.get_extra_info("peername"),
        )

    def connection_lost(self, exc: Exception | None) -> None:
        self._transport = None
        self.satellite.connection_lost(self)

    def data_received(self, data: bytes) -> None:
        self._buffer += data

        while self._transport is not None:
            frame = self._read_frame()

            if frame is None:
                return

            msg_type, payload = frame
            msg_class = MESSAGE_TYPE_TO_PROTO.get(msg_type)

            if msg_class is None:
                logger.debug("Ignoring unknown ESPHome message type %s", msg_type)
                continue

            try:
                self._handle(msg_class.FromString(payload))
            except Exception:
                logger.exception(
                    "Voice satellite %s failed handling %s",
                    self.satellite.name,
                    msg_class.__name__,
                )

    def _read_frame(self) -> tuple[int, bytes] | None:
        """Pop one complete frame (preamble 0x00, varint length, varint type,
        payload) off the buffer, or None if it hasn't fully arrived yet."""
        pos = 0

        if not self._buffer:
            return None

        if self._buffer[0] != 0x00:
            logger.error(
                "Voice satellite %s: bad frame preamble (encrypted client?), "
                "closing connection",
                self.satellite.name,
            )
            self.close()
            return None

        pos = 1
        length, pos = self._read_varuint(pos)

        if length is None:
            return None

        msg_type, pos = self._read_varuint(pos)

        if msg_type is None or len(self._buffer) < pos + length:
            return None

        payload = bytes(self._buffer[pos : pos + length])
        del self._buffer[: pos + length]
        return msg_type, payload

    def _read_varuint(self, pos: int) -> tuple[int | None, int]:
        result = 0
        shift = 0

        while pos < len(self._buffer):
            byte = self._buffer[pos]
            pos += 1
            result |= (byte & 0x7F) << shift

            if not byte & 0x80:
                return result, pos

            shift += 7

        return None, pos

    def send(self, msgs: Iterable[message.Message]) -> None:
        """Send messages; must be called on the event loop thread."""
        if self._transport is None:
            return

        packets = [
            (PROTO_TO_MESSAGE_TYPE[msg.__class__], msg.SerializeToString())
            for msg in msgs
        ]

        if packets:
            self._transport.writelines(make_plain_text_packets(packets))

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    # -- connection-level messages ------------------------------------------

    def _handle(self, msg: message.Message) -> None:
        if isinstance(msg, HelloRequest):
            self.satellite.connection_ready(self)
            self.send(
                [
                    HelloResponse(
                        api_version_major=API_VERSION_MAJOR,
                        api_version_minor=API_VERSION_MINOR,
                        server_info=f"swatch {_esphome_version()}",
                        name=self.satellite.device_name,
                    )
                ]
            )
        elif isinstance(msg, AuthenticationRequest):
            # No password: any (legacy) authentication attempt succeeds.
            self.send([AuthenticationResponse()])
        elif isinstance(msg, PingRequest):
            self.send([PingResponse()])
        elif isinstance(msg, DisconnectRequest):
            self.send([DisconnectResponse()])
            self.close()
        else:
            self.send(self.satellite.handle_message(self, msg))


# -----------------------------------------------------------------------------
# The satellite


class VoiceSatellite:
    """A voice satellite: an AudioSource consumer on the audio thread, and an
    ESPHome API server on the voice event loop.

    State that the audio thread acts on (whether to stream audio, whether a
    wake word may start a run) is only ever *changed* on the event loop; the
    audio thread just reads those flags and schedules anything it detects
    back onto the loop with call_soon_threadsafe.
    """

    def __init__(
        self,
        config: VoiceSatelliteConfig,
        speaker: ProtectSpeaker | None = None,
    ) -> None:
        assert config.name is not None
        self.config = config
        self.name: str = config.name
        self.device_name = f"swatch-{self.name}".replace("_", "-").lower()
        self.friendly_name = config.friendly_name or self.name.replace("_", " ").title()
        self.mac_address = satellite_mac_address(self.name)
        self.speaker = (
            speaker
            if speaker is not None
            else (ProtectSpeaker(config.protect) if config.protect else None)
        )

        self.loop: asyncio.AbstractEventLoop | None = None
        self.port = config.port
        self._server: asyncio.base_events.Server | None = None
        self._connection: EspHomeConnection | None = None
        self._va_subscribed = False

        # -- wake words (models are only *used* on the audio thread) --------
        self.available_wake_words = find_wake_words(config.wake_word_dir)
        self._ww_lock = threading.Lock()
        self._features = MicroWakeWordFeatures()
        self._loaded_wake_words: dict[str, MicroWakeWord] = {}
        self.active_wake_words: list[MicroWakeWord] = []
        self.set_active_wake_words(config.wake_words)
        self._stop_word: MicroWakeWord | None = (
            MicroWakeWord.from_config(BUILTIN_MODELS_DIR / "stop.json")
            if config.stop_word
            else None
        )
        self._last_wake_time = 0.0

        # -- pipeline state (changed on the loop only) ----------------------
        self.streaming_audio = False
        self.pipeline_active = False
        self.speaking = False
        self._tts_url: str | None = None
        self._tts_played = False
        self._continue_conversation = False
        self._announcing = False
        self._play_task: asyncio.Task[None] | None = None
        self._last_run_activity = 0.0
        self._accepted = False
        self._pending_mic: list[bytes] = []
        self._pending_mic_bytes = 0
        self._done_cue_pending = False
        self._cue_dir: str | None = None
        self._cues: dict[str, str] = {}

        if self.speaker is not None and (config.wake_sound or config.done_sound):
            self._cue_dir = tempfile.mkdtemp(prefix=f"swatch-{self.name}-cues-")

            for cue, enabled, notes in (
                ("wake", config.wake_sound, WAKE_CUE_NOTES),
                ("done", config.done_sound, DONE_CUE_NOTES),
            ):
                if enabled:
                    self._cues[cue] = os.path.join(self._cue_dir, f"{cue}.wav")
                    write_cue(self._cues[cue], notes)

    # -- lifecycle (loop thread) --------------------------------------------

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._server = await loop.create_server(
            lambda: EspHomeConnection(self), host="0.0.0.0", port=self.config.port
        )
        # The actual port, in case the config asked for any free one (0).
        self.port = self._server.sockets[0].getsockname()[1]
        logger.info(
            "Voice satellite %s listening for Home Assistant on port %s "
            "(wake words: %s)",
            self.name,
            self.port,
            ", ".join(ww.id for ww in self.active_wake_words) or "none",
        )

        if self.speaker is not None:
            loop.create_task(self.speaker.connect())

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()

        if self._connection is not None:
            self._connection.close()

        if self.speaker is not None:
            try:
                await self.speaker.close()
            except Exception:
                logger.debug("Error closing UniFi Protect session", exc_info=True)

        if self._cue_dir is not None:
            shutil.rmtree(self._cue_dir, ignore_errors=True)

    # -- wake word configuration --------------------------------------------

    def set_active_wake_words(self, ids: Iterable[str]) -> None:
        """Activate up to MAX_ACTIVE_WAKE_WORDS of the available wake words,
        loading their models on first use; unknown ids are skipped."""
        active: list[MicroWakeWord] = []

        for ww_id in ids:
            if len(active) >= MAX_ACTIVE_WAKE_WORDS:
                break

            available = self.available_wake_words.get(ww_id)

            if available is None:
                logger.warning(
                    "Voice satellite %s: unknown wake word %s", self.name, ww_id
                )
                continue

            if ww_id not in self._loaded_wake_words:
                model = available.load()

                if self.config.wake_word_threshold is not None:
                    model.probability_cutoff = self.config.wake_word_threshold

                self._loaded_wake_words[ww_id] = model

            active.append(self._loaded_wake_words[ww_id])

        with self._ww_lock:
            self.active_wake_words = active

    # -- audio (AudioSource thread) -----------------------------------------

    def on_audio(self, raw: bytes) -> None:
        if self.streaming_audio:
            self._call_soon(self._mic_chunk, raw)

        features = list(self._features.process_streaming(raw))

        if not features:
            return

        detected: MicroWakeWord | None = None
        stopped = False

        with self._ww_lock:
            # Run every model on every feature, even while a pipeline is
            # active, so each one's sliding window stays in step with the
            # audio -- otherwise it would resume on stale context.
            for wake_word in self.active_wake_words:
                for feature in features:
                    if wake_word.process_streaming(feature) and detected is None:
                        detected = wake_word

            if self._stop_word is not None:
                for feature in features:
                    if self._stop_word.process_streaming(feature):
                        stopped = True

        if detected is not None and not self.pipeline_active:
            now = time.monotonic()

            if now - self._last_wake_time > self.config.refractory_seconds:
                self._last_wake_time = now
                logger.info(
                    "Voice satellite %s: heard wake word %s",
                    self.name,
                    detected.wake_word,
                )
                self._call_soon(self.wakeup, detected.wake_word)

        if stopped and self.speaking:
            logger.info("Voice satellite %s: heard stop word", self.name)
            self._call_soon(self.stop_speaking)

    def on_stream_end(self) -> None:
        with self._ww_lock:
            self._features.reset()

            for wake_word in self.active_wake_words:
                wake_word.reset()

    def on_disconnect(self) -> None:
        """The camera's stream dropped: a run that was still listening won't
        hear anything more, so end it rather than leave it hanging."""
        self._call_soon(self._end_streaming_run)

    def _call_soon(self, callback: Any, *args: Any) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(callback, *args)

    # -- Home Assistant messages (loop thread) ------------------------------

    def _send(self, msgs: Iterable[message.Message]) -> None:
        if self._connection is not None:
            self._connection.send(msgs)

    def connection_ready(self, connection: EspHomeConnection) -> None:
        if self._connection is connection:
            return

        # Newest connection wins, e.g. Home Assistant reconnecting before
        # noticing its old socket died.
        if self._connection is not None:
            old = self._connection
            self._connection = None
            old.close()

        self._connection = connection
        logger.info("Voice satellite %s: Home Assistant connected", self.name)

    def connection_lost(self, connection: EspHomeConnection) -> None:
        if connection is not self._connection:
            return

        logger.info("Voice satellite %s: Home Assistant disconnected", self.name)
        self._connection = None
        self._va_subscribed = False
        self.streaming_audio = False
        self.pipeline_active = False
        self._continue_conversation = False

        if self._play_task is not None:
            self._play_task.cancel()

    def handle_message(
        self, connection: EspHomeConnection, msg: message.Message
    ) -> list[message.Message]:
        if isinstance(msg, DeviceInfoRequest):
            return [
                DeviceInfoResponse(
                    uses_password=False,
                    name=self.device_name,
                    friendly_name=self.friendly_name,
                    mac_address=self.mac_address,
                    esphome_version=_esphome_version(),
                    manufacturer="MaiorDomus",
                    model="Swatch Voice Satellite",
                    project_name="MaiorDomus.Swatch",
                    project_version="voice-satellite",
                    voice_assistant_feature_flags=SUPPORTED_FEATURES,
                )
            ]

        if isinstance(msg, ListEntitiesRequest):
            # No entities of our own: the assist_satellite entity (and its
            # pipeline/wake word selects) come from the feature flags.
            return [ListEntitiesDoneResponse()]

        if isinstance(msg, SubscribeVoiceAssistantRequest):
            self._va_subscribed = msg.subscribe
            return []

        if isinstance(msg, VoiceAssistantConfigurationRequest):
            return [self._configuration_response()]

        if isinstance(msg, VoiceAssistantSetConfiguration):
            self.set_active_wake_words(msg.active_wake_words)
            logger.info(
                "Voice satellite %s: wake words set to %s",
                self.name,
                ", ".join(ww.id for ww in self.active_wake_words) or "none",
            )
            return []

        if isinstance(msg, VoiceAssistantResponse):
            if msg.error:
                logger.warning(
                    "Voice satellite %s: Home Assistant refused the run", self.name
                )
                self._finish_run()
            else:
                self._run_accepted()

            return []

        if isinstance(msg, VoiceAssistantEventResponse):
            data = {arg.name: arg.value for arg in msg.data}
            self.handle_voice_event(VoiceAssistantEventType(msg.event_type), data)
            return []

        if isinstance(msg, VoiceAssistantAnnounceRequest):
            self.handle_announcement(msg)
            return []

        return []

    def _configuration_response(self) -> VoiceAssistantConfigurationResponse:
        return VoiceAssistantConfigurationResponse(
            available_wake_words=[
                VoiceAssistantWakeWord(
                    id=ww.id,
                    wake_word=ww.wake_word,
                    trained_languages=ww.trained_languages,
                )
                for ww in self.available_wake_words.values()
            ],
            active_wake_words=[ww.id for ww in self.active_wake_words],
            max_active_wake_words=MAX_ACTIVE_WAKE_WORDS,
        )

    # -- pipeline (loop thread) ---------------------------------------------

    def wakeup(self, wake_word_phrase: str) -> None:
        if self.pipeline_active:
            return

        if self._connection is None or not self._va_subscribed:
            logger.info(
                "Voice satellite %s: wake word ignored, Home Assistant isn't connected",
                self.name,
            )
            return

        self._start_run(wake_word_phrase)

    def _start_run(self, wake_word_phrase: str = "") -> None:
        self._last_run_activity = time.monotonic()
        self._pending_mic.clear()
        self._pending_mic_bytes = 0
        self._accepted = False
        self.pipeline_active = True
        self._tts_url = None
        self._tts_played = False
        self._continue_conversation = False
        self._send(
            [VoiceAssistantRequest(start=True, wake_word_phrase=wake_word_phrase)]
        )
        self.streaming_audio = True
        self._done_cue_pending = True
        self._play_cue("wake")

    def _mic_chunk(self, raw: bytes) -> None:
        """Send mic audio for the current run -- or, until Home Assistant has
        confirmed the run, hold it back. Home Assistant clears its audio
        queue when it starts the run, so audio sent straight after the start
        request can be thrown away, clipping the first word of the command;
        ESPHome devices wait for the confirmation the same way."""
        if not self.streaming_audio:
            return

        if self._accepted:
            self._send([VoiceAssistantAudio(data=raw)])
            return

        self._pending_mic.append(raw)
        self._pending_mic_bytes += len(raw)

        while self._pending_mic_bytes > MAX_PENDING_MIC_BYTES:
            self._pending_mic_bytes -= len(self._pending_mic.pop(0))

    def _run_accepted(self) -> None:
        self._accepted = True
        pending, self._pending_mic = self._pending_mic, []
        self._pending_mic_bytes = 0

        if self.streaming_audio and pending:
            self._send([VoiceAssistantAudio(data=raw) for raw in pending])

    def _end_streaming_run(self) -> None:
        if self.streaming_audio:
            self.streaming_audio = False
            self._send([VoiceAssistantRequest(start=False)])

    def handle_voice_event(
        self, event_type: VoiceAssistantEventType, data: dict[str, str]
    ) -> None:
        logger.debug("Voice satellite %s: %s %s", self.name, event_type.name, data)
        self._last_run_activity = time.monotonic()

        if event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START:
            self._tts_url = data.get("url")
            self._tts_played = False
            self._continue_conversation = False
            self.pipeline_active = True
        elif event_type in (
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
            VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
        ):
            self.streaming_audio = False

            if self._done_cue_pending:
                self._done_cue_pending = False
                self._play_cue("done")

            if text := data.get("text"):
                logger.info("Voice satellite %s: heard %r", self.name, text)
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_PROGRESS:
            if data.get("tts_start_streaming") == "1":
                # The reply is ready to stream before the agent has finished.
                self._play_tts()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END:
            self._continue_conversation = data.get("continue_conversation") == "1"
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END:
            self._tts_url = data.get("url") or self._tts_url
            self._play_tts()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END:
            self.streaming_audio = False

            if not self._tts_played:
                self._finish_run()
        elif event_type == VoiceAssistantEventType.VOICE_ASSISTANT_ERROR:
            logger.warning(
                "Voice satellite %s: pipeline error %s: %s",
                self.name,
                data.get("code"),
                data.get("message"),
            )

    def handle_announcement(self, msg: VoiceAssistantAnnounceRequest) -> None:
        urls = [u for u in (msg.preannounce_media_id, msg.media_id) if u]
        logger.info("Voice satellite %s: announcing %r", self.name, msg.text)

        if self._play_task is not None:
            self._play_task.cancel()

        self.pipeline_active = True
        self.streaming_audio = False
        self._announcing = True
        self._continue_conversation = msg.start_conversation
        self._play(urls)

    def _play_cue(self, cue: str) -> None:
        """Play a listening cue without waiting for it: the mic keeps
        streaming meanwhile, and a reply queues up behind it."""
        path = self._cues.get(cue)

        if path is not None and self.speaker is not None and self.loop is not None:
            self.loop.create_task(self._play_cue_file(path))

    async def _play_cue_file(self, path: str) -> None:
        assert self.speaker is not None

        try:
            await self.speaker.play(path)
        except Exception:
            logger.warning(
                "Voice satellite %s: failed to play cue", self.name, exc_info=True
            )

    def _play_tts(self) -> None:
        if not self._tts_url or self._tts_played:
            return

        self._tts_played = True
        self._play([self._tts_url])

    def _play(self, urls: list[str]) -> None:
        assert self.loop is not None
        self.speaking = True
        self._play_task = self.loop.create_task(self._play_urls(urls))

    async def _play_urls(self, urls: list[str]) -> None:
        try:
            if self.speaker is None:
                logger.info(
                    "Voice satellite %s: no speaker configured, not playing reply",
                    self.name,
                )
            else:
                for url in urls:
                    logger.info("Voice satellite %s: playing reply", self.name)
                    await self.speaker.play(url)

            # Let the speaker's last words die out before the mic reopens.
            await asyncio.sleep(self.config.echo_guard_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Voice satellite %s: failed to play reply", self.name)
        finally:
            self.speaking = False

        self._reply_finished()

    def stop_speaking(self) -> None:
        if not self.speaking:
            return

        self._continue_conversation = False

        if self.speaker is not None and self.loop is not None:
            self.loop.create_task(self.speaker.stop())

    def _reply_finished(self) -> None:
        # Home Assistant takes this as "done playing" for TTS replies as well
        # as announcements; it's what returns the satellite entity to idle.
        logger.info("Voice satellite %s: reply finished", self.name)
        self._send([VoiceAssistantAnnounceFinished()])
        self._announcing = False

        if self._continue_conversation and self._connection is not None:
            self._start_run()
        else:
            self._finish_run()

    def check_run_timeout(self) -> None:
        """Give up on a run Home Assistant has gone quiet on (a reply that
        is still playing doesn't count -- that has its own end)."""
        if (
            self.pipeline_active
            and not self.speaking
            and time.monotonic() - self._last_run_activity > RUN_TIMEOUT_SECONDS
        ):
            logger.warning(
                "Voice satellite %s: no response from Home Assistant, ending run",
                self.name,
            )
            self._end_streaming_run()
            self._finish_run()

    def _finish_run(self) -> None:
        self._done_cue_pending = False
        self.streaming_audio = False
        self.pipeline_active = False
        self._continue_conversation = False


class VoiceSatelliteServer(threading.Thread):
    """Runs every voice satellite's ESPHome API server on one asyncio event
    loop, alongside swatch's (threaded) detection and Flask code."""

    def __init__(
        self,
        satellites: Iterable[VoiceSatellite],
        stop_event: multiprocessing.Event,
    ) -> None:
        threading.Thread.__init__(self)
        self.name = "voice_satellites"
        self.satellites = list(satellites)
        self.stop_event = stop_event
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()

    def run(self) -> None:
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._main())
        except Exception:
            logger.exception("Voice satellite server crashed")
        finally:
            self.ready.set()
            self.loop.close()

    async def _main(self) -> None:
        for satellite in self.satellites:
            try:
                await satellite.start(self.loop)
            except OSError:
                logger.exception(
                    "Voice satellite %s could not listen on port %s",
                    satellite.name,
                    satellite.config.port,
                )

        self.ready.set()

        while not self.stop_event.is_set():
            await asyncio.sleep(0.5)

            for satellite in self.satellites:
                satellite.check_run_timeout()

        for satellite in self.satellites:
            await satellite.stop()
