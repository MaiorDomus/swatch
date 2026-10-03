"""Tests for voice satellites."""

import asyncio
import json
import multiprocessing
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any
from unittest import mock

from aioesphomeapi import APIClient
from aioesphomeapi.model import VoiceAssistantEventType, VoiceAssistantFeature

from swatch.config import ProtectConfig, VoiceSatelliteConfig
from swatch.voice import (
    EspHomeConnection,
    ProtectSpeaker,
    VoiceSatellite,
    VoiceSatelliteServer,
    find_wake_words,
    satellite_mac_address,
    write_cue,
)


def make_config(**overrides: Any) -> VoiceSatelliteConfig:
    return VoiceSatelliteConfig.model_validate(
        {"name": "living_room", "rtsp_url": "unused", "port": 0, **overrides}
    )


class FakeWakeWord:
    """Stands in for a MicroWakeWord: fires on the features it's told to."""

    def __init__(self, ww_id: str, wake_word: str) -> None:
        self.id = ww_id
        self.wake_word = wake_word
        self.fire = False

    def process_streaming(self, _features: Any) -> bool:
        return self.fire

    def reset(self) -> None:
        pass


class FakeSpeaker:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.stopped = False
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def play(self, url: str) -> None:
        self.played.append(url)

    async def stop(self) -> None:
        self.stopped = True

    async def close(self) -> None:
        pass


class TestMacAddress(unittest.TestCase):
    def test_is_stable(self) -> None:
        assert satellite_mac_address("living_room") == satellite_mac_address(
            "living_room"
        )

    def test_differs_per_satellite(self) -> None:
        assert satellite_mac_address("a") != satellite_mac_address("b")

    def test_is_a_locally_administered_unicast_address(self) -> None:
        first_octet = int(satellite_mac_address("living_room").split(":")[0], 16)
        assert first_octet & 0x02
        assert not first_octet & 0x01


class TestFindWakeWords(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write(self, name: str, config: dict[str, Any]) -> None:
        with open(Path(self.tmp_dir) / f"{name}.json", "w") as f:
            json.dump(config, f)

    def test_builtin_wake_words(self) -> None:
        found = find_wake_words()
        assert {"okay_nabu", "hey_jarvis", "hey_mycroft", "alexa"} <= set(found)
        assert found["okay_nabu"].wake_word == "Okay Nabu"
        assert "nl" in found["okay_nabu"].trained_languages

    def test_extra_dir_adds_micro_models(self) -> None:
        self._write(
            "he_huis",
            {
                "type": "micro",
                "wake_word": "Hé Huis",
                "model": "he_huis.tflite",
                "trained_languages": ["nl"],
                "micro": {},
            },
        )
        found = find_wake_words(self.tmp_dir)
        assert found["he_huis"].wake_word == "Hé Huis"
        assert "okay_nabu" in found

    def test_non_micro_models_are_skipped(self) -> None:
        self._write(
            "oww", {"type": "openWakeWord", "wake_word": "x", "model": "oww.tflite"}
        )
        assert "oww" not in find_wake_words(self.tmp_dir)

    def test_broken_configs_are_skipped(self) -> None:
        with open(Path(self.tmp_dir) / "broken.json", "w") as f:
            f.write("{not json")
        assert "okay_nabu" in find_wake_words(self.tmp_dir)


class TestActiveWakeWords(unittest.TestCase):
    def test_default_is_okay_nabu(self) -> None:
        satellite = VoiceSatellite(make_config())
        assert [ww.id for ww in satellite.active_wake_words] == ["okay_nabu"]

    def test_models_use_their_own_threshold_by_default(self) -> None:
        satellite = VoiceSatellite(make_config())
        assert satellite.active_wake_words[0].probability_cutoff == 0.85

    def test_threshold_override(self) -> None:
        satellite = VoiceSatellite(make_config(wake_word_threshold=0.7))
        satellite.set_active_wake_words(["okay_nabu", "hey_jarvis"])
        assert [ww.probability_cutoff for ww in satellite.active_wake_words] == [
            0.7,
            0.7,
        ]

    def test_unknown_ids_are_skipped(self) -> None:
        satellite = VoiceSatellite(make_config())
        satellite.set_active_wake_words(["nope", "hey_jarvis"])
        assert [ww.id for ww in satellite.active_wake_words] == ["hey_jarvis"]

    def test_at_most_two_are_active(self) -> None:
        satellite = VoiceSatellite(make_config())
        satellite.set_active_wake_words(["okay_nabu", "hey_jarvis", "alexa"])
        assert [ww.id for ww in satellite.active_wake_words] == [
            "okay_nabu",
            "hey_jarvis",
        ]

    def test_config_rejects_more_than_two(self) -> None:
        with self.assertRaises(ValueError):
            make_config(wake_words=["okay_nabu", "hey_jarvis", "alexa"])


class TestOnAudio(unittest.TestCase):
    """The audio-thread side: what a detection schedules on the loop."""

    def setUp(self) -> None:
        self.satellite = VoiceSatellite(make_config(), speaker=FakeSpeaker())
        self.wake_word = FakeWakeWord("okay_nabu", "Okay Nabu")
        self.stop_word = FakeWakeWord("stop", "Stop")
        self.satellite.active_wake_words = [self.wake_word]  # type: ignore[list-item]
        self.satellite._stop_word = self.stop_word  # type: ignore[assignment]
        self.scheduled: list[tuple[str, tuple[Any, ...]]] = []
        self.satellite._call_soon = (  # type: ignore[method-assign]
            lambda cb, *args: self.scheduled.append((cb.__name__, args))
        )

    def _feed(self) -> None:
        # One 64 ms chunk is enough audio for several feature windows.
        self.satellite.on_audio(b"\x00" * 2048)

    def test_wake_word_schedules_wakeup(self) -> None:
        self.wake_word.fire = True
        self._feed()
        assert ("wakeup", ("Okay Nabu",)) in self.scheduled

    def test_wake_word_ignored_while_pipeline_active(self) -> None:
        self.satellite.pipeline_active = True
        self.wake_word.fire = True
        self._feed()
        assert not [s for s in self.scheduled if s[0] == "wakeup"]

    def test_refractory_period(self) -> None:
        self.wake_word.fire = True
        self._feed()
        self._feed()
        assert len([s for s in self.scheduled if s[0] == "wakeup"]) == 1

    def test_audio_is_sent_only_while_streaming(self) -> None:
        self._feed()
        assert not [s for s in self.scheduled if s[0] == "_mic_chunk"]

        self.satellite.streaming_audio = True
        self._feed()
        assert [s for s in self.scheduled if s[0] == "_mic_chunk"]

    def test_stop_word_only_while_speaking(self) -> None:
        self.stop_word.fire = True
        self._feed()
        assert ("stop_speaking", ()) not in self.scheduled

        self.satellite.speaking = True
        self._feed()
        assert ("stop_speaking", ()) in self.scheduled


class TestCues(unittest.TestCase):
    def test_cue_is_a_short_quiet_wav(self) -> None:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        path = str(Path(tmp) / "cue.wav")
        write_cue(path, (880.0, 1320.0))

        with wave.open(path) as cue:
            frames = cue.readframes(cue.getnframes())
            seconds = cue.getnframes() / cue.getframerate()

        samples = [
            int.from_bytes(frames[i : i + 2], "little", signed=True)
            for i in range(0, len(frames), 2)
        ]
        assert 0.1 < seconds < 0.5
        assert max(abs(v) for v in samples) < 0.2 * 32767
        assert abs(samples[0]) < 50  # faded in, no click

    def test_cues_follow_config(self) -> None:
        both = VoiceSatellite(make_config(), speaker=FakeSpeaker())
        assert set(both._cues) == {"wake", "done"}

        wake_only = VoiceSatellite(make_config(done_sound=False), speaker=FakeSpeaker())
        assert set(wake_only._cues) == {"wake"}

        none = VoiceSatellite(
            make_config(wake_sound=False, done_sound=False), speaker=FakeSpeaker()
        )
        assert none._cues == {}

    def test_no_cues_without_a_speaker(self) -> None:
        assert VoiceSatellite(make_config())._cues == {}


class TestMicHoldBack(unittest.TestCase):
    """Mic audio waits for Home Assistant to confirm the run, so the start
    of the command isn't lost when Home Assistant clears its audio queue."""

    def setUp(self) -> None:
        self.satellite = VoiceSatellite(make_config())
        self.sent: list[Any] = []
        self.satellite._send = self.sent.extend  # type: ignore[method-assign]
        self.satellite._start_run("Okay Nabu")
        self.sent.clear()  # the start request itself

    def _audio(self) -> list[bytes]:
        return [m.data for m in self.sent]

    def test_audio_is_held_until_the_run_is_accepted(self) -> None:
        self.satellite._mic_chunk(b"doe")
        self.satellite._mic_chunk(b"de")
        assert self.sent == []

        self.satellite._run_accepted()
        assert self._audio() == [b"doe", b"de"]

        self.satellite._mic_chunk(b"lampen")
        assert self._audio() == [b"doe", b"de", b"lampen"]

    def test_held_audio_is_capped(self) -> None:
        chunk = b"\x00" * 2048
        for _ in range(200):  # ~12.8 s, more than the cap
            self.satellite._mic_chunk(chunk)

        self.satellite._run_accepted()
        assert len(self.sent) * len(chunk) <= 16000 * 2 * 10

    def test_held_audio_is_dropped_if_the_run_ended(self) -> None:
        self.satellite._mic_chunk(b"doe")
        self.satellite.streaming_audio = False
        self.satellite._run_accepted()
        assert self.sent == []

    def test_a_new_run_starts_with_an_empty_buffer(self) -> None:
        self.satellite._mic_chunk(b"old")
        self.satellite._start_run("Okay Nabu")
        self.sent.clear()
        self.satellite._run_accepted()
        assert self.sent == []


class FakeTalkbackStream:
    is_running = True


class FakeCamera:
    """A Protect camera whose talkback never finishes."""

    name = "Living Room"

    def __init__(self) -> None:
        self.talkback_stream = FakeTalkbackStream()
        self.stopped = False

    async def play_audio(self, url: str, **_kwargs: Any) -> None:
        await asyncio.sleep(3600)

    async def stop_audio(self) -> None:
        self.stopped = True


class TestProtectSpeaker(unittest.IsolatedAsyncioTestCase):
    def _speaker(self) -> tuple[ProtectSpeaker, FakeCamera]:
        speaker = ProtectSpeaker(
            ProtectConfig(host="h", username="u", password="p", camera="Living Room")
        )
        camera = FakeCamera()

        async def get_camera() -> FakeCamera:
            speaker._camera = camera
            return camera

        speaker._get_camera = get_camera  # type: ignore[method-assign]
        return speaker, camera

    async def test_hung_playback_is_stopped(self) -> None:
        speaker, camera = self._speaker()

        with mock.patch("swatch.voice.PLAY_TIMEOUT_SECONDS", 0.1):
            with self.assertRaises(TimeoutError):
                await speaker.play("/tmp/reply.mp3")

        assert camera.stopped
        assert speaker._camera is None  # reconnect on the next reply

    async def test_http_replies_are_downloaded_first(self) -> None:
        """Home Assistant's chunked TTS responses make the talkback reader
        fail at the very end, so a reply is fetched to a file first."""
        from aiohttp import web

        async def tts(request: web.Request) -> web.StreamResponse:
            response = web.StreamResponse()
            response.enable_chunked_encoding()
            await response.prepare(request)
            await response.write(b"ID3fake-mp3")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/api/tts_proxy/reply.mp3", tts)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]

        speaker, _camera = self._speaker()
        played: list[tuple[str, bytes]] = []

        async def play_file(path: str) -> None:
            with open(path, "rb") as f:
                played.append((path, f.read()))

        speaker._play_file = play_file  # type: ignore[method-assign]

        try:
            await speaker.play(
                f"http://127.0.0.1:{port}/api/tts_proxy/reply.mp3?token=x"
            )
        finally:
            await runner.cleanup()

        assert played[0][1] == b"ID3fake-mp3"
        assert played[0][0].endswith(".mp3")
        assert not Path(played[0][0]).exists()  # cleaned up afterwards

    async def test_plays_are_queued_not_overlapped(self) -> None:
        """The camera can only play one stream at a time, so a reply that
        arrives while a cue is still playing waits for it."""
        speaker, _camera = self._speaker()
        log: list[str] = []

        async def play_file(path: str) -> None:
            log.append(f"start {path}")
            await asyncio.sleep(0.05)
            log.append(f"end {path}")

        speaker._play_file = play_file  # type: ignore[method-assign]
        await asyncio.gather(speaker.play("/cue.wav"), speaker.play("/reply.wav"))
        assert log == [
            "start /cue.wav",
            "end /cue.wav",
            "start /reply.wav",
            "end /reply.wav",
        ]

    async def test_local_files_are_played_directly(self) -> None:
        speaker, _camera = self._speaker()
        played: list[str] = []

        async def play_file(path: str) -> None:
            played.append(path)

        speaker._play_file = play_file  # type: ignore[method-assign]
        await speaker.play("/media/chime.mp3")
        assert played == ["/media/chime.mp3"]

    async def test_connect_failure_is_not_raised(self) -> None:
        speaker, _camera = self._speaker()

        async def broken() -> None:
            raise OSError("console unreachable")

        speaker._get_camera = broken  # type: ignore[method-assign,assignment]
        await speaker.connect()


class TestFraming(unittest.TestCase):
    def test_frames_split_across_reads_are_reassembled(self) -> None:
        satellite = VoiceSatellite(make_config())
        connection = EspHomeConnection(satellite)
        transport = mock.Mock(spec=asyncio.Transport)
        connection.connection_made(transport)
        handled: list[Any] = []
        connection._handle = handled.append  # type: ignore[method-assign]

        # PingRequest (type 7, empty) then DeviceInfoRequest (type 9, empty).
        data = bytes([0x00, 0x00, 0x07, 0x00, 0x00, 0x09])
        connection.data_received(data[:2])
        assert not handled
        connection.data_received(data[2:4])
        assert len(handled) == 1
        connection.data_received(data[4:])
        assert [type(m).__name__ for m in handled] == [
            "PingRequest",
            "DeviceInfoRequest",
        ]

    def test_encrypted_client_is_disconnected(self) -> None:
        satellite = VoiceSatellite(make_config())
        connection = EspHomeConnection(satellite)
        transport = mock.Mock(spec=asyncio.Transport)
        connection.connection_made(transport)
        connection.data_received(bytes([0x01, 0x00, 0x00]))
        transport.close.assert_called_once()


class TestHomeAssistantSession(unittest.IsolatedAsyncioTestCase):
    """End to end against aioesphomeapi's own client -- the library Home
    Assistant's ESPHome integration uses -- playing Home Assistant's side."""

    async def asyncSetUp(self) -> None:
        self.stop_event = multiprocessing.Event()
        self.speaker = FakeSpeaker()
        self.satellite = VoiceSatellite(
            make_config(echo_guard_seconds=0), speaker=self.speaker
        )
        self.server = VoiceSatelliteServer([self.satellite], self.stop_event)
        self.server.start()
        assert self.server.ready.wait(10)

        self.client = APIClient("127.0.0.1", self.satellite.port, None)
        await self.client.connect(login=True)

        self.starts: list[str | None] = []
        self.audio: list[bytes] = []
        self.finished = 0

        async def handle_start(
            conversation_id: str, flags: int, audio_settings: Any, phrase: str | None
        ) -> int:
            self.starts.append(phrase)
            return 0

        async def handle_stop(abort: bool) -> None:
            pass

        async def handle_audio(data: bytes, data2: bytes | None = None) -> None:
            self.audio.append(data)

        async def handle_finished(_msg: Any) -> None:
            self.finished += 1

        self.client.subscribe_voice_assistant(
            handle_start=handle_start,
            handle_stop=handle_stop,
            handle_audio=handle_audio,
            handle_announcement_finished=handle_finished,
        )
        # Let the subscription reach the satellite.
        await self._settle()

    async def asyncTearDown(self) -> None:
        await self.client.disconnect()
        self.stop_event.set()
        self.server.join(10)

    async def _settle(self, seconds: float = 0.2) -> None:
        await asyncio.sleep(seconds)

    def _on_loop(self, callback: Any, *args: Any) -> None:
        self.server.loop.call_soon_threadsafe(callback, *args)

    async def test_speaker_connects_at_startup(self) -> None:
        assert self.speaker.connected

    async def test_device_info(self) -> None:
        info = await self.client.device_info()
        assert info.name == "swatch-living-room"
        assert info.friendly_name == "Living Room"
        assert info.mac_address == satellite_mac_address("living_room")
        flags = info.voice_assistant_feature_flags_compat(self.client.api_version)
        assert flags & VoiceAssistantFeature.VOICE_ASSISTANT
        assert flags & VoiceAssistantFeature.API_AUDIO
        assert flags & VoiceAssistantFeature.ANNOUNCE

    async def test_wake_word_configuration(self) -> None:
        config = await self.client.get_voice_assistant_configuration(5)
        assert "okay_nabu" in [ww.id for ww in config.available_wake_words]
        assert list(config.active_wake_words) == ["okay_nabu"]
        assert config.max_active_wake_words == 2

        await self.client.set_voice_assistant_configuration(["hey_jarvis"])
        await self._settle()
        config = await self.client.get_voice_assistant_configuration(5)
        assert list(config.active_wake_words) == ["hey_jarvis"]

    async def test_full_run(self) -> None:
        self._on_loop(self.satellite.wakeup, "Okay Nabu")
        await self._settle()
        assert self.starts == ["Okay Nabu"]

        self.satellite.on_audio(b"\x01\x00" * 1024)
        await self._settle()
        assert len(self.audio) == 1

        events = VoiceAssistantEventType
        self.client.send_voice_assistant_event(events.VOICE_ASSISTANT_RUN_START, {})
        self.client.send_voice_assistant_event(
            events.VOICE_ASSISTANT_STT_END, {"text": "doe het licht uit"}
        )
        await self._settle()
        self.satellite.on_audio(b"\x01\x00" * 1024)
        await self._settle()
        assert len(self.audio) == 1  # no more audio once speech has ended

        self.client.send_voice_assistant_event(
            events.VOICE_ASSISTANT_TTS_END, {"url": "http://ha/reply.mp3"}
        )
        self.client.send_voice_assistant_event(events.VOICE_ASSISTANT_RUN_END, {})
        await self._settle()
        played = [Path(p).name for p in self.speaker.played]
        # rising cue on wake, falling cue when speech ended, then the reply
        assert played == ["wake.wav", "done.wav", "reply.mp3"]
        assert self.finished == 1
        assert not self.satellite.pipeline_active

    async def test_run_without_reply_ends_on_run_end(self) -> None:
        self._on_loop(self.satellite.wakeup, "Okay Nabu")
        await self._settle()
        self.client.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, {}
        )
        await self._settle()
        assert not self.satellite.pipeline_active
        assert [Path(p).name for p in self.speaker.played] == ["wake.wav"]

    async def test_continue_conversation_starts_a_new_run(self) -> None:
        events = VoiceAssistantEventType
        self._on_loop(self.satellite.wakeup, "Okay Nabu")
        await self._settle()
        self.client.send_voice_assistant_event(
            events.VOICE_ASSISTANT_INTENT_END, {"continue_conversation": "1"}
        )
        self.client.send_voice_assistant_event(
            events.VOICE_ASSISTANT_TTS_END, {"url": "http://ha/question.mp3"}
        )
        await self._settle()
        assert len(self.starts) == 2
        assert self.satellite.streaming_audio

    async def test_announcement(self) -> None:
        await self.client.send_voice_assistant_announcement_await_response(
            "http://ha/announce.mp3", 5, "de was is klaar"
        )
        assert self.speaker.played == ["http://ha/announce.mp3"]
        await self._settle()
        assert not self.satellite.pipeline_active

    async def test_wake_word_ignored_without_subscription(self) -> None:
        self.satellite._va_subscribed = False
        self._on_loop(self.satellite.wakeup, "Okay Nabu")
        await self._settle()
        assert self.starts == []
        assert not self.satellite.pipeline_active


if __name__ == "__main__":
    unittest.main()
