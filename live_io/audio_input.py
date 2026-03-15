import collections
import math
import os
import tempfile
import threading
import time
import wave
from dataclasses import dataclass

import pyaudio

from live_io.query_input import QueryInput


@dataclass(slots=True)
class AudioCaptureConfig:
    wake_phrase: str = "hello"
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    wake_window_s: float = 2.5
    wake_cooldown_s: float = 0.4
    command_max_record_s: float = 8.0
    silence_threshold: float = 550.0
    min_silence_duration_s: float = 1.0
    min_command_duration_s: float = 0.75
    pre_roll_s: float = 0.35
    transcription_model: str = "gpt-4o-transcribe"
    language: str | None = None
    input_device_index: int | None = None


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _chunk_rms(chunk: bytes) -> float:
    sample_count = len(chunk) // 2
    if sample_count == 0:
        return 0.0

    total = 0.0
    for i in range(0, len(chunk), 2):
        sample = int.from_bytes(chunk[i : i + 2], byteorder="little", signed=True)
        total += float(sample * sample)
    return math.sqrt(total / sample_count)


def _write_wav(path: str, frames: list[bytes], sample_rate: int, channels: int) -> None:
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))


class MicQueryInput(QueryInput):
    def __init__(self, audio_config: AudioCaptureConfig, transcriber) -> None:
        self._audio_config = audio_config
        self._transcriber = transcriber
        self._queries: collections.deque[str] = collections.deque()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll_query(self) -> str | None:
        with self._lock:
            if not self._queries:
                return None
            return self._queries.popleft()

    def release(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        audio = pyaudio.PyAudio()
        stream = None
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=self._audio_config.channels,
                rate=self._audio_config.sample_rate,
                input=True,
                frames_per_buffer=self._audio_config.chunk_size,
                input_device_index=self._audio_config.input_device_index,
            )
            print(f"audio query input ready; say '{self._audio_config.wake_phrase}'")

            while not self._stop_event.is_set():
                wake_window = self._capture_fixed_window(stream)
                if not wake_window:
                    continue

                wake_text = self._transcribe_frames(wake_window)
                if not wake_text:
                    continue
                print("wake transcript:", wake_text)

                if _normalize_text(self._audio_config.wake_phrase) not in _normalize_text(
                    wake_text
                ):
                    time.sleep(self._audio_config.wake_cooldown_s)
                    continue

                print("wake phrase detected; recording command")
                command_frames = self._record_command(
                    stream,
                    seed_frames=self._tail_frames(
                        wake_window,
                        seconds=max(1.0, self._audio_config.pre_roll_s),
                    ),
                )
                command_text = self._transcribe_frames(command_frames)
                if not command_text:
                    print("no command transcription produced")
                    time.sleep(self._audio_config.wake_cooldown_s)
                    continue

                command_text = self._strip_wake_phrase(command_text)
                if not command_text:
                    print("command was empty after removing wake phrase")
                    time.sleep(self._audio_config.wake_cooldown_s)
                    continue

                with self._lock:
                    self._queries.append(command_text)
                print("audio query:", command_text)
                time.sleep(self._audio_config.wake_cooldown_s)
        except Exception as exc:
            print(f"audio query input error: {exc}")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

    def _capture_fixed_window(self, stream) -> list[bytes]:
        frame_count = max(
            1,
            int(
                self._audio_config.wake_window_s
                * self._audio_config.sample_rate
                / self._audio_config.chunk_size
            ),
        )
        frames = []
        for _ in range(frame_count):
            if self._stop_event.is_set():
                return []
            frames.append(
                stream.read(
                    self._audio_config.chunk_size,
                    exception_on_overflow=False,
                )
            )
        return frames

    def _record_command(self, stream, seed_frames: list[bytes] | None = None) -> list[bytes]:
        pre_roll_chunks = max(
            0,
            int(
                self._audio_config.pre_roll_s
                * self._audio_config.sample_rate
                / self._audio_config.chunk_size
            ),
        )
        pre_roll = collections.deque(maxlen=max(1, pre_roll_chunks))
        frames = list(seed_frames or [])
        started_at = time.time()
        silence_started_at = None

        while not self._stop_event.is_set():
            chunk = stream.read(
                self._audio_config.chunk_size,
                exception_on_overflow=False,
            )
            rms = _chunk_rms(chunk)
            pre_roll.append(chunk)

            if not frames:
                if rms >= self._audio_config.silence_threshold:
                    frames.extend(pre_roll)
                    silence_started_at = None
                elif time.time() - started_at >= self._audio_config.command_max_record_s:
                    break
                continue
            else:
                frames.append(chunk)

            if rms < self._audio_config.silence_threshold:
                if silence_started_at is None:
                    silence_started_at = time.time()
                elif (
                    time.time() - silence_started_at
                    >= self._audio_config.min_silence_duration_s
                    and time.time() - started_at
                    >= self._audio_config.min_command_duration_s
                ):
                    break
            else:
                silence_started_at = None

            if time.time() - started_at >= self._audio_config.command_max_record_s:
                break

        return frames

    def _tail_frames(self, frames: list[bytes], seconds: float) -> list[bytes]:
        if not frames:
            return []
        frame_count = max(
            1,
            int(seconds * self._audio_config.sample_rate / self._audio_config.chunk_size),
        )
        return frames[-frame_count:]

    def _transcribe_frames(self, frames: list[bytes]) -> str:
        if not frames:
            return ""

        fd, temp_path = tempfile.mkstemp(prefix="mirror_mind_", suffix=".wav")
        os.close(fd)
        try:
            _write_wav(
                temp_path,
                frames,
                sample_rate=self._audio_config.sample_rate,
                channels=self._audio_config.channels,
            )
            return self._transcriber(
                temp_path,
                model=self._audio_config.transcription_model,
                language=self._audio_config.language,
            ).strip()
        except Exception as exc:
            print(f"transcription error: {exc}")
            return ""
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def _strip_wake_phrase(self, transcript: str) -> str:
        transcript = transcript.strip()
        lowered = transcript.lower()
        wake_lower = self._audio_config.wake_phrase.lower()
        wake_idx = lowered.find(wake_lower)
        if wake_idx < 0:
            return transcript
        trimmed = transcript[wake_idx + len(self._audio_config.wake_phrase) :].strip()
        return trimmed.strip(" ,.:;-")
