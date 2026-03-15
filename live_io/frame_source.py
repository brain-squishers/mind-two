from abc import ABC, abstractmethod
import json
import threading
import time
import urllib.request

import cv2
import numpy as np
from live_runtime import LatestFrameCapture


class FrameSource(ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def read_latest(self):
        raise NotImplementedError

    @abstractmethod
    def release(self):
        raise NotImplementedError

    def get_status(self):
        return {}


class WebcamFrameSource(FrameSource):
    def __init__(self, camera_index: int):
        self._capture = LatestFrameCapture(camera_index)

    def start(self):
        self._capture.start()
        return self

    def read_latest(self):
        return self._capture.read_latest()

    def release(self):
        self._capture.release()

    def get_status(self):
        return {
            "source": "webcam",
            "ready": True,
        }


class ServerStreamFrameSource(FrameSource):
    def __init__(
        self,
        stream_url: str,
        poll_interval_s: float = 0.03,
        timeout_s: float = 2.0,
    ):
        self.stream_url = stream_url
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._frame = None
        self._last_error = ""
        self._last_state = "starting"
        self._last_meta = {}
        self._status_logged = set()
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _reader(self):
        while not self._stopped:
            try:
                meta_url = self._build_meta_url()
                with urllib.request.urlopen(
                    meta_url,
                    timeout=self.timeout_s,
                ) as response:
                    meta = json.loads(response.read().decode("utf-8"))
                with self._lock:
                    self._last_meta = meta
                    if meta.get("ready"):
                        self._last_state = "ready"
                        self._last_error = ""
                    elif meta.get("connected"):
                        self._last_state = "waiting_for_frames"
                    else:
                        self._last_state = "waiting_for_client"

                with urllib.request.urlopen(
                    self.stream_url,
                    timeout=self.timeout_s,
                ) as response:
                    encoded = response.read()
                np_buffer = np.frombuffer(encoded, dtype=np.uint8)
                frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._last_state = "ready"
                        self._last_error = ""
            except Exception:
                with self._lock:
                    if self._last_state == "starting":
                        self._last_state = "unreachable"
                    self._last_error = "Failed to fetch server stream frame"
                time.sleep(self.poll_interval_s)
                continue

            time.sleep(self.poll_interval_s)

    def _build_meta_url(self):
        if self.stream_url.endswith("/latest-frame"):
            return f"{self.stream_url}/meta"
        return self.stream_url

    def read_latest(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def release(self):
        self._stopped = True
        self._thread.join(timeout=1.0)

    def get_status(self):
        with self._lock:
            return {
                "source": "server",
                "ready": self._frame is not None,
                "state": self._last_state,
                "error": self._last_error,
                "meta": dict(self._last_meta),
            }
