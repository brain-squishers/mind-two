from abc import ABC, abstractmethod

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

