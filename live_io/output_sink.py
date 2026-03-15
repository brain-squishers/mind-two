from abc import ABC, abstractmethod


class OutputSink(ABC):
    @abstractmethod
    def publish_text(self, text: str) -> None:
        raise NotImplementedError


class DebugTextOutputSink(OutputSink):
    def __init__(self):
        self.latest_text = ""

    def publish_text(self, text: str) -> None:
        if not text or text == self.latest_text:
            return
        self.latest_text = text
        print("output:", text)
