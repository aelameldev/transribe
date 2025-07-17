

from abc import abstractmethod, ABC

from dtos import TranscriptionResult


class Transcriber(ABC):
    """Abstract base class for transcribers"""

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file and return transcription result"""
        pass