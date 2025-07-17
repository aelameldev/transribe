from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    """Data class for transcription results"""
    text: str
    confidence: float
    duration: float
