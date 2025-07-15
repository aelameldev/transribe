from pydantic import BaseModel
from typing import Optional, Dict, Any


class TranscriptionResponse(BaseModel):
    success: bool
    transcription: Optional[str] = None
    model_used: Optional[str] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    video_info: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    