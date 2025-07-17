"""
YouTube Audio Transcription System
A modular system for extracting audio from YouTube videos and transcribing using Vosk
"""

from services.transcriber_service import TranscriptionService

from dtos import TranscriptionResult
from utils.logger import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# Create FastAPI app
app = FastAPI(
    title="YouTube Transcription API (Wav2Vec2)",
    description="A scalable API for transcribing YouTube videos using Meta's Wav2Vec2",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionRequest(BaseModel):
    url: str

# Initialize transcription service
MODEL_PATH = "./models"

@app.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_youtube_video(request: TranscriptionRequest):
    try:
        service = TranscriptionService(MODEL_PATH)
        
        result = service.transcribe_youtube_video(request.url)
        
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )