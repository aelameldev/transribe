
from typing import Optional
import logging
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, validator
from transcription_service import Wav2Vec2TranscriptionService
import yt_dlp

import uvicorn

from dtos import TranscriptionResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class TranscriptionRequest(BaseModel):
    url: HttpUrl
    model_name: Optional[str] = "facebook/wav2vec2-base-960h"
    
    @validator('model_name')
    def validate_model_name(cls, v):
        valid_models = [
            'facebook/wav2vec2-base-960h',
            'facebook/wav2vec2-large-960h',
            'facebook/wav2vec2-large-960h-lv60',
            'facebook/wav2vec2-large-960h-lv60-self',
            'jonatasgrosman/wav2vec2-large-xlsr-53-english',
            'jonatasgrosman/wav2vec2-large-xlsr-53-spanish',
            'jonatasgrosman/wav2vec2-large-xlsr-53-french',
            'jonatasgrosman/wav2vec2-large-xlsr-53-german',
            'jonatasgrosman/wav2vec2-large-xlsr-53-italian',
            'jonatasgrosman/wav2vec2-large-xlsr-53-portuguese',
            'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
            'jonatasgrosman/wav2vec2-large-xlsr-53-arabic',
            'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn',
            'jonatasgrosman/wav2vec2-large-xlsr-53-japanese',
            'jonatasgrosman/wav2vec2-large-xlsr-53-korean'
        ]
        if v not in valid_models:
            raise ValueError(f'Model must be one of: {valid_models}')
        return v


# Initialize service
transcription_service = Wav2Vec2TranscriptionService()

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

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": transcription_service.device,
        "models_loaded": list(transcription_service.models.keys())
    }

# Main transcription endpoint
@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_video(request: TranscriptionRequest):
    """
    Transcribe a YouTube video using Wav2Vec2
    
    - **url**: YouTube video URL
    - **model_name**: Wav2Vec2 model to use
    """
    # Validate YouTube URL
    url_str = str(request.url)
    if not any(domain in url_str for domain in ["youtube.com", "youtu.be"]):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Check video length limit
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'user_agent': random.choice(transcription_service.user_agents),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url_str, download=False)
            duration = info.get('duration', 0)
            if duration and duration > 3600:  # 1 hour limit
                raise HTTPException(
                    status_code=413, 
                    detail="Video too long. Maximum duration is 1 hour."
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Failed to extract video info: {str(e)}")
    
    # Process the video
    result = transcription_service.process_video(url_str, request.model_name)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    
    return result


# Get available models
@app.get("/models")
def get_models():
    """Get available Wav2Vec2 models"""
    return {
        "models": {
            "facebook/wav2vec2-base-960h": {
                "language": "English",
                "size": "~95MB",
                "accuracy": "Good",
                "description": "Base model trained on 960h of LibriSpeech"
            },
            "facebook/wav2vec2-large-960h": {
                "language": "English", 
                "size": "~315MB",
                "accuracy": "Better",
                "description": "Large model trained on 960h of LibriSpeech"
            },
            "facebook/wav2vec2-large-960h-lv60": {
                "language": "English",
                "size": "~315MB", 
                "accuracy": "Best",
                "description": "Large model with self-supervised pre-training"
            },
            "jonatasgrosman/wav2vec2-large-xlsr-53-english": {
                "language": "English",
                "size": "~315MB",
                "accuracy": "Excellent",
                "description": "Fine-tuned XLSR-53 model for English"
            },
            "jonatasgrosman/wav2vec2-large-xlsr-53-spanish": {
                "language": "Spanish",
                "size": "~315MB",
                "accuracy": "Excellent", 
                "description": "Fine-tuned XLSR-53 model for Spanish"
            },
            "jonatasgrosman/wav2vec2-large-xlsr-53-french": {
                "language": "French",
                "size": "~315MB",
                "accuracy": "Excellent",
                "description": "Fine-tuned XLSR-53 model for French"
            }
        }
    }

# Compare models endpoint
@app.post("/compare")
def compare_models(url: str, models: list[str]):
    """Compare transcription results from multiple models"""
    if len(models) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 models per comparison")
    
    results = {}
    for model_name in models:
        try:
            request = TranscriptionRequest(url=url, model_name=model_name)
            result = transcribe_video(request)
            results[model_name] = result
        except Exception as e:
            results[model_name] = {"error": str(e)}
    
    return {"comparison": results}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )