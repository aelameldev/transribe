import os
from dtos import TranscriptionResult
from exceptions import TranscriptionError
from services.transcribers.transcriber import Transcriber
import vosk
import wave
import json
from typing import List, Dict, Any

from utils.logger import logger

class VoskTranscriber(Transcriber):
    """Handles transcription using Vosk"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Vosk model"""
        try:
            if not os.path.exists(self.model_path):
                raise TranscriptionError(f"Model path does not exist: {self.model_path}")
            
            self.model = vosk.Model(self.model_path)
            logger.info(f"Vosk model loaded from: {self.model_path}")
            
        except Exception as e:
            raise TranscriptionError(f"Failed to load Vosk model: {str(e)}")
    
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file"""
        try:
            # Open audio file
            with wave.open(audio_path, 'rb') as wf:
                # Validate audio format
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    raise TranscriptionError("Audio must be mono 16-bit PCM")
                
                # Initialize recognizer
                rec = vosk.KaldiRecognizer(self.model, wf.getframerate())
                rec.SetWords(True)
                
                # Process audio
                segments = []
                full_text = ""
                
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if result.get('text'):
                            segments.append(result)
                            full_text += result['text'] + " "
                
                # Get final result
                final_result = json.loads(rec.FinalResult())
                if final_result.get('text'):
                    segments.append(final_result)
                    full_text += final_result['text']
                
                # Calculate duration
                duration = wf.getnframes() / wf.getframerate()
                
                return TranscriptionResult(
                    text=full_text.strip(),
                    confidence=self._calculate_confidence(segments),
                    duration=duration
                )
                
        except Exception as e:
            raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")
    
    def _calculate_confidence(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate average confidence from segments"""
        if not segments:
            return 0.0
        
        total_confidence = 0.0
        word_count = 0
        
        for segment in segments:
            if 'result' in segment:
                for word in segment['result']:
                    if 'conf' in word:
                        total_confidence += word['conf']
                        word_count += 1
        
        return total_confidence / word_count if word_count > 0 else 0.0

