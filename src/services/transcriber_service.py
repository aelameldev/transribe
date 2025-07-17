

import os
import tempfile
from typing import Optional, List

from dtos import TranscriptionResult
from services.extractors.audio import AudioExtractor, YouTubeAudioExtractor
from services.processors.audio_processor import AudioProcessor
from services.transcribers.vosk_transcriber import VoskTranscriber
from utils.logger import logger

class TranscriptionService:
    """Main service class that orchestrates the transcription process"""
    
    def __init__(self, model_path: str, audio_extractor: AudioExtractor = None):
        self.model_path = model_path
        self.audio_extractor = audio_extractor or YouTubeAudioExtractor()
        self.audio_processor = AudioProcessor()
        self.transcriber = VoskTranscriber(model_path)
    
    def transcribe_youtube_video(self, url: str, cleanup: bool = True) -> TranscriptionResult:
        """Complete transcription pipeline for YouTube videos"""
        temp_dir = tempfile.mkdtemp()
        audio_file = None
        processed_audio = None
        
        try:
            logger.info(f"Starting transcription for: {url}")
            
            # Extract audio
            audio_file = self.audio_extractor.extract_audio(url, temp_dir)
            
            # Prepare audio for transcription
            processed_audio = self.audio_processor.prepare_audio(audio_file)
            
            # Transcribe
            result = self.transcriber.transcribe(processed_audio)
            
            logger.info("Transcription completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
            
        finally:
            # Cleanup temporary files
            if cleanup:
                self._cleanup_files([audio_file, processed_audio])
                self._cleanup_directory(temp_dir)
    
    def _cleanup_files(self, files: List[Optional[str]]) -> None:
        """Clean up temporary files"""
        for file_path in files:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
    
    def _cleanup_directory(self, directory: str) -> None:
        """Clean up temporary directory"""
        try:
            import shutil
            shutil.rmtree(directory)
            logger.debug(f"Cleaned up directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to clean up directory {directory}: {e}")
