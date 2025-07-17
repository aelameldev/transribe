
from exceptions import TranscriptionError
from pydub import AudioSegment
import tempfile
from utils.logger import logger


class AudioProcessor:
    """Handles audio file processing for transcription"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def prepare_audio(self, audio_path: str) -> str:
        """Prepare audio file for transcription (convert to required format)"""
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1).set_frame_rate(self.sample_rate)
            
            # Create temporary WAV file
            temp_path = tempfile.mktemp(suffix='.wav')
            audio.export(temp_path, format='wav')
            
            logger.info(f"Audio prepared for transcription: {temp_path}")
            return temp_path
            
        except Exception as e:
            raise TranscriptionError(f"Failed to prepare audio: {str(e)}")
