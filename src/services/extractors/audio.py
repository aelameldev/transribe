from abc import ABC, abstractmethod
from pydub.utils import which
from exceptions import AudioExtractorError
import yt_dlp
from pathlib import Path
from utils.logger import logger
import os

class AudioExtractor(ABC):
    """Abstract base class for audio extractors"""
    
    @abstractmethod
    def extract_audio(self, url: str, output_path: str) -> str:
        """Extract audio from URL and save to output path"""
        pass


class YouTubeAudioExtractor(AudioExtractor):
    """Concrete implementation for YouTube audio extraction"""
    
    def __init__(self, audio_format: str = 'wav', quality: str = 'best'):
        self.audio_format = audio_format
        self.quality = quality
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """Validate required dependencies"""
        if not which("ffmpeg"):
            raise AudioExtractorError("ffmpeg is required but not found in PATH")
    
    def extract_audio(self, url: str, output_path: str) -> str:
        """Extract audio from YouTube URL"""
        try:
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'{output_path}/%(title)s.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': self.audio_format,
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'unknown')
                
                # Download and extract audio
                ydl.download([url])
                
                # Find the downloaded file
                audio_file = self._find_audio_file(output_path, title)
                logger.info(f"Audio extracted successfully: {audio_file}")
                
                return audio_file
                
        except Exception as e:
            raise AudioExtractorError(f"Failed to extract audio: {str(e)}")
    
    def _find_audio_file(self, output_path: str, title: str) -> str:
        """Find the downloaded audio file"""
        path = Path(output_path)
        
        # Look for files with the title
        for file in path.glob(f"*{self.audio_format}"):
            if title.replace(" ", "_") in file.name or title in file.name:
                return str(file)
        
        # Fallback: return the most recent audio file
        audio_files = list(path.glob(f"*.{self.audio_format}"))
        if audio_files:
            return str(max(audio_files, key=os.path.getctime))
        
        raise AudioExtractorError("No audio file found after extraction")
