import logging
import os
import tempfile
import time
import shutil
from dtos import TranscriptionResponse
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2CTCTokenizer
import soundfile as sf
import numpy as np
from typing import  Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Wav2Vec2TranscriptionService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
    def get_model_and_tokenizer(self, model_name: str):
        """Load and cache Wav2Vec2 models and tokenizers"""
        if model_name not in self.models:
            logger.info(f"Loading Wav2Vec2 model: {model_name}")
            self.tokenizers[model_name] = Wav2Vec2Tokenizer.from_pretrained(model_name)
            self.models[model_name] = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            
        return self.models[model_name], self.tokenizers[model_name]
    
    def get_ydl_opts(self, output_path: str) -> Dict[str, Any]:
        """Get yt-dlp options with anti-blocking measures"""
        return {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': 1,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'retries': 3,
            'fragment_retries': 3,
            'extractor_retries': 3,
            'user_agent': random.choice(self.user_agents),
            'headers': {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            },
            'sleep_interval': 1,
            'max_sleep_interval': 3,
        }
    
    def download_audio(self, url: str, max_retries: int = 3) -> tuple[str, Dict[str, Any]]:
        """Download audio from YouTube URL with retry logic"""
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.%(ext)s")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(random.uniform(2, 5))  # Random delay between retries
                
                ydl_opts = self.get_ydl_opts(audio_path)
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                
                # Find the downloaded audio file
                audio_file = None
                for file in os.listdir(temp_dir):
                    if file.startswith("audio."):
                        audio_file = os.path.join(temp_dir, file)
                        break
                
                if not audio_file or not os.path.exists(audio_file):
                    raise Exception("Audio download failed - file not found")
                
                video_info = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date', 'Unknown')
                }
                
                logger.info(f"Successfully downloaded audio on attempt {attempt + 1}")
                return audio_file, video_info
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Try alternative approach on final attempt
                    try:
                        return self.download_audio_alternative(url, temp_dir)
                    except Exception as alt_e:
                        raise Exception(f"All download attempts failed. Last error: {str(e)}")
                
                # Clean up temp files before retry
                try:
                    for file in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, file)
                        # if os.path.isfile(file_path):
                        #     os.remove(file_path)
                except:
                    pass
    
    def download_audio_alternative(self, url: str, temp_dir: str) -> tuple[str, Dict[str, Any]]:
        """Alternative download method with different settings"""
        audio_path = os.path.join(temp_dir, "audio.%(ext)s")
        
        # More aggressive options for blocked videos
        ydl_opts = {
            'format': 'worst[ext=mp4]/worst',  # Try worst quality first
            'outtmpl': audio_path,
            'extractaudio': True,
            'audioformat': 'wav',
            'audioquality': 1,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '128',  # Lower quality
            }],
            'quiet': True,
            'no_warnings': True,
            'retries': 5,
            'fragment_retries': 5,
            'extractor_retries': 5,
            'user_agent': random.choice(self.user_agents),
            'headers': {
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
            },
            'sleep_interval': 2,
            'max_sleep_interval': 5,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        
        # Find the downloaded audio file
        audio_file = None
        for file in os.listdir(temp_dir):
            if file.startswith("audio."):
                audio_file = os.path.join(temp_dir, file)
                break
        
        if not audio_file or not os.path.exists(audio_file):
            raise Exception("Alternative download method failed")
        
        video_info = {
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0),
            'upload_date': info.get('upload_date', 'Unknown')
        }
        
        return audio_file, video_info
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio for Wav2Vec2"""
        # Load audio file
        speech, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            speech = torchaudio.transforms.Resample(sample_rate, 16000)(torch.tensor(speech)).numpy()
        
        # Normalize
        speech = speech.astype(np.float32)
        speech = (speech - speech.mean()) / (speech.std() + 1e-8)
        
        return speech
    
    def transcribe_audio(self, audio_path: str, model_name: str) -> Dict[str, Any]:
        """Transcribe audio using Wav2Vec2"""
        model, tokenizer = self.get_model_and_tokenizer(model_name)
        
        # Preprocess audio
        speech = self.preprocess_audio(audio_path)
        
        # Tokenize
        inputs = tokenizer(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = model(inputs['input_values']).logits
        
        # Get predictions
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        
        # Calculate confidence score (average max probability)
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        confidence = max_probs.mean().item()
        
        return {
            'text': transcription,
            'confidence': confidence
        }
    
    def process_video(self, url: str, model_name: str = "facebook/wav2vec2-base-960h") -> TranscriptionResponse:
        """Main processing function"""
        start_time = time.time()
        temp_dir = None
        
        try:
            # Download audio
            logger.info(f"Downloading audio from: {url}")
            audio_path = "/Users/aea/Workspace/builds/yt-transcribe/test.mp3"
            # audio_path, video_info = self.download_audio(url)
            temp_dir = os.path.dirname(audio_path)
            
            # Transcribe audio
            logger.info(f"Transcribing audio with model: {model_name}")
            result = self.transcribe_audio(audio_path, model_name)
            
            processing_time = time.time() - start_time
            
            return TranscriptionResponse(
                success=True,
                transcription=result["text"].strip(),
                model_used=model_name,
                #duration=video_info.get("duration"),
                processing_time=processing_time,
                #video_info=video_info,
                confidence_score=result.get("confidence", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return TranscriptionResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        finally:
            # Cleanup temp files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir: {e}")