
import psycopg
from psycopg import rows

import logging
import os
import torch
import whisper
import ssl
import re
import os
import unicodedata
import yt_dlp
import time
import shutil


logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cmgmtDB")
MODEL_ROOT = os.getenv("MODEL_ROOT", "/Users/aea/Workspace/builds/transribe/cron_app/models")
ssl._create_default_https_context = ssl._create_unverified_context


SQL_FETCH_PROCESSING = """
    SELECT id, url
    FROM transcriptions
    WHERE status = 'QUEUE_FOR_PROCESSING'
    ORDER BY id
    LIMIT %s
"""

SQL_MARK_PROCESSING = """
    UPDATE transcriptions
    SET status = 'PROCESSING', video_title = %s
    WHERE id = %s
"""

SQL_MARK_DONE = """
    UPDATE transcriptions
    SET status = 'COMPLETED', transcript = %s, processing_time = %s
    WHERE id = %s
"""

SQL_MARK_FAILED = """
    UPDATE transcriptions
    SET status = 'FAILED'
    WHERE id = %s
"""

def sanitize_title(title: str, max_length: int = 50) -> str:
    """Remove spaces, special characters, and emojis from title, limit length."""
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
    title = re.sub(r"_+", "_", title).strip("_")
    if len(title) > max_length:
        title = title[:max_length].rstrip("_-")
    return title

def get_youtube_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    # Handle various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def find_existing_audio_file(output_path: str) -> str:
    """Check for existing audio files in the given path with various extensions."""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.webm', '.ogg', '.flac']
    base_name = "audio"
    
    for ext in audio_extensions:
        audio_file = os.path.join(output_path, f"{base_name}{ext}")
        if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
            logger.info(f"🔍 Found existing audio file: {audio_file}")
            return audio_file
    
    return None

def download_audio(youtube_url, base_output="audios") -> tuple:
    """Download audio from YouTube into a sanitized title folder, or use existing file if found.
    Returns:
        tuple: (sanitized_title, audio_file_path, folder_path)
    """
    # Extract YouTube video ID for consistent folder naming
    video_id = get_youtube_video_id(youtube_url)
    
    # Extract info first (without download)
    ydl_opts_info = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        title = info.get("title", "audio")

    # Create folder name using both sanitized title and video ID for uniqueness
    sanitized_title = sanitize_title(title)
    if video_id:
        folder_name = f"{sanitized_title}_{video_id}"
    else:
        folder_name = sanitized_title
        
    output_path = os.path.join(base_output, folder_name)
    
    # Ensure folder exists
    os.makedirs(output_path, exist_ok=True)

    # Check if any audio file already exists in the folder
    existing_audio = find_existing_audio_file(output_path)
    if existing_audio:
        logger.info(f"🔄 Using existing audio file: {existing_audio}")
        return (title, sanitized_title, existing_audio, output_path)

    logger.info(f"📥 Downloading audio from: {youtube_url}")
    audio_file_path = os.path.join(output_path, "audio.mp3")

    # Now download audio inside this folder
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}/audio.%(ext)s",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "quiet": False,  # set to False to see errors
        "retries": 10,
        "fragment_retries": 10,
        "postprocessor_args": [
             "-vn"   # strip video (ensures only audio is processed)
        ],
        "keepvideo": False,   # delete the original file after conversion
        "overwrites": True,   # overwrite if file already exists
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    logger.info(f"✅ Audio downloaded successfully: {audio_file_path}")
    return (title,sanitized_title, audio_file_path, output_path)

def transcribe_with_whisperx(audio_path, model_size="small.en", device=None):
    """Transcribe audio with WhisperX and return transcript text."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔹 Using device: {device}")

    try:
        # Load ASR model
        model = whisper.load_model(name=model_size, device=device, download_root=MODEL_ROOT)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Trying with SSL context disabled...")
        
        # Alternative approach - disable SSL verification temporarily
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        model = whisper.load_model(model_size, device)

    # Transcribe
    result = model.transcribe(audio_path)
    segments = result["segments"]

    # Collect raw transcript
    transcript = " ".join([seg["text"].strip() for seg in segments])

    return transcript

def cleanup_audio_folder(audio_path: str) -> bool:
    """Delete the audio file and its containing folder safely."""
    try:
        # Get the folder containing the audio file
        folder_path = os.path.dirname(audio_path)
        
        if os.path.exists(folder_path):
            # Remove the entire folder and its contents
            shutil.rmtree(folder_path)
            logger.info(f"🗑️ Cleaned up audio folder: {folder_path}")
            return True
        else:
            logger.warning(f"⚠️ Audio folder not found for cleanup: {folder_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Error cleaning up audio folder {folder_path}: {e}")
        return False

def validate_youtube_url(url: str) -> bool:
    """Validate if the given string is a YouTube link."""
    if not url.strip():
        return False
    youtube_regex = re.compile(
        r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$",
        re.IGNORECASE,
    )
    return bool(youtube_regex.match(url))



def process_requests(limit: int = 5):
    conn = psycopg.connect(DATABASE_URL)
    try:
        with conn:
            with conn.cursor(row_factory=rows.dict_row) as cur:
                cur.execute(SQL_FETCH_PROCESSING, [limit])
                data = cur.fetchall()

                for row in data:
                    req_id, url = row["id"], row["url"]
                    logger.info(f"Picked request id={req_id}, file={url}")
                    
                    # Start timing the processing
                    start_time = time.time()

                    try:
                        if not validate_youtube_url(url):
                            logger.error(f"❌ Invalid YouTube URL for request id={req_id}: {url}")
                            continue

                        # Download audio
                        title, sanitize_title, audio_path, folder_path = download_audio(url)

                        # Mark as PROCESSING
                        cur.execute(SQL_MARK_PROCESSING, [title, req_id])
                        conn.commit()

                        # Transcribe
                        transcript = transcribe_with_whisperx(audio_path)

                        # Calculate processing time in seconds
                        processing_time = round(time.time() - start_time, 2)
                        logger.info(f"⏱️ Processing took {processing_time} seconds for request id={req_id}")

                        # Mark as DONE with transcript and processing time
                        cur.execute(SQL_MARK_DONE, [transcript, processing_time, req_id])
                        
                        # Clean up audio files after successful transcription
                        cleanup_audio_folder(folder_path)
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing request id={req_id}: {e}")
                        # Optionally, you could mark the request as FAILED in the DB here
                        cur.execute(SQL_MARK_FAILED, [req_id])

                        continue

                    logger.info(f"✅ Finished request id={req_id}")

    finally:
        conn.close()

