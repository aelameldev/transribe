
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


logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cmgmtDB")
MODEL_ROOT = os.getenv("MODEL_ROOT", "/app/models")
ssl._create_default_https_context = ssl._create_unverified_context


SQL_FETCH_PROCESSING = """
    SELECT id, url
    FROM transcriptions
    WHERE status = 'PROCESSING'
    ORDER BY id
    LIMIT %s
"""

SQL_MARK_PENDING = """
    UPDATE transcriptions
    SET status = 'PENDING'
    WHERE id = %s
"""

SQL_MARK_DONE = """
    UPDATE transcriptions
    SET status = 'DONE', transcript = %s
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

def download_audio(youtube_url, base_output="audios") -> str:
    """Download audio from YouTube into a sanitized title folder."""
    # Extract info first (without download)
    ydl_opts_info = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        title = info.get("title", "audio")

    # Sanitize title for folder name
    folder_name = sanitize_title(title)
    output_path = os.path.join(base_output, folder_name)

    # Ensure folder exists
    os.makedirs(output_path, exist_ok=True)

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

    return os.path.join(output_path, "audio.mp3")

def transcribe_with_whisperx(audio_path, model_size="small.en", device=None):
    """Transcribe audio with WhisperX and return transcript text."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🔹 Using device: {device}")

    try:
        # Load ASR model
        model = whisper.load_model(model_size=model_size, device=device, download_root=MODEL_ROOT)
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

                    try:
                        if not validate_youtube_url(url):
                            logger.error(f"❌ Invalid YouTube URL for request id={req_id}: {url}")
                            continue

                        # Mark as PENDING
                        cur.execute(SQL_MARK_PENDING, [req_id])
                        conn.commit()

                        # Download audio
                        audio_path = download_audio(url)

                        # Transcribe
                        transcript = transcribe_with_whisperx(audio_path)

                        # Mark as DONE with transcript
                        cur.execute(SQL_MARK_DONE, [transcript, req_id])
                    except Exception as e:
                        logger.error(f"❌ Error processing request id={req_id}: {e}")
                        # Optionally, you could mark the request as FAILED in the DB here
                        cur.execute(SQL_MARK_FAILED, [req_id])

                        continue

                    logger.info(f"✅ Finished request id={req_id}")

    finally:
        conn.close()

