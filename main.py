import yt_dlp
import torch
import whisper
import ssl
import re
import os
import unicodedata
import sys

ssl._create_default_https_context = ssl._create_unverified_context

def sanitize_title(title: str, max_length: int = 50) -> str:
    """Remove spaces, special characters, and emojis from title, limit length."""
    # Normalize Unicode (remove accents/emojis)
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    # Replace non-alphanumeric with underscores
    title = re.sub(r"[^a-zA-Z0-9_-]", "_", title)
    # Remove duplicate underscores
    title = re.sub(r"_+", "_", title).strip("_")
    # Truncate to max_length
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
        model = whisper.load_model(model_size, device)
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

if __name__ == "__main__":
    youtube_link = input("Enter YouTube video URL: ").strip()

    if validate_youtube_url(youtube_link) is False:
        print("❌ Invalid YouTube link. Please provide a valid YouTube URL.")
        sys.exit(1)
         
    print("📥 Downloading audio...")
    audio_file = download_audio(youtube_link, "downloaded_audios")

    print("📝 Transcribing with WhisperX...")
    transcript = transcribe_with_whisperx(audio_file)

    print("\n=== RAW TRANSCRIPT ===\n")
    print(transcript)
