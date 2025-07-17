from datetime import datetime
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod


from utils.logger import logger

class TranscriptionWriter(ABC):
    """Abstract base class for transcription writers"""
    
    @abstractmethod
    def write(self, result, output_path: str) -> None:
        """Write transcription result to file"""
        pass


class PlainTextWriter(TranscriptionWriter):
    """Write transcription as plain text"""
    
    def write(self, result, output_path: str) -> None:
        """Write transcription to plain text file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.text)
            logger.info(f"Plain text transcription written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write plain text: {e}")
            raise


class SRTWriter(TranscriptionWriter):
    """Write transcription as SRT subtitle file"""
    
    def write(self, result, output_path: str) -> None:
        """Write transcription to SRT subtitle file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                subtitle_count = 1
                
                for segment in result.segments:
                    text = segment.get('text', '').strip()
                    if not text:
                        continue
                    
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', start_time + 2)  # Default 2 seconds if no end time
                    
                    # Format timestamps
                    start_srt = self._seconds_to_srt_time(start_time)
                    end_srt = self._seconds_to_srt_time(end_time)
                    
                    # Write SRT entry
                    f.write(f"{subtitle_count}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{text}\n\n")
                    
                    subtitle_count += 1
            
            logger.info(f"SRT transcription written to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write SRT: {e}")
            raise
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

   

class TranscriptionFileManager:
    """Manages writing transcription results to multiple file formats"""
    
    def __init__(self):
        self.writers = {
            'txt': PlainTextWriter(),
            'srt': SRTWriter(),
        }
    
    def write_transcription(self, result, output_path: str, formats: List[str] = None) -> List[str]:
        """
        Write transcription to file(s)
        
        Args:
            result: TranscriptionResult object
            output_path: Base output path (without extension)
            formats: List of formats to write (e.g., ['txt', 'json', 'srt'])
        
        Returns:
            List of created file paths
        """
        if formats is None:
            formats = ['txt', 'json']
        
        created_files = []
        base_path = Path(output_path)
        
        for format_type in formats:
            if format_type not in self.writers:
                logger.warning(f"Unknown format: {format_type}")
                continue
            
            try:
                file_path = base_path.with_suffix(f'.{format_type}')
                self.writers[format_type].write(result, str(file_path))
                created_files.append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to write {format_type} format: {e}")
        
        return created_files
    
    def write_summary_report(self, result, output_path: str) -> str:
        """Write a comprehensive summary report"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("TRANSCRIPTION SUMMARY REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {result.duration:.2f} seconds\n")
                f.write(f"Overall Confidence: {result.confidence:.2f}\n")
                f.write(f"Total Segments: {len(result.segments)}\n")
                f.write(f"Word Count: {len(result.text.split())}\n\n")
                
                f.write("METADATA:\n")
                f.write("-" * 20 + "\n")
                for key, value in result.metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                f.write("FULL TRANSCRIPTION:\n")
                f.write("-" * 20 + "\n")
                f.write(result.text)
                f.write("\n\n")
                
                f.write("SEGMENT BREAKDOWN:\n")
                f.write("-" * 20 + "\n")
                for i, segment in enumerate(result.segments):
                    f.write(f"Segment {i+1}:\n")
                    f.write(f"  Text: {segment.get('text', '')}\n")
                    f.write(f"  Start: {segment.get('start', 0):.2f}s\n")
                    f.write(f"  End: {segment.get('end', 0):.2f}s\n")
                    f.write("\n")
            
            logger.info(f"Summary report written to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to write summary report: {e}")
            raise
