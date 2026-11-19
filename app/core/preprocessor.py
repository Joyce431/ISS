"""File preprocessing for detection pipeline"""
import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from app.config import settings
from app.utils.file_utils import (
    get_file_extension, get_file_size_mb, ensure_directory,
    get_temp_path, safe_filename
)

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Preprocessing pipeline for all media types.
    
    Handles:
    - File validation (size, format)
    - Type detection
    - Extraction (email headers/body, image metadata, audio samples, video frames)
    - Normalization (resize, resample, decode)
    """
    
    def __init__(self):
        self.max_sizes = {
            "email": settings.MAX_EMAIL_SIZE_MB,
            "image": settings.MAX_IMAGE_SIZE_MB,
            "audio": settings.MAX_AUDIO_SIZE_MB,
            "video": settings.MAX_VIDEO_SIZE_MB,
        }
        self.supported = {
            "email": settings.SUPPORTED_EMAIL_FORMATS,
            "image": settings.SUPPORTED_IMAGE_FORMATS,
            "audio": settings.SUPPORTED_AUDIO_FORMATS,
            "video": settings.SUPPORTED_VIDEO_FORMATS,
        }
    
    # ------------------------------------------------------------------
    # Type detection & validation
    # ------------------------------------------------------------------
    
    def detect_type(self, file_path: str) -> str:
        """Detect media type from extension"""
        ext = get_file_extension(file_path)
        for media_type, extensions in self.supported.items():
            if ext in extensions:
                return media_type
        return "unknown"
    
    def validate(self, file_path: str,
                 override_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate a file before processing.
        
        Returns dict with keys: valid, media_type, size_mb, error (if any)
        """
        if not os.path.isfile(file_path):
            return {"valid": False, "error": "File does not exist"}
        
        media_type = override_type or self.detect_type(file_path)
        
        if media_type == "unknown":
            ext = get_file_extension(file_path)
            return {
                "valid": False,
                "error": f"Unsupported file extension: '{ext}'",
                "supported": self.supported
            }
        
        size_mb = get_file_size_mb(file_path)
        max_size = self.max_sizes.get(media_type, 50)
        
        if size_mb > max_size:
            return {
                "valid": False,
                "error": f"File too large: {size_mb:.2f} MB (max {max_size} MB)",
                "size_mb": size_mb,
                "max_size_mb": max_size
            }
        
        if size_mb == 0:
            return {"valid": False, "error": "File is empty"}
        
        return {
            "valid": True,
            "media_type": media_type,
            "size_mb": size_mb,
            "name": os.path.basename(file_path),
            "path": file_path
        }
    
    # ------------------------------------------------------------------
    # Email preprocessing
    # ------------------------------------------------------------------
    
    def preprocess_email(self, file_path: str) -> Dict[str, Any]:
        """
        Parse email file (.eml, .msg, .txt) into structured data.
        
        Returns dict with: subject, sender, recipient, date, body,
                           headers, urls, attachments, raw_text
        """
        import email
        from email import policy
        
        result = {
            "subject": "",
            "sender": "",
            "recipient": "",
            "date": "",
            "body": "",
            "headers": {},
            "urls": [],
            "attachments": [],
            "raw_text": ""
        }
        
        try:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read()
            
            msg = email.message_from_bytes(raw_bytes, policy=policy.default)
            
            result["subject"] = str(msg.get('subject', ''))
            result["sender"] = str(msg.get('from', ''))
            result["recipient"] = str(msg.get('to', ''))
            result["date"] = str(msg.get('date', ''))
            
            # Collect all headers
            for key in msg.keys():
                result["headers"][key] = str(msg.get(key, ''))
            
            # Extract body
            if msg.is_multipart():
                body_parts = []
                for part in msg.walk():
                    content_type = part.get_content_type()
                    disposition = str(part.get('Content-Disposition', ''))
                    
                    if 'attachment' in disposition:
                        result["attachments"].append({
                            "filename": part.get_filename(),
                            "content_type": content_type,
                            "size": len(part.get_payload(decode=True) or b'')
                        })
                    elif content_type == 'text/plain':
                        try:
                            body_parts.append(
                                part.get_payload(decode=True).decode('utf-8', errors='replace')
                            )
                        except Exception:
                            pass
                    elif content_type == 'text/html' and not body_parts:
                        try:
                            html = part.get_payload(decode=True).decode('utf-8', errors='replace')
                            body_parts.append(self._strip_html(html))
                        except Exception:
                            pass
                
                result["body"] = "\n".join(body_parts)
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    result["body"] = payload.decode('utf-8', errors='replace')
                else:
                    result["body"] = str(msg.get_payload())
            
            # Extract URLs from body
            result["urls"] = self._extract_urls(result["body"])
            
            # Combined raw text for analysis
            result["raw_text"] = f"{result['subject']}\n{result['body']}"
            
        except Exception as e:
            logger.warning(f"Email parsing failed, using raw text: {e}")
            from app.utils.file_utils import read_text_file
            text = read_text_file(file_path, max_bytes=5_000_000)
            result["body"] = text
            result["raw_text"] = text
            result["urls"] = self._extract_urls(text)
        
        return result
    
    def _strip_html(self, html: str) -> str:
        """Remove HTML tags from string"""
        import re
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        import re
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return list(set(re.findall(pattern, text, re.IGNORECASE)))
    
    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    
    def preprocess_image(self, file_path: str) -> Dict[str, Any]:
        """
        Preprocess image file.
        
        Returns dict with: width, height, format, mode, exif,
                           normalized_array, original_size
        """
        from PIL import Image, ExifTags
        
        result = {
            "width": 0,
            "height": 0,
            "format": "",
            "mode": "",
            "exif": {},
            "has_alpha": False,
            "aspect_ratio": 0.0
        }
        
        try:
            with Image.open(file_path) as img:
                result["width"] = img.width
                result["height"] = img.height
                result["format"] = img.format or "UNKNOWN"
                result["mode"] = img.mode
                result["has_alpha"] = img.mode in ('RGBA', 'LA', 'P')
                result["aspect_ratio"] = round(img.width / max(img.height, 1), 4)
                
                # EXIF extraction
                try:
                    exif = img.getexif()
                    if exif:
                        for tag_id, value in exif.items():
                            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                            # Only keep JSON-serializable values
                            if isinstance(value, (str, int, float)):
                                result["exif"][str(tag)] = value
                            else:
                                result["exif"][str(tag)] = str(value)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def normalize_image(self, file_path: str,
                        target_size: Tuple[int, int] = (380, 380)) -> Optional[np.ndarray]:
        """
        Load and normalize image to a numpy array.
        
        Returns float32 array of shape (H, W, 3) in range [0, 1],
        or None if loading fails.
        """
        from PIL import Image
        
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                img = img.resize(target_size, Image.LANCZOS)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                return arr
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            return None
    
    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------
    
    def preprocess_audio(self, file_path: str,
                         target_sr: int = 16000,
                         max_duration: float = 60.0) -> Dict[str, Any]:
        """
        Preprocess audio file.
        
        Returns dict with: sample_rate, duration, channels, samples, waveform
        """
        import librosa
        import soundfile as sf
        
        result = {
            "sample_rate": target_sr,
            "duration": 0.0,
            "channels": 1,
            "samples": 0,
            "waveform": None
        }
        
        try:
            # Try soundfile first
            try:
                audio, sr = sf.read(file_path)
                if audio.ndim > 1:
                    result["channels"] = audio.shape[1]
                    audio = np.mean(audio, axis=1)
            except Exception:
                audio, sr = librosa.load(file_path, sr=None, mono=True)
                result["channels"] = 1
            
            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            
            # Truncate
            max_samples = int(max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Normalize to float32 in [-1, 1]
            audio = audio.astype(np.float32)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            
            result["sample_rate"] = sr
            result["samples"] = len(audio)
            result["duration"] = round(len(audio) / sr, 3)
            result["waveform"] = audio
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def extract_audio_from_video(self, video_path: str,
                                 output_path: Optional[str] = None) -> Optional[str]:
        """
        Extract audio track from a video file using ffmpeg.
        Returns path to extracted WAV, or None if no audio.
        """
        import subprocess
        
        if output_path is None:
            output_path = get_temp_path(prefix="audio_", suffix=".wav")
        
        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                output_path
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            
            if result.returncode != 0 or not os.path.exists(output_path):
                logger.info(f"No audio track in {video_path}")
                return None
            
            if os.path.getsize(output_path) < 1000:
                return None
            
            return output_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return None
    
    # ------------------------------------------------------------------
    # Video preprocessing
    # ------------------------------------------------------------------
    
    def preprocess_video(self, file_path: str,
                         max_frames: int = 30) -> Dict[str, Any]:
        """
        Preprocess video file: extract metadata and sampled frames.
        
        Returns dict with: fps, frame_count, duration, width, height,
                           frames (list of numpy arrays), sampled_indices
        """
        import cv2
        
        result = {
            "fps": 0.0,
            "frame_count": 0,
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "frames": [],
            "sampled_indices": []
        }
        
        cap = None
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                result["error"] = "Cannot open video"
                return result
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0.0
            
            result["fps"] = round(fps, 2)
            result["frame_count"] = total_frames
            result["duration"] = round(duration, 2)
            result["width"] = width
            result["height"] = height
            
            # Compute sampled indices
            if total_frames <= max_frames:
                indices = list(range(total_frames))
            else:
                indices = list(np.linspace(0, total_frames - 1, max_frames, dtype=int))
            
            result["sampled_indices"] = [int(i) for i in indices]
            
            # Read frames
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
            
            result["frames"] = frames
            result["frames_extracted"] = len(frames)
            
        except Exception as e:
            logger.error(f"Video preprocessing failed: {e}")
            result["error"] = str(e)
        finally:
            if cap is not None:
                cap.release()
        
        return result
    
    # ------------------------------------------------------------------
    # Generic dispatcher
    # ------------------------------------------------------------------
    
    def preprocess(self, file_path: str,
                   media_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Dispatch to the correct preprocessor based on media type.
        
        Returns a dict containing 'media_type' and the preprocessed data.
        """
        validation = self.validate(file_path, override_type=media_type)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"]}
        
        media_type = validation["media_type"]
        output = {
            "success": True,
            "media_type": media_type,
            "file_path": file_path,
            "file_size_mb": validation["size_mb"]
        }
        
        if media_type == "email":
            output["data"] = self.preprocess_email(file_path)
        elif media_type == "image":
            output["data"] = self.preprocess_image(file_path)
        elif media_type == "audio":
            output["data"] = self.preprocess_audio(file_path)
        elif media_type == "video":
            output["data"] = self.preprocess_video(file_path)
        
        return output
    
    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    
    def batch_validate(self, file_paths: List[str]) -> Dict[str, Any]:
        """Validate a batch of files"""
        results = []
        for path in file_paths:
            results.append(self.validate(path))
        
        return {
            "total": len(file_paths),
            "valid": sum(1 for r in results if r["valid"]),
            "invalid": sum(1 for r in results if not r["valid"]),
            "results": results
        }