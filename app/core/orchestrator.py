"""Main detection orchestrator - routes files to appropriate detectors"""
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.config import settings
from app.models.text_detector import PhishingTextDetector
from app.models.image_detector import ImageDeepfakeDetector
from app.models.audio_detector import AudioDeepfakeDetector
from app.models.video_detector import VideoDeepfakeDetector
from app.models.logo_detector import LogoDetector
from app.core.integrity import compute_sha256, verify_file_unchanged

logger = logging.getLogger(__name__)

# UTF-8 BOM bytes and character
_BOM_BYTES = b'\xef\xbb\xbf'
_BOM_CHAR = '\ufeff'


class DetectionOrchestrator:
    """
    Main pipeline that:
    1. Validates input files
    2. Routes to appropriate detector(s)
    3. Combines results
    4. Verifies file integrity
    5. Generates comprehensive report
    """

    def __init__(self):
        self.text_detector = PhishingTextDetector(device=settings.DEVICE)
        self.image_detector = ImageDeepfakeDetector(device=settings.DEVICE)
        self.audio_detector = AudioDeepfakeDetector(device=settings.DEVICE)
        self.video_detector = VideoDeepfakeDetector(device=settings.DEVICE)
        self.logo_detector = LogoDetector()
        self.start_time = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        ext = Path(file_path).suffix.lower()

        if ext in settings.SUPPORTED_EMAIL_FORMATS:
            return "email"
        elif ext in settings.SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif ext in settings.SUPPORTED_AUDIO_FORMATS:
            return "audio"
        elif ext in settings.SUPPORTED_VIDEO_FORMATS:
            return "video"
        else:
            return "unknown"

    def _validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate file exists, size, and format"""
        if not os.path.exists(file_path):
            return {"valid": False, "error": "File does not exist"}

        file_size = os.path.getsize(file_path)
        file_type = self._get_file_type(file_path)

        size_limits = {
            "email": settings.MAX_EMAIL_SIZE_MB,
            "image": settings.MAX_IMAGE_SIZE_MB,
            "audio": settings.MAX_AUDIO_SIZE_MB,
            "video": settings.MAX_VIDEO_SIZE_MB,
        }

        max_size = size_limits.get(file_type, 50)
        size_mb = file_size / (1024 * 1024)

        if size_mb > max_size:
            return {
                "valid": False,
                "error": f"File too large: {size_mb:.2f}MB (max: {max_size}MB)",
                "file_size_mb": round(size_mb, 2),
                "max_size_mb": max_size,
            }

        if file_type == "unknown":
            return {
                "valid": False,
                "error": f"Unsupported file type: {Path(file_path).suffix}",
            }

        return {
            "valid": True,
            "file_type": file_type,
            "file_size_mb": round(size_mb, 4),
            "file_name": os.path.basename(file_path),
        }

    # ------------------------------------------------------------------
    # Email reading (BOM-safe + robust fallbacks)
    # ------------------------------------------------------------------

    def _read_email(self, email_path: str) -> Dict[str, str]:
        """
        Read email file (supports .eml, .msg, .txt).

        Handles:
        - UTF-8 BOM (added by PowerShell's Out-File)
        - Multipart and single-part messages
        - HTML-only bodies (converted to text)
        - Raw fallback if parsing fails
        """
        import email
        from email import policy

        with open(email_path, "rb") as f:
            content = f.read()

        # Strip UTF-8 BOM if present
        if content.startswith(_BOM_BYTES):
            content = content[len(_BOM_BYTES):]

        result = {
            "subject": "",
            "sender": "",
            "recipient": "",
            "body": "",
        }

        try:
            msg = email.message_from_bytes(content, policy=policy.default)

            result["subject"] = str(msg.get("subject", "") or "")
            result["sender"] = str(msg.get("from", "") or "")
            result["recipient"] = str(msg.get("to", "") or "")

            body_parts: List[str] = []

            if msg.is_multipart():
                # Prefer text/plain parts
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(
                                payload.decode("utf-8", errors="replace")
                            )

                # If no plain text, fall back to HTML → stripped text
                if not body_parts:
                    for part in msg.walk():
                        if part.get_content_type() == "text/html":
                            payload = part.get_payload(decode=True)
                            if payload:
                                html = payload.decode("utf-8", errors="replace")
                                body_parts.append(self._strip_html(html))
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    decoded = payload.decode("utf-8", errors="replace")
                    if msg.get_content_type() == "text/html":
                        decoded = self._strip_html(decoded)
                    body_parts.append(decoded)
                else:
                    body_parts.append(str(msg.get_payload()))

            result["body"] = "\n".join(body_parts).strip()

        except Exception as e:
            logger.warning(f"Email parsing failed ({e}); using raw text")

        # Fallback 1: if body is empty or suspiciously short, use raw content
        if len(result["body"].strip()) < 20:
            raw_text = content.decode("utf-8", errors="replace").lstrip(_BOM_CHAR)
            result["body"] = raw_text

        # Fallback 2: if sender/subject still empty, try to pull them from raw text
        if not result["subject"]:
            m = re.search(r"^Subject:\s*(.+)$", result["body"], re.MULTILINE | re.IGNORECASE)
            if m:
                result["subject"] = m.group(1).strip()

        if not result["sender"]:
            m = re.search(r"^From:\s*(.+)$", result["body"], re.MULTILINE | re.IGNORECASE)
            if m:
                result["sender"] = m.group(1).strip()

        if not result["recipient"]:
            m = re.search(r"^To:\s*(.+)$", result["body"], re.MULTILINE | re.IGNORECASE)
            if m:
                result["recipient"] = m.group(1).strip()

        # Log what we recovered (helps debugging)
        logger.debug(
            f"Email read: subject={result['subject']!r}, "
            f"sender={result['sender']!r}, "
            f"body_len={len(result['body'])}"
        )

        return result

    @staticmethod
    def _strip_html(html: str) -> str:
        """Remove HTML tags/scripts/styles from a string"""
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(self, file_path: str, analysis_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis method.

        Args:
            file_path: Path to file to analyze
            analysis_type: Optional override for file type

        Returns:
            Comprehensive analysis report
        """
        self.start_time = time.time()

        # Step 1: Compute original hash (integrity baseline)
        original_hash = compute_sha256(file_path)

        # Step 2: Validate file
        validation = self._validate_file(file_path)
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["error"],
                "file_hash": original_hash,
                "timestamp": datetime.now().isoformat(),
            }

        file_type = analysis_type or validation["file_type"]

        # Step 3: Route to appropriate detector
        try:
            if file_type == "email":
                email_data = self._read_email(file_path)

                # Ensure the detector always sees substantial text
                body_text = email_data["body"]
                if len(body_text.strip()) < 50:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        body_text = f.read().lstrip(_BOM_CHAR)

                result = self.text_detector.detect(
                    text=body_text,
                    subject=email_data["subject"],
                    sender=email_data["sender"],
                )
                result["email_metadata"] = {
                    "subject": email_data["subject"],
                    "sender": email_data["sender"],
                    "recipient": email_data["recipient"],
                }

            elif file_type == "image":
                result = self.image_detector.detect(file_path)
                # Also check for logos
                try:
                    logo_result = self.logo_detector.detect(file_path)
                    result["logo_analysis"] = logo_result
                except Exception as logo_err:
                    logger.warning(f"Logo detection failed: {logo_err}")
                    result["logo_analysis"] = {"error": str(logo_err)}

            elif file_type == "audio":
                result = self.audio_detector.detect(file_path)

            elif file_type == "video":
                result = self.video_detector.detect(file_path)

            else:
                return {
                    "success": False,
                    "error": f"Unsupported analysis type: {file_type}",
                    "file_hash": original_hash,
                }

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Detection failed: {str(e)}",
                "file_hash": original_hash,
                "timestamp": datetime.now().isoformat(),
            }

        # Step 4: Verify file integrity (file should be unchanged)
        integrity_check = verify_file_unchanged(file_path, original_hash)

        # Step 5: Calculate processing time
        processing_time = time.time() - self.start_time

        # Step 6: Build comprehensive report
        report = {
            "success": True,
            "analysis_id": f"DG-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "name": validation["file_name"],
                "type": file_type,
                "size_mb": validation["file_size_mb"],
                "sha256": original_hash,
            },
            "detection_result": result,
            "integrity": integrity_check,
            "performance": {
                "processing_time_seconds": round(processing_time, 3),
                "processing_time_ms": round(processing_time * 1000, 1),
            },
        }

        # Overall risk assessment
        report["overall_risk"] = {
            "score": result.get("risk_score", 0),
            "level": result.get("risk_level", "UNKNOWN"),
            "is_threat": result.get(
                "is_deepfake", result.get("is_phishing", False)
            ),
            "confidence": result.get("confidence", 0),
        }

        return report

    def analyze_batch(self, file_paths: list) -> Dict[str, Any]:
        """Analyze multiple files"""
        results = []
        for path in file_paths:
            results.append(self.analyze(path))

        return {
            "batch_id": f"BATCH-{int(time.time())}",
            "total_files": len(file_paths),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results,
        }