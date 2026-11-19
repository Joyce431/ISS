"""Image Deepfake Detector using EfficientNet + Forensic Analysis"""
import os
import tempfile
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import timm
import cv2
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ImageDeepfakeDetector:
    """
    Multi-modal image analysis:
    1. EfficientNet-B4 deepfake classifier
    2. Error Level Analysis (ELA)
    3. Metadata forensics
    4. Noise pattern analysis
    5. Face detection & landmark analysis
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str = None):
        """Load EfficientNet-B4 for deepfake detection"""
        try:
            self.model = timm.create_model(
                'efficientnet_b4',
                pretrained=True,
                num_classes=2,
            )

            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Image model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Error Level Analysis (cross-platform)
    # ------------------------------------------------------------------

    def _error_level_analysis(self, image_path: str) -> Dict[str, Any]:
        """
        ELA detects image manipulation by analyzing compression artifacts.

        Uses tempfile so the temp JPEG works on Windows, Linux and macOS
        (hard-coded /tmp/ fails on Windows).
        """
        temp_path = None
        try:
            original = Image.open(image_path).convert('RGB')

            # Cross-platform temp file
            fd, temp_path = tempfile.mkstemp(prefix="dg_ela_", suffix=".jpg")
            os.close(fd)

            # Save at known quality and compare
            original.save(temp_path, 'JPEG', quality=90)
            compressed = Image.open(temp_path)

            # Compute difference
            diff = (
                np.array(original, dtype=np.float32)
                - np.array(compressed, dtype=np.float32)
            )
            ela_image = np.abs(diff)

            # Analyze ELA statistics
            mean_ela = float(np.mean(ela_image))
            max_ela = float(np.max(ela_image))
            std_ela = float(np.std(ela_image))

            # High ELA in localized regions indicates manipulation
            ela_score = 0.0
            if max_ela > 50:
                ela_score += 0.30
            if std_ela > 15:
                ela_score += 0.20
            if mean_ela > 10:
                ela_score += 0.20

            return {
                "mean_ela": round(mean_ela, 2),
                "max_ela": round(max_ela, 2),
                "std_ela": round(std_ela, 2),
                "ela_score": round(min(ela_score, 1.0), 4),
                "suspicious": ela_score > 0.40,
            }
        except Exception as e:
            logger.error(f"ELA failed: {e}")
            return {"ela_score": 0.0, "error": str(e)}
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Metadata analysis
    # ------------------------------------------------------------------

    def _metadata_analysis(self, image_path: str) -> Dict[str, Any]:
        """Analyze image metadata for manipulation signs"""
        try:
            from PIL.ExifTags import TAGS
            image = Image.open(image_path)
            exif_data = image.getexif()

            findings: List[str] = []
            risk_score = 0.0

            if not exif_data:
                findings.append("No EXIF metadata (possible editing)")
                risk_score += 0.15

            metadata: Dict[str, str] = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[str(tag)] = str(value)

            # Check for editing software
            editing_software = [
                'photoshop', 'gimp', 'lightroom', 'affinity',
                'paint', 'canva', 'midjourney', 'dall-e', 'stable diffusion',
            ]
            software = metadata.get('Software', '').lower()
            for sw in editing_software:
                if sw in software:
                    findings.append(f"Editing software detected: {sw}")
                    risk_score += 0.30

            return {
                "has_exif": bool(exif_data),
                "metadata": metadata,
                "findings": findings,
                "metadata_score": round(min(risk_score, 1.0), 4),
            }
        except Exception as e:
            return {"metadata_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Noise analysis
    # ------------------------------------------------------------------

    def _noise_analysis(self, image_path: str) -> Dict[str, Any]:
        """Analyze noise patterns for inconsistencies"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"noise_score": 0.0, "error": "Could not read image"}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Laplacian variance (blur detection)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Noise estimation
            noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
            noise_std = float(np.std(noise))

            # Block artifact detection (JPEG compression)
            h, w = gray.shape
            block_scores = []
            for i in range(0, h - 8, 8):
                for j in range(0, w - 8, 8):
                    block = gray[i:i + 8, j:j + 8]
                    block_scores.append(float(np.std(block)))

            block_variance = float(np.std(block_scores)) if block_scores else 0.0

            noise_score = 0.0
            findings: List[str] = []

            if laplacian_var < 50:
                findings.append("Low image sharpness (possible smoothing)")
                noise_score += 0.20
            if noise_std > 15:
                findings.append("High noise inconsistency")
                noise_score += 0.25
            if block_variance > 20:
                findings.append("Inconsistent compression artifacts")
                noise_score += 0.20

            return {
                "laplacian_variance": round(laplacian_var, 2),
                "noise_std": round(noise_std, 2),
                "block_variance": round(block_variance, 2),
                "findings": findings,
                "noise_score": round(min(noise_score, 1.0), 4),
            }
        except Exception as e:
            return {"noise_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Face analysis
    # ------------------------------------------------------------------

    def _face_analysis(self, image_path: str) -> Dict[str, Any]:
        """Detect faces and analyze facial regions for deepfake artifacts"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"faces_detected": 0, "face_score": 0.0}

            if self.face_cascade is None or self.face_cascade.empty():
                return {
                    "faces_detected": 0,
                    "face_score": 0.0,
                    "error": "Face cascade unavailable",
                }

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            face_count = len(faces)
            findings: List[str] = []
            face_score = 0.0

            for (x, y, w, h) in faces:
                face_region = img[y:y + h, x:x + w]

                if w > 0 and h > 0:
                    # Edge consistency
                    edges = cv2.Canny(face_region, 50, 150)
                    edge_density = np.sum(edges > 0) / (w * h)
                    if edge_density > 0.15:
                        findings.append(
                            "High edge density in facial region (possible blending)"
                        )
                        face_score += 0.15

                    # Colour consistency
                    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                    hue_std = np.std(hsv[:, :, 0])
                    if hue_std > 30:
                        findings.append("Color inconsistency in facial region")
                        face_score += 0.20

            return {
                "faces_detected": face_count,
                "findings": findings,
                "face_score": round(min(face_score, 1.0), 4),
            }
        except Exception as e:
            return {"faces_detected": 0, "face_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _model_inference(self, image_path: str) -> Dict[str, Any]:
        """Run EfficientNet deepfake classifier"""
        if self.model is None:
            return {"model_score": 0.5, "model_available": False}

        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=-1)
                deepfake_prob = probs[0][1].item()

            return {"model_score": deepfake_prob, "model_available": True}
        except Exception as e:
            logger.error(f"Image model inference failed: {e}")
            return {"model_score": 0.5, "model_available": False}

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, image_path: str) -> Dict[str, Any]:
        """Main detection method for images"""
        # Run all analysis components
        model_result = self._model_inference(image_path)
        ela_result = self._error_level_analysis(image_path)
        metadata_result = self._metadata_analysis(image_path)
        noise_result = self._noise_analysis(image_path)
        face_result = self._face_analysis(image_path)

        # Weighted ensemble
        weights = {
            "model": 0.40,
            "ela": 0.20,
            "metadata": 0.10,
            "noise": 0.15,
            "face": 0.15,
        }

        final_score = (
            model_result["model_score"] * weights["model"]
            + ela_result.get("ela_score", 0) * weights["ela"]
            + metadata_result.get("metadata_score", 0) * weights["metadata"]
            + noise_result.get("noise_score", 0) * weights["noise"]
            + face_result.get("face_score", 0) * weights["face"]
        )
        final_score = round(float(final_score), 4)

        # Risk level
        if final_score >= 0.85:
            risk_level = "CRITICAL"
        elif final_score >= 0.70:
            risk_level = "HIGH"
        elif final_score >= 0.50:
            risk_level = "MEDIUM"
        elif final_score >= 0.30:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        # Collect findings
        all_findings: List[str] = []
        all_findings.extend(ela_result.get("findings", []))
        all_findings.extend(metadata_result.get("findings", []))
        all_findings.extend(noise_result.get("findings", []))
        all_findings.extend(face_result.get("findings", []))

        recommendations: List[str] = []
        if final_score >= 0.70:
            recommendations.append("Do not trust this image as authentic")
            recommendations.append("Verify through independent sources")
            recommendations.append("Report if used in fraudulent context")
        elif final_score >= 0.50:
            recommendations.append("Exercise caution - possible manipulation detected")
            recommendations.append("Seek additional verification")
        else:
            recommendations.append("No strong manipulation indicators found")
            recommendations.append("Remain cautious with unexpected media")

        return {
            "detector_type": "image_deepfake",
            "risk_score": final_score,
            "risk_level": risk_level,
            "is_deepfake": final_score >= 0.70,
            "confidence": round(abs(final_score - 0.5) * 2, 4),
            "components": {
                "model_analysis": model_result,
                "ela_analysis": ela_result,
                "metadata_analysis": metadata_result,
                "noise_analysis": noise_result,
                "face_analysis": face_result,
            },
            "findings": all_findings,
            "recommendations": recommendations,
        }