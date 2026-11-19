"""Video Deepfake Detector combining Frame Analysis + Temporal Consistency"""
import os
import tempfile
import torch
import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple
from app.models.image_detector import ImageDeepfakeDetector

logger = logging.getLogger(__name__)


class VideoDeepfakeDetector:
    """
    Video analysis pipeline:
    1. Frame sampling and per-frame image analysis
    2. Temporal consistency checking (optical flow)
    3. Face tracking across frames
    4. (Future) Audio track analysis
    """

    def __init__(self, device: str = "cpu", max_frames: int = 30):
        self.device = device
        self.max_frames = max_frames
        self.image_detector = ImageDeepfakeDetector(device=device)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Extract evenly-spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0.0

        # Compute frame indices to sample
        if total_frames <= self.max_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)

        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()

        info = {
            "total_frames": total_frames,
            "fps": round(float(fps), 2) if fps else 0.0,
            "duration": round(float(duration), 2),
            "sampled_frames": len(frames),
            "extracted_indices": [int(i) for i in indices],
        }
        return frames, info

    # ------------------------------------------------------------------
    # Temporal analysis
    # ------------------------------------------------------------------

    def _temporal_analysis(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze temporal consistency between frames"""
        try:
            if len(frames) < 3:
                return {"temporal_score": 0.0, "error": "Insufficient frames"}

            findings: List[str] = []
            score = 0.0

            gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

            # Optical flow between consecutive frames
            flow_magnitudes: List[float] = []
            for i in range(len(gray_frames) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[i], gray_frames[i + 1],
                    None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_magnitudes.append(float(np.mean(magnitude)))

            flow_std = float(np.std(flow_magnitudes)) if flow_magnitudes else 0.0
            flow_mean = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0

            if flow_std > 2.0:
                findings.append("Inconsistent motion patterns between frames")
                score += 0.30

            if 0.0 < flow_mean < 0.5:
                findings.append("Unusually low motion (possible frame interpolation)")
                score += 0.20

            # Frame-to-frame difference
            frame_diffs = []
            for i in range(len(gray_frames) - 1):
                diff = np.mean(np.abs(
                    gray_frames[i].astype(float) - gray_frames[i + 1].astype(float)
                ))
                frame_diffs.append(float(diff))

            if frame_diffs and np.mean(frame_diffs) < 2.0:
                findings.append("Very low frame-to-frame variation")
                score += 0.20

            return {
                "flow_mean": round(flow_mean, 4),
                "flow_std": round(flow_std, 4),
                "frame_diff_mean": round(float(np.mean(frame_diffs)), 4) if frame_diffs else 0.0,
                "findings": findings,
                "temporal_score": round(min(score, 1.0), 4),
            }
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {"temporal_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Face tracking
    # ------------------------------------------------------------------

    def _face_tracking(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Track faces across frames for consistency"""
        try:
            findings: List[str] = []
            score = 0.0
            face_counts: List[int] = []
            face_positions: List[Tuple[int, int]] = []

            if self.face_cascade is None or self.face_cascade.empty():
                return {
                    "face_counts": [],
                    "avg_faces": 0.0,
                    "findings": ["Face cascade not available"],
                    "face_tracking_score": 0.0,
                }

            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                face_counts.append(len(faces))

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_positions.append((x + w // 2, y + h // 2))

            # Face count consistency
            if len(set(face_counts)) > 1:
                findings.append("Inconsistent face detection across frames")
                score += 0.15

            # Position smoothness
            if len(face_positions) > 2:
                positions = np.array(face_positions)
                diffs = np.diff(positions, axis=0)
                jump_distances = np.sqrt(np.sum(diffs ** 2, axis=1))

                if np.max(jump_distances) > 100:
                    findings.append("Sudden face position jumps detected")
                    score += 0.25

                if np.std(jump_distances) > 30:
                    findings.append("Irregular face movement patterns")
                    score += 0.20

            return {
                "face_counts": face_counts,
                "avg_faces": round(float(np.mean(face_counts)), 2) if face_counts else 0.0,
                "findings": findings,
                "face_tracking_score": round(min(score, 1.0), 4),
            }
        except Exception as e:
            logger.error(f"Face tracking failed: {e}")
            return {"face_tracking_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Frame-level image analysis (Windows-safe)
    # ------------------------------------------------------------------

    def _analyze_frames(self, frames: List[np.ndarray],
                        step: int = 5) -> Dict[str, Any]:
        """
        Run per-frame image analysis.
        Uses tempfile so paths are valid on Windows, Linux and macOS.
        """
        frame_scores: List[float] = []
        frame_errors: List[str] = []
        frames_analyzed = 0

        temp_paths: List[str] = []
        try:
            for i in range(0, len(frames), step):
                # Create cross-platform temp file
                fd, frame_path = tempfile.mkstemp(
                    prefix=f"dg_frame_{i}_", suffix=".jpg"
                )
                os.close(fd)
                temp_paths.append(frame_path)

                # Write frame
                try:
                    ok = cv2.imwrite(frame_path, frames[i])
                except Exception as e:
                    frame_errors.append(f"Frame {i} write error: {e}")
                    continue

                if not ok or not os.path.exists(frame_path):
                    frame_errors.append(f"Frame {i} not written")
                    continue

                # Analyze
                try:
                    result = self.image_detector.detect(frame_path)
                    frame_scores.append(float(result.get("risk_score", 0.5)))
                    frames_analyzed += 1
                except Exception as e:
                    frame_errors.append(f"Frame {i} analysis error: {e}")
        finally:
            # Cleanup
            for path in temp_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass

        return {
            "avg_score": round(float(np.mean(frame_scores)), 4) if frame_scores else 0.0,
            "max_score": round(float(np.max(frame_scores)), 4) if frame_scores else 0.0,
            "min_score": round(float(np.min(frame_scores)), 4) if frame_scores else 0.0,
            "frames_analyzed": frames_analyzed,
            "total_frames_in_video": len(frames),
            "frame_errors": frame_errors[:5],  # keep first few for debugging
        }

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, video_path: str) -> Dict[str, Any]:
        """Main detection method for video"""
        try:
            frames, video_info = self._extract_frames(video_path)

            if len(frames) == 0:
                return {
                    "detector_type": "video_deepfake",
                    "error": "No frames extracted",
                    "risk_score": 0.0,
                    "risk_level": "UNKNOWN",
                }

            # Per-frame image analysis
            frame_result = self._analyze_frames(frames, step=5)

            # Temporal + face tracking
            temporal_result = self._temporal_analysis(frames)
            face_result = self._face_tracking(frames)

            # Weighted ensemble
            weights = {
                "frame_avg": 0.35,
                "frame_max": 0.15,
                "temporal": 0.30,
                "face_tracking": 0.20,
            }

            final_score = (
                frame_result["avg_score"] * weights["frame_avg"]
                + frame_result["max_score"] * weights["frame_max"]
                + temporal_result.get("temporal_score", 0) * weights["temporal"]
                + face_result.get("face_tracking_score", 0) * weights["face_tracking"]
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

            # Combine findings
            all_findings: List[str] = []
            all_findings.extend(temporal_result.get("findings", []))
            all_findings.extend(face_result.get("findings", []))
            if frame_result["frame_errors"]:
                all_findings.append(
                    f"{len(frame_result['frame_errors'])} frame-level errors"
                )

            recommendations: List[str] = []
            if final_score >= 0.70:
                recommendations.append("Do not trust this video as authentic")
                recommendations.append("Verify through independent sources")
                recommendations.append("Report if used in fraudulent context")
            elif final_score >= 0.50:
                recommendations.append("Exercise caution - possible manipulation")
                recommendations.append("Seek additional verification")
            else:
                recommendations.append("No strong manipulation indicators found")
                recommendations.append("Remain cautious with unexpected media")

            return {
                "detector_type": "video_deepfake",
                "risk_score": final_score,
                "risk_level": risk_level,
                "is_deepfake": final_score >= 0.70,
                "confidence": round(abs(final_score - 0.5) * 2, 4),
                "video_info": video_info,
                "components": {
                    "frame_analysis": frame_result,
                    "temporal_analysis": temporal_result,
                    "face_tracking": face_result,
                },
                "findings": all_findings,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.error(f"Video detection failed: {e}", exc_info=True)
            return {
                "detector_type": "video_deepfake",
                "error": str(e),
                "risk_score": 0.0,
                "risk_level": "UNKNOWN",
            }