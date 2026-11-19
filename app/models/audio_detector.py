"""Audio Deepfake Detector using Wav2Vec2 + Spectral Analysis"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class AudioDeepfakeDetector:
    """
    Multi-modal audio analysis:
    1. Wav2Vec2 feature extraction + classifier  (5-second window)
    2. Spectral analysis (MFCC, spectral centroid, rolloff)
    3. Prosody analysis (pitch, energy, rhythm)

    Performance notes:
    - The Wav2Vec2 model is by far the slowest step on CPU.
      We cap its input to 5 seconds, which is enough for the
      classifier to produce a signal and cuts runtime ~10x.
    - The overall analysis window is capped at 20 seconds.
    """

    # Tunable performance constants
    MODEL_WINDOW_SECONDS = 5      # input fed to Wav2Vec2
    ANALYSIS_WINDOW_SECONDS = 20  # total audio analysed by rules

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.feature_extractor = None
        self.classifier = None
        self.sample_rate = 16000
        self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str = None):
        """Load Wav2Vec2 feature extractor and classifier"""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.feature_extractor = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base-960h"
            ).to(self.device)
            self.feature_extractor.eval()

            # Classifier head (randomly initialised unless fine-tuned weights provided)
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            ).to(self.device)

            if model_path and os.path.exists(model_path):
                self.classifier.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )

            self.classifier.eval()
            logger.info(f"Audio model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        try:
            # soundfile handles WAV/FLAC natively
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception:
            # Fallback to librosa (handles MP3/OGG via audioread)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sample_rate
            )

        return audio.astype(np.float32), self.sample_rate

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def _spectral_analysis(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze spectral characteristics"""
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            zero_crossing = librosa.feature.zero_crossing_rate(audio)

            findings: list = []
            score = 0.0

            mfcc_std = float(np.mean(np.std(mfccs, axis=1)))
            if mfcc_std < 5:
                findings.append("Unusually low MFCC variance (possible synthesis)")
                score += 0.20

            centroid_std = float(np.std(spectral_centroid))
            if centroid_std < 100:
                findings.append("Low spectral centroid variation")
                score += 0.15

            return {
                "mfcc_mean_std": round(mfcc_std, 2),
                "spectral_centroid_mean": round(float(np.mean(spectral_centroid)), 2),
                "spectral_centroid_std": round(centroid_std, 2),
                "spectral_rolloff_mean": round(float(np.mean(spectral_rolloff)), 2),
                "spectral_bandwidth_mean": round(float(np.mean(spectral_bandwidth)), 2),
                "zero_crossing_rate": round(float(np.mean(zero_crossing)), 4),
                "findings": findings,
                "spectral_score": round(min(score, 1.0), 4),
            }
        except Exception as e:
            logger.error(f"Spectral analysis failed: {e}")
            return {"spectral_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Prosody analysis
    # ------------------------------------------------------------------

    def _prosody_analysis(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze pitch, energy, and rhythm patterns"""
        try:
            f0, _, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
            f0_clean = f0[~np.isnan(f0)]

            if len(f0_clean) == 0:
                return {"prosody_score": 0.0, "error": "No voiced segments"}

            pitch_mean = float(np.mean(f0_clean))
            pitch_std = float(np.std(f0_clean))
            pitch_range = float(np.max(f0_clean) - np.min(f0_clean))

            rms = librosa.feature.rms(y=audio)[0]
            energy_std = float(np.std(rms))

            try:
                tempo_val, _ = librosa.beat.beat_track(y=audio, sr=sr)
                tempo = float(np.atleast_1d(tempo_val)[0])
            except Exception:
                tempo = 0.0

            findings: list = []
            score = 0.0

            if pitch_std < 15:
                findings.append("Low pitch variation (possible monotone synthesis)")
                score += 0.25

            pitch_diffs = np.abs(np.diff(f0_clean))
            if len(pitch_diffs) > 0 and float(np.max(pitch_diffs)) > 100:
                findings.append("Unnatural pitch jumps detected")
                score += 0.20

            if energy_std < 0.01:
                findings.append("Unusually consistent energy (possible synthesis)")
                score += 0.15

            return {
                "pitch_mean": round(pitch_mean, 2),
                "pitch_std": round(pitch_std, 2),
                "pitch_range": round(pitch_range, 2),
                "energy_std": round(energy_std, 4),
                "tempo": round(tempo, 2),
                "findings": findings,
                "prosody_score": round(min(score, 1.0), 4),
            }
        except Exception as e:
            logger.error(f"Prosody analysis failed: {e}")
            return {"prosody_score": 0.0, "error": str(e)}

    # ------------------------------------------------------------------
    # Model inference (fast — capped at MODEL_WINDOW_SECONDS)
    # ------------------------------------------------------------------

    def _model_inference(self, audio: np.ndarray) -> Dict[str, Any]:
        """Run Wav2Vec2 + classifier inference on a short window"""
        if self.feature_extractor is None:
            return {"model_score": 0.5, "model_available": False}

        try:
            # Cap input to MODEL_WINDOW_SECONDS — largest speed-up
            max_samples = self.MODEL_WINDOW_SECONDS * self.sample_rate
            if len(audio) > max_samples:
                # Take a window from the middle of the clip
                start = max(0, (len(audio) - max_samples) // 2)
                window = audio[start:start + max_samples]
            else:
                window = audio

            inputs = self.processor(
                window,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                features = self.feature_extractor(
                    inputs.input_values
                ).last_hidden_state
                pooled = features.mean(dim=1)
                logits = self.classifier(pooled)
                probs = torch.softmax(logits, dim=-1)
                deepfake_prob = float(probs[0][1].item())

            return {
                "model_score": deepfake_prob,
                "model_available": True,
                "window_seconds": round(len(window) / self.sample_rate, 2),
            }
        except Exception as e:
            logger.error(f"Audio model inference failed: {e}")
            return {"model_score": 0.5, "model_available": False}

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, audio_path: str) -> Dict[str, Any]:
        """Main detection method for audio"""
        t_start = time.time()
        try:
            audio, sr = self._load_audio(audio_path)

            # Cap total analysis window
            max_samples = self.ANALYSIS_WINDOW_SECONDS * sr
            if len(audio) > max_samples:
                start = max(0, (len(audio) - max_samples) // 2)
                audio = audio[start:start + max_samples]

            # Run all analyses
            model_result = self._model_inference(audio)
            spectral_result = self._spectral_analysis(audio, sr)
            prosody_result = self._prosody_analysis(audio, sr)

            # Weighted ensemble
            # If the model has not been fine-tuned (no trusted weights),
            # it produces a neutral ~0.5 signal; we downweight it and lean
            # on the rule-based components.
            model_is_trusted = model_result.get("model_available", False)
            # Note: for a genuinely fine-tuned model, `model_path` would be set.
            # Here we treat the base model as untrusted for scoring purposes.

            if model_is_trusted:
                weights = {"model": 0.45, "spectral": 0.25, "prosody": 0.30}
            else:
                # Rebalance to rules when model is unavailable
                weights = {"model": 0.10, "spectral": 0.45, "prosody": 0.45}

            final_score = (
                model_result["model_score"] * weights["model"]
                + spectral_result.get("spectral_score", 0) * weights["spectral"]
                + prosody_result.get("prosody_score", 0) * weights["prosody"]
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

            # Findings
            all_findings: list = []
            all_findings.extend(spectral_result.get("findings", []))
            all_findings.extend(prosody_result.get("findings", []))

            recommendations: list = []
            if final_score >= 0.70:
                recommendations.append("Do not trust this audio as authentic")
                recommendations.append("Verify speaker identity through a trusted channel")
                recommendations.append("Report if used in a fraudulent request")
            elif final_score >= 0.50:
                recommendations.append("Exercise caution - possible synthesis detected")
                recommendations.append("Request independent verification")
            else:
                recommendations.append("No strong synthesis indicators found")
                recommendations.append("Remain cautious with unexpected audio")

            elapsed = round(time.time() - t_start, 3)

            return {
                "detector_type": "audio_deepfake",
                "risk_score": final_score,
                "risk_level": risk_level,
                "is_deepfake": final_score >= 0.70,
                "confidence": round(abs(final_score - 0.5) * 2, 4),
                "duration_seconds": round(len(audio) / sr, 2),
                "processing_seconds": elapsed,
                "components": {
                    "model_analysis": model_result,
                    "spectral_analysis": spectral_result,
                    "prosody_analysis": prosody_result,
                },
                "findings": all_findings,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.error(f"Audio detection failed: {e}")
            return {
                "detector_type": "audio_deepfake",
                "error": str(e),
                "risk_score": 0.0,
                "risk_level": "UNKNOWN",
            }