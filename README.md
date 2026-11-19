# 🛡️ DeepGuard AI

**AI-Generated Phishing Email & Deepfake Media Detector**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Educational-lightgrey)]()

---

## 📖 Overview

DeepGuard AI is a defensive cybersecurity tool that detects:

- 🎣 **AI-generated phishing emails** — transformer-based text classification + URL/header heuristics
- 🖼️ **Deepfake images** — CNN classifier + Error Level Analysis + metadata forensics
- 🔊 **Deepfake audio** — Wav2Vec2 features + spectral & prosody analysis
- 🎥 **Deepfake videos** — per-frame image analysis + temporal consistency + face tracking

The tool guarantees **file integrity**: every file is SHA-256 hashed before and after analysis, and a `integrity_preserved` flag is returned in every report.

> ⚠️ **Educational use only.** This is a prototype, not a certified security product. Results should always be reviewed by a human analyst.

---

## 🏗️ Architecture
