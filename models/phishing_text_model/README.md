# Phishing Text Model

Fine-tuned DistilBERT for phishing email classification.

## Files
- `config.json` — Model configuration
- `pytorch_model.bin` — Model weights
- `tokenizer_config.json` — Tokenizer config
- `vocab.txt` — Tokenizer vocabulary

## Training
Trained on a labelled dataset of phishing and legitimate emails.
See `training/train_text_model.py` for the training script.

## Fallback
If this directory is empty, `PhishingTextDetector` falls back to
the base `distilbert-base-uncased` model and relies more heavily on
rule-based heuristics.