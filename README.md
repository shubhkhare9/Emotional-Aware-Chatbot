# Emotion-Aware Chatbot

This project packages the full emotion-classification pipeline into a runnable Python app. It covers data preparation, classical baseline comparison, RoBERTa fine-tuning, SHAP-based explainability, response generation, and chatbot delivery through both CLI and web interfaces. The repository includes training and evaluation notebooks, the trained local RoBERTa classifier in `models/best_model`, generated report artifacts, and app entrypoints for CLI and web usage:

- `main.py predict` for one-shot emotion classification
- `main.py chat` for terminal chat
- `app.py` or `main.py serve` for a Gradio web app

## Project layout

- `notebooks/`: end-to-end experimentation, training, explainability, and app notebooks
- `models/best_model`: trained RoBERTa emotion classifier
- `reports/`: exported metrics, charts, response-generation outputs, and report assets
- `reports/response_generation/pipeline_config.json`: generation settings and emotion prompts
- `emotion_chatbot/`: reusable runtime code
- `main.py`: CLI entrypoint
- `app.py`: Gradio app entrypoint
- `web_app.py`: custom HTML chatbot server

## Notebook workflow

The notebooks follow the project pipeline from model comparison to deployment:

- `02_model_comparison_tfidf_svm_bilstm.ipynb`: baseline model comparison
- `03_full_roberta_pipeline.ipynb`: RoBERTa training and evaluation pipeline
- `04_shap_explainability_fixed.ipynb`: SHAP explainability workflow
- `05_mistral_response_generation_fix.ipynb`: empathetic response-generation experiments
- `06_gradio_chatbot (1).ipynb`: Gradio chatbot prototyping

## Dataset details

The project is built around a processed emotion dataset pipeline stored under `data/` and summarized in `artifacts/`.

Primary source datasets:

- GoEmotions: `https://github.com/google-research/google-research/tree/master/goemotions`
- EmpatheticDialogues: `https://github.com/facebookresearch/EmpatheticDialogues`

GoEmotions provides fine-grained emotion annotations that are remapped into the project’s final classification targets, while EmpatheticDialogues supports the response-generation side of the chatbot by grounding it in emotionally aware conversational patterns.

- `data/interim/merged_raw.csv`: merged raw source data used before filtering and remapping
- `data/interim/balanced_dataset_with_flags.csv`: balanced dataset with augmentation flags used to equalize class counts
- `data/processed/train.csv`, `valid.csv`, `test.csv`: final model-ready splits

The final processed dataset contains `118,848` examples split as:

- train: `95,078`
- validation: `11,885`
- test: `11,885`

The emotion taxonomy is consolidated into 6 final classes:

- `Anger`: mapped from labels such as `anger`, `annoyance`, `disapproval`, `disgust`
- `Fear`: mapped from `fear`, `nervousness`
- `Joy`: mapped from fine-grained positive labels such as `amusement`, `excitement`, `gratitude`, `joy`, `optimism`, and related variants
- `Love`: mapped from `admiration`, `caring`, `desire`, `love`
- `Neutral`: mapped from `neutral`
- `Sadness`: mapped from `disappointment`, `embarrassment`, `grief`, `remorse`, `sadness`

Balancing metadata in `artifacts/balanced_class_distribution.csv` shows the final training target is evenly distributed at `19,808` samples per class after augmentation and resampling.

Filtering metadata in `artifacts/class_distribution.csv` documents how the raw merged dataset was curated, including retained non-neutral examples, neutral-only retention, and removed ambiguous or unmapped rows.

## Model details

The main classifier is a fine-tuned `RoBERTaForSequenceClassification` model stored in `models/best_model/`. It is used as the emotion detection backbone for the CLI, Gradio app, and HTML chatbot.

Core model configuration from `models/best_model/config.json`:

- architecture: `RoBERTaForSequenceClassification`
- encoder depth: `12` transformer layers
- hidden size: `768`
- attention heads: `12`
- feed-forward size: `3072`
- maximum position embeddings: `514`
- vocabulary size: `50,265`
- dropout: `0.1`

This architecture makes the model strong at sentence-level contextual understanding, which is important for distinguishing closely related affective signals such as `fear` vs `sadness` or `joy` vs `love`.

## Model architecture and pipeline

At a high level, the system follows this flow:

1. User text is cleaned and tokenized with the RoBERTa tokenizer.
2. The token sequence is passed through the 12-layer RoBERTa encoder.
3. A sequence-classification head predicts one of the 6 final emotion classes.
4. The predicted emotion is passed into the response layer, which either uses configurable LLM prompting or a template-based fallback generator.
5. The UI surfaces the label, confidence-oriented output, and optional explainability artifacts.

In practical terms, the architecture combines two layers:

- an emotion classification layer for understanding the user message
- a response generation layer for producing empathetic chatbot replies

The visual summaries in `reports/project_report_assets/roberta_architecture.png` and `reports/project_report_assets/system_pipeline.png` complement this overview and can be used in reports or presentations.

## Explainability

Model explainability is handled through the SHAP workflow captured in `04_shap_explainability_fixed.ipynb`.

- SHAP is used to estimate which words or tokens contributed most strongly to a predicted emotion.
- This helps verify whether the classifier is focusing on emotionally meaningful cues instead of irrelevant artifacts.
- Explainability is especially useful when comparing confusing classes, validating behavior on edge cases, and presenting the model in an academic or project-demo setting.

The Gradio application supports optional SHAP-based interpretation, although it can be slower than plain inference because explanation requires extra computation.

## Performance summary

Based on `reports/metrics.json`, the saved RoBERTa classifier achieves:

- accuracy: `0.8395`
- macro precision: `0.8351`
- macro recall: `0.8395`
- macro F1: `0.8354`
- weighted F1: `0.8354`

The recorded training run completed `10` epochs, and the additional CSV and image outputs under `reports/` provide deeper per-class and confusion-matrix analysis.

## Report artifacts

Generated outputs used for evaluation and reporting are stored under `reports/`, including:

- `metrics.json`, `classification_report.csv`, and `confusion_matrix.csv`
- `confusion_matrix.png` and `training_curves.png`
- `project_report_assets/` visual assets such as the architecture, pipeline, and model comparison charts
- `response_generation/` files for prompt configuration and batch-generated responses

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Predict one text:

```bash
python main.py predict "I am feeling nervous about tomorrow."
```

Open terminal chat:

```bash
python main.py chat
```

Launch the web app:

```bash
python app.py
```

Or:

```bash
python main.py serve
```

Launch the custom HTML chatbot:

```bash
python web_app.py
```

## Generator behavior

Emotion detection runs locally from the saved RoBERTa model.

Response generation tries to use the model from `reports/response_generation/pipeline_config.json`:

- default: `mistralai/Mistral-7B-Instruct-v0.2`

If that model is already cached or the machine can download and load it, the chatbot will use it. If not, the project falls back to a built-in empathetic template generator so the app still runs instead of crashing.

To force template replies:

```bash
EMOTION_CHATBOT_DISABLE_LLM=1 python app.py
```

To force Hugging Face loading:

```bash
python main.py --generator hf chat
```

## Notes

- SHAP explanations are optional in the Gradio app and can be slow.
- The HTML app lives in `templates/index.html` and is served by `web_app.py`.
- Loading Mistral-7B is resource-heavy; GPU is strongly recommended for full notebook-equivalent response generation.
- The current project folder does not include local Mistral weights, only the model ID reference, so first-time generation may require internet access.
