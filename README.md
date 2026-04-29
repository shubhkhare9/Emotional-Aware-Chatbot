# Emotion-Aware Chatbot

This project packages the full emotion-classification pipeline into a runnable Python app. It includes the training and evaluation notebooks, the trained local RoBERTa classifier in `models/best_model`, generated report artifacts, and app entrypoints for CLI and web usage:

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
