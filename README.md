# Emotion-Aware Chatbot

This project packages your notebook pipeline into a runnable Python app. It uses the trained local RoBERTa classifier in `models/best_model` and exposes:

- `main.py predict` for one-shot emotion classification
- `main.py chat` for terminal chat
- `app.py` or `main.py serve` for a Gradio web app

## Project layout

- `models/best_model`: trained RoBERTa emotion classifier
- `reports/response_generation/pipeline_config.json`: generation settings and emotion prompts
- `emotion_chatbot/`: reusable runtime code
- `main.py`: CLI entrypoint
- `app.py`: Gradio app entrypoint

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
