from __future__ import annotations

import argparse
import base64
import io
import threading
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request, session

from emotion_chatbot import EmotionAwarePipeline, load_config

MODEL_OPTIONS = {
    "qwen": {
        "label": "Qwen 1.5B",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Faster replies, lighter model",
    },
    "mistral": {
        "label": "Mistral 7B",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Slower but stronger long-form replies",
    },
}


def figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def create_app(project_root: str | Path | None = None, generator: str = "auto") -> Flask:
    config = load_config(project_root)
    pipeline = EmotionAwarePipeline(config, prefer_generator=generator)

    app = Flask(__name__, template_folder="templates")
    app.secret_key = "emotion-aware-chatbot-local-dev"
    state_store: dict[str, dict[str, list]] = {}
    shap_jobs: dict[str, dict[str, str | bool]] = {}

    def get_state() -> dict[str, list]:
        session_id = session.get("chatbot_session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            session["chatbot_session_id"] = session_id
        return state_store.setdefault(
            session_id,
            {
                "chat_history": [],
                "emotion_history": [],
            },
        )

    @app.get("/")
    def index():
        default_option = MODEL_OPTIONS["qwen"]
        return render_template(
            "index.html",
            generator_backend="huggingface",
            generator_status=(
                f"Choose a generator. {default_option['label']} is recommended for speed; "
                "the model loads on first use."
            ),
            model_options=MODEL_OPTIONS,
            default_model_choice="qwen",
            config=config,
        )

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        user_message = str(payload.get("message", "")).strip()
        max_tokens = int(payload.get("max_new_tokens", config.max_new_tokens))
        temperature = float(payload.get("temperature", config.temperature))
        run_shap = bool(payload.get("run_shap", False))
        model_choice = str(payload.get("model_choice", "qwen")).strip().lower()

        if not user_message:
            return jsonify({"error": "Please enter a message."}), 400
        if model_choice not in MODEL_OPTIONS:
            return jsonify({"error": "Invalid generator choice."}), 400

        state = get_state()
        result = pipeline.generate_response(
            user_text=user_message,
            history=state["chat_history"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            generator_model_id=MODEL_OPTIONS[model_choice]["model_id"],
        )
        emotion = result["emotion"]
        confidence = result["confidence"]
        scores = result["all_scores"]

        turn = len(state["emotion_history"]) + 1
        state["chat_history"].append([user_message, result["reply"]])
        state["emotion_history"].append((turn, emotion, confidence))

        confidence_fig = pipeline.make_confidence_chart(scores)
        confidence_image = figure_to_data_uri(confidence_fig)

        trend_fig = pipeline.make_trend_chart(state["emotion_history"])
        trend_image = figure_to_data_uri(trend_fig)

        shap_job_id = None
        if run_shap:
            shap_job_id = str(uuid.uuid4())
            shap_jobs[shap_job_id] = {
                "ready": False,
                "html": "<div class='subtle'>Computing SHAP highlights...</div>",
            }

            def compute_shap(job_id: str, text: str, label: str) -> None:
                html = pipeline.get_shap_html(text, label)
                shap_jobs[job_id] = {"ready": True, "html": html}

            thread = threading.Thread(
                target=compute_shap,
                args=(shap_job_id, user_message, emotion),
                daemon=True,
            )
            thread.start()
            shap_html = shap_jobs[shap_job_id]["html"]
        else:
            shap_html = (
                "<div class='shap-placeholder-card'>"
                f"<strong>{emotion}</strong> detected with <strong>{confidence:.1%}</strong> confidence."
                f"<div class='shap-subtle'>{result['generator_status']}</div>"
                "</div>"
            )

        return jsonify(
            {
                "message": user_message,
                "reply": result["reply"],
                "emotion": emotion,
                "confidence": confidence,
                "scores": scores,
                "confidence_chart": confidence_image,
                "trend_chart": trend_image,
                "shap_html": shap_html,
                "shap_job_id": shap_job_id,
                "chat_history": state["chat_history"],
                "emotion_history": state["emotion_history"],
                "generator_backend": result["generator_backend"],
                "generator_status": result["generator_status"],
                "generator_choice": model_choice,
                "generator_label": MODEL_OPTIONS[model_choice]["label"],
            }
        )

    @app.get("/api/shap/<job_id>")
    def shap_status(job_id: str):
        job = shap_jobs.get(job_id)
        if not job:
            return jsonify({"error": "SHAP job not found."}), 404
        return jsonify(job)

    @app.post("/api/reset")
    def reset():
        state = get_state()
        state["chat_history"].clear()
        state["emotion_history"].clear()
        empty_trend = figure_to_data_uri(pipeline.make_trend_chart([]))
        return jsonify(
            {
                "ok": True,
                "trend_chart": empty_trend,
                "message": "Conversation reset.",
            }
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the HTML chatbot app")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--generator", choices=["auto", "hf", "template"], default="auto")
    args = parser.parse_args()

    app = create_app(args.project_root, generator=args.generator)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
