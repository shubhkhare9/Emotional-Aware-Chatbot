from __future__ import annotations

from pathlib import Path

import gradio as gr

from .config import load_config
from .pipeline import EMOTION_COLORS, EmotionAwarePipeline


def build_demo(project_root: str | Path | None = None) -> gr.Blocks:
    config = load_config(project_root)
    pipeline = EmotionAwarePipeline(config)

    def chat(
        user_message: str,
        chat_history: list[list[str]],
        emotion_history: list[tuple[int, str, float]],
        max_tokens: int,
        temperature: float,
        run_shap: bool,
    ):
        if not user_message.strip():
            yield chat_history, None, "<p>Please type a message.</p>", None, emotion_history
            return

        result = pipeline.generate_response(
            user_text=user_message,
            history=chat_history,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        emotion = result["emotion"]
        confidence = result["confidence"]
        scores = result["all_scores"]
        color = EMOTION_COLORS.get(emotion, "#888")

        turn = len(emotion_history) + 1
        emotion_history = emotion_history + [(turn, emotion, confidence)]
        conf_fig = pipeline.make_confidence_chart(scores)
        trend_fig = pipeline.make_trend_chart(emotion_history)

        if run_shap:
            shap_html = pipeline.get_shap_html(user_message, emotion)
        else:
            shap_html = (
                f"<div style='padding:10px;background:#f8f9fa;border-radius:8px;'>"
                f"<b style='color:{color}'>{emotion}</b> detected with "
                f"<b>{confidence:.1%}</b> confidence.<br>"
                f"<small style='color:gray;'>{result['generator_status']}</small></div>"
            )

        updated_history = chat_history + [[user_message, result["reply"]]]
        yield updated_history, conf_fig, shap_html, trend_fig, emotion_history

    def reset_chat():
        return [], "<p style='color:gray;'>Conversation reset.</p>", None, [], pipeline.make_trend_chart([])

    with gr.Blocks(
        title="Emotion-Aware Chatbot",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
            .chat-col { min-height: 520px; }
            .panel { background: #f8fafc; border-radius: 12px; padding: 12px; }
            footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            "# Emotion-Aware Chatbot\n"
            "**RoBERTa** detects emotion from your local trained model. "
            "**Mistral** is used for generation when available; otherwise the app "
            "falls back to a built-in empathetic response template."
        )

        emotion_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3, elem_classes="chat-col"):
                chatbot = gr.Chatbot(label="Conversation", height=420, bubble_full_width=False)
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=5,
                        lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary", scale=1)
                    shap_toggle = gr.Checkbox(
                        label="Show SHAP highlights",
                        value=False,
                        scale=2,
                    )

            with gr.Column(scale=2):
                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### Emotion Analysis")
                    conf_plot = gr.Plot(label="Confidence Scores")

                with gr.Group(elem_classes="panel"):
                    gr.Markdown("### SHAP Token Highlights")
                    shap_display = gr.HTML(
                        value="<p style='color:gray;'>Send a message to see the prediction details.</p>"
                    )

        with gr.Row():
            with gr.Group(elem_classes="panel"):
                gr.Markdown("### Emotion Trend Across Conversation")
                trend_plot = gr.Plot(value=pipeline.make_trend_chart([]), label="Emotion Trend")

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                max_tokens_slider = gr.Slider(minimum=64, maximum=512, value=config.max_new_tokens, step=32, label="Max new tokens")
                temperature_slider = gr.Slider(minimum=0.1, maximum=1.2, value=config.temperature, step=0.05, label="Temperature")

        gr.Examples(
            examples=[
                ["I just got promoted today and I still cannot believe it."],
                ["I am so tired of being ignored by everyone around me."],
                ["My grandmother passed away last week and I miss her so much."],
                ["I am terrified about my exam results tomorrow."],
                ["I love spending evenings with my family."],
                ["Can you explain how machine learning works?"],
            ],
            inputs=msg_box,
            label="Try examples",
        )

        send_inputs = [msg_box, chatbot, emotion_state, max_tokens_slider, temperature_slider, shap_toggle]
        send_outputs = [chatbot, conf_plot, shap_display, trend_plot, emotion_state]

        send_btn.click(fn=chat, inputs=send_inputs, outputs=send_outputs).then(lambda: "", outputs=msg_box)
        msg_box.submit(fn=chat, inputs=send_inputs, outputs=send_outputs).then(lambda: "", outputs=msg_box)
        reset_btn.click(
            fn=reset_chat,
            outputs=[chatbot, shap_display, conf_plot, emotion_state, trend_plot],
        )

    return demo
