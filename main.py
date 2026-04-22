from __future__ import annotations

import argparse
import json

from emotion_chatbot import EmotionAwarePipeline, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emotion-aware chatbot CLI")
    parser.add_argument("--project-root", default=".", help="Path to the project root")
    parser.add_argument(
        "--generator",
        choices=["auto", "hf", "template"],
        default="auto",
        help="Response generator backend",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict emotion for one text")
    predict_parser.add_argument("text", help="Input text")

    chat_parser = subparsers.add_parser("chat", help="Open an interactive terminal chat")
    chat_parser.add_argument("--max-new-tokens", type=int, default=None)
    chat_parser.add_argument("--temperature", type=float, default=None)

    serve_parser = subparsers.add_parser("serve", help="Launch the Gradio app")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=7860)
    serve_parser.add_argument("--share", action="store_true")

    return parser


def run_predict(pipeline: EmotionAwarePipeline, text: str) -> int:
    result = pipeline.predict_emotion(text)
    print(json.dumps(result, indent=2))
    return 0


def run_chat(
    pipeline: EmotionAwarePipeline,
    max_new_tokens: int | None,
    temperature: float | None,
) -> int:
    print("Type 'exit' to quit.\n")
    history: list[list[str]] = []
    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return 0
        result = pipeline.generate_response(
            user_text=user_text,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        history.append([user_text, result["reply"]])
        print(f"Emotion: {result['emotion']} ({result['confidence']:.1%})")
        print(f"Generator: {result['generator_backend']} - {result['generator_status']}")
        print(f"Bot: {result['reply']}\n")


def run_serve(project_root: str, host: str, port: int, share: bool) -> int:
    from emotion_chatbot.app_ui import build_demo

    demo = build_demo(project_root)
    demo.launch(server_name=host, server_port=port, share=share)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        return run_serve(args.project_root, args.host, args.port, args.share)

    config = load_config(args.project_root)
    pipeline = EmotionAwarePipeline(config, prefer_generator=args.generator)
    if pipeline.generator_warning:
        print(pipeline.generator_warning)

    if args.command == "predict":
        return run_predict(pipeline, args.text)
    if args.command == "chat":
        return run_chat(pipeline, args.max_new_tokens, args.temperature)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
