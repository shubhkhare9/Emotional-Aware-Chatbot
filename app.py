from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Emotion-Aware Chatbot app")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    from emotion_chatbot.app_ui import build_demo

    demo = build_demo(args.project_root)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
