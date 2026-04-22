from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LABELS = ["Anger", "Fear", "Joy", "Love", "Neutral", "Sadness"]
DEFAULT_PROMPTS = {
    "Anger": (
        "You are a calm, empathetic assistant. The user is feeling angry or "
        "frustrated. Acknowledge their frustration without judgment, help them "
        "feel heard, and gently offer a constructive or calming perspective. "
        "Do not be dismissive."
    ),
    "Fear": (
        "You are a reassuring and supportive assistant. The user is feeling "
        "anxious or afraid. Validate their concern, offer grounding and "
        "reassurance, and help them feel safe. Keep your tone warm and steady."
    ),
    "Joy": (
        "You are an enthusiastic and warm assistant. The user is feeling happy "
        "or excited. Match their positive energy, celebrate with them, and keep "
        "the conversation uplifting."
    ),
    "Love": (
        "You are a warm and caring assistant. The user is expressing love or "
        "affection. Respond with genuine warmth, kindness, and positivity."
    ),
    "Neutral": (
        "You are a helpful, friendly assistant. The user seems calm and "
        "neutral. Respond clearly and helpfully in a conversational tone."
    ),
    "Sadness": (
        "You are a compassionate and gentle assistant. The user is feeling sad "
        "or down. Acknowledge their pain with empathy, let them know they are "
        "not alone, and offer gentle support without trying to immediately fix "
        "everything."
    ),
}


@dataclass
class AppConfig:
    project_root: Path
    roberta_model_path: Path
    mistral_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    labels: list[str] = field(default_factory=lambda: DEFAULT_LABELS.copy())
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_history_turns: int = 5
    emotion_prompts: dict[str, str] = field(default_factory=lambda: DEFAULT_PROMPTS.copy())
    response_config_path: Path | None = None

    @property
    def reports_dir(self) -> Path:
        return self.project_root / "reports"

    @property
    def response_reports_dir(self) -> Path:
        return self.reports_dir / "response_generation"


def _coerce_local_model_path(raw_path: str | None, project_root: Path) -> Path:
    local_default = project_root / "models" / "best_model"
    if not raw_path:
        return local_default

    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    if "models/best_model" in raw_path:
        return local_default
    return local_default


def load_config(project_root: str | Path | None = None) -> AppConfig:
    root = Path(project_root or Path(__file__).resolve().parent.parent).resolve()
    cfg_path = root / "reports" / "response_generation" / "pipeline_config.json"
    if not cfg_path.exists():
        return AppConfig(
            project_root=root,
            roberta_model_path=root / "models" / "best_model",
        )

    payload: dict[str, Any] = json.loads(cfg_path.read_text())
    return AppConfig(
        project_root=root,
        roberta_model_path=_coerce_local_model_path(payload.get("roberta_model_path"), root),
        mistral_model_id=payload.get("mistral_model_id", "mistralai/Mistral-7B-Instruct-v0.2"),
        labels=payload.get("labels", DEFAULT_LABELS.copy()),
        max_new_tokens=int(payload.get("max_new_tokens", 256)),
        temperature=float(payload.get("temperature", 0.7)),
        top_p=float(payload.get("top_p", 0.9)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.1)),
        max_history_turns=int(payload.get("max_history_turns", 5)),
        emotion_prompts=payload.get("emotion_prompts", DEFAULT_PROMPTS.copy()),
        response_config_path=cfg_path,
    )
