from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

from .config import AppConfig


EMOTION_COLORS = {
    "Anger": "#FF4B4B",
    "Fear": "#9B59B6",
    "Joy": "#F4D03F",
    "Love": "#FF6B9D",
    "Neutral": "#95A5A6",
    "Sadness": "#3498DB",
}

DEFAULT_SYSTEM = "You are a helpful and empathetic assistant."


@dataclass
class GeneratorStatus:
    backend: str
    ready: bool
    detail: str


class EmotionClassifier:
    def __init__(self, config: AppConfig) -> None:
        if not config.roberta_model_path.exists():
            raise FileNotFoundError(
                f"RoBERTa model not found at {config.roberta_model_path}. "
                "Expected local files under models/best_model."
            )
        self.config = config
        self.device = self._resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config.roberta_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.roberta_model_path)
        self.model.to(self.device)
        self.model.eval()
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.label2id = {label: index for index, label in enumerate(self.labels)}

    @staticmethod
    def _resolve_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def predict(self, text: str, max_length: int = 128) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()
        pred_idx = int(probs.argmax())
        return {
            "label": self.labels[pred_idx],
            "confidence": float(probs[pred_idx]),
            "scores": {label: float(prob) for label, prob in zip(self.labels, probs)},
        }

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        probs_all: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=128,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probs = torch.softmax(self.model(**encoded).logits, dim=-1).detach().cpu().numpy()
                probs_all.append(probs)
        return np.vstack(probs_all)


class TemplateResponseGenerator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.status = GeneratorStatus(
            backend="template",
            ready=True,
            detail="Using built-in empathetic response templates.",
        )

    def generate(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        history: list[list[str]] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        del history, max_new_tokens, temperature, top_p, repetition_penalty
        opener = {
            "Anger": "It sounds really frustrating, and your reaction makes sense.",
            "Fear": "That sounds scary, and I want to help you slow things down for a moment.",
            "Joy": "That sounds wonderful, and I’m glad you shared it.",
            "Love": "That’s genuinely warm and meaningful.",
            "Neutral": "I’m here with you and happy to help.",
            "Sadness": "I’m sorry you’re carrying that right now.",
        }.get(emotion, "I’m here with you.")
        follow_up = {
            "Anger": "If you want, tell me what happened and we can think through the next step together.",
            "Fear": "Take one slow breath, then tell me what feels most urgent right now.",
            "Joy": "What part of it feels most exciting to you?",
            "Love": "What would you like to say or do with that feeling?",
            "Neutral": "Tell me a bit more and I’ll do my best to help clearly.",
            "Sadness": "If you want, you can tell me more about what has been hardest.",
        }.get(emotion, "Tell me a bit more.")
        return (
            f"{opener} I detected `{emotion}` with {confidence:.1%} confidence from your message. "
            f"{follow_up}\n\nYou said: {user_text}"
        )


class HuggingFaceResponseGenerator:
    def __init__(self, config: AppConfig, model_id: str | None = None) -> None:
        self.config = config
        self.model_id = model_id or config.mistral_model_id
        self._model = None
        self._tokenizer = None
        self.status = GeneratorStatus(
            backend="huggingface",
            ready=False,
            detail=f"Model not loaded yet: {self.model_id}",
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = str(text)
        for token in ("<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"):
            cleaned = cleaned.replace(token, " ")
        return " ".join(cleaned.split()).strip()

    def _build_messages(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        history: list[list[str]] | None = None,
    ) -> list[dict[str, str]]:
        system = self.config.emotion_prompts.get(emotion, DEFAULT_SYSTEM)
        emotion_context = (
            f"[Detected emotion: {emotion} (confidence: {confidence:.0%})]\n"
            f"User message: {user_text}"
        )
        messages = [{"role": "system", "content": system}]
        for user_turn, bot_turn in (history or [])[-self.config.max_history_turns :]:
            messages.append({"role": "user", "content": self._clean_text(user_turn)})
            messages.append({"role": "assistant", "content": self._clean_text(bot_turn)})
        messages.append({"role": "user", "content": emotion_context})
        return messages

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}

        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                load_kwargs["dtype"] = torch.float16
        elif torch.backends.mps.is_available():
            load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._model = model
        self._tokenizer = tokenizer
        self.status = GeneratorStatus(
            backend="huggingface",
            ready=True,
            detail=f"Loaded generator: {self.model_id}",
        )

    def generate(
        self,
        user_text: str,
        emotion: str,
        confidence: float,
        history: list[list[str]] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> str:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None
        messages = self._build_messages(user_text, emotion, confidence, history)
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer(prompt_text, return_tensors="pt")
        if not hasattr(self._model, "hf_device_map"):
            model_inputs = {key: value.to(self._model.device) for key, value in model_inputs.items()}

        output_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            top_p=top_p if top_p is not None else self.config.top_p,
            repetition_penalty=(
                repetition_penalty
                if repetition_penalty is not None
                else self.config.repetition_penalty
            ),
            do_sample=True,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        prompt_length = model_inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][prompt_length:]
        reply = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self._clean_text(reply) or "I'm here with you. Tell me a little more."


class EmotionAwarePipeline:
    def __init__(self, config: AppConfig, prefer_generator: str = "auto") -> None:
        self.config = config
        self.classifier = EmotionClassifier(config)
        self.prefer_generator = prefer_generator
        self.generator_cache: dict[str, Any] = {}
        self.generator, self.generator_warning = self._build_generator(prefer_generator)

    def _build_generator(
        self,
        prefer_generator: str,
        model_id: str | None = None,
    ) -> tuple[Any, str | None]:
        if os.getenv("EMOTION_CHATBOT_DISABLE_LLM") == "1" or prefer_generator == "template":
            return TemplateResponseGenerator(self.config), None

        try:
            generator = HuggingFaceResponseGenerator(self.config, model_id=model_id)
            if prefer_generator == "hf":
                generator._ensure_loaded()
            return generator, None
        except Exception as exc:
            warning = (
                "Falling back to template responses because the Hugging Face "
                f"generator could not be prepared: {exc}"
            )
            return TemplateResponseGenerator(self.config), warning

    def get_generator(
        self,
        prefer_generator: str | None = None,
        model_id: str | None = None,
    ) -> tuple[Any, str | None]:
        effective_preference = prefer_generator or self.prefer_generator
        cache_key = f"{effective_preference}:{model_id or self.config.mistral_model_id}"
        if cache_key not in self.generator_cache:
            generator, warning = self._build_generator(effective_preference, model_id=model_id)
            self.generator_cache[cache_key] = generator
            if effective_preference == self.prefer_generator and model_id is None:
                self.generator = generator
                self.generator_warning = warning
            return generator, warning
        return self.generator_cache[cache_key], None

    def predict_emotion(self, text: str) -> dict[str, Any]:
        return self.classifier.predict(text)

    def generate_response(
        self,
        user_text: str,
        history: list[list[str]] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        generator_model_id: str | None = None,
        generator_preference: str | None = None,
    ) -> dict[str, Any]:
        emotion_result = self.predict_emotion(user_text)
        generator, warning = self.get_generator(
            prefer_generator=generator_preference,
            model_id=generator_model_id,
        )
        try:
            reply = generator.generate(
                user_text=user_text,
                emotion=emotion_result["label"],
                confidence=emotion_result["confidence"],
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )
        except Exception as exc:
            generator = TemplateResponseGenerator(self.config)
            warning = (
                "Falling back to template responses because the Hugging Face "
                f"generator failed while generating: {exc}"
            )
            reply = generator.generate(
                user_text=user_text,
                emotion=emotion_result["label"],
                confidence=emotion_result["confidence"],
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )
        return {
            "user_text": user_text,
            "emotion": emotion_result["label"],
            "confidence": emotion_result["confidence"],
            "all_scores": emotion_result["scores"],
            "reply": reply,
            "generator_backend": generator.status.backend,
            "generator_status": warning or generator.status.detail,
            "generator_model_id": getattr(generator, "model_id", "template"),
        }

    def get_shap_html(self, text: str, emotion_label: str) -> str:
        try:
            import shap
        except Exception as exc:
            return f"<p style='color:gray;'>SHAP is not installed: {exc}</p>"

        try:
            masker = shap.maskers.Text(self.classifier.tokenizer)
            explainer = shap.Explainer(
                lambda texts: self.classifier.predict_proba([str(text) for text in texts]),
                masker,
                output_names=self.classifier.labels,
            )
            shap_values = explainer([text], batch_size=4)
            class_idx = self.classifier.label2id[emotion_label]
            token_vals = shap_values.values[0][:, class_idx]
            token_data = shap_values.data[0]
            abs_max = max(abs(float(token_vals.max())), abs(float(token_vals.min())), 1e-9)
            norm_vals = token_vals / abs_max
            parts = ["<div style='font-size:15px;line-height:2.2;font-family:monospace;'>"]
            for token, raw_value, norm_value in zip(token_data, token_vals, norm_vals):
                token_str = str(token)
                if token_str in {"<s>", "</s>", "<pad>", ""}:
                    continue
                token_str = token_str.replace("Ġ", " ").replace("▁", " ")
                if norm_value > 0:
                    alpha = min(float(norm_value) * 0.8, 0.85)
                    bg = f"rgba(46,204,113,{alpha:.2f})"
                else:
                    alpha = min(abs(float(norm_value)) * 0.8, 0.85)
                    bg = f"rgba(231,76,60,{alpha:.2f})"
                parts.append(
                    "<span "
                    f"title='SHAP: {float(raw_value):.4f}' "
                    f"style='background:{bg};border-radius:3px;padding:1px 4px;margin:1px;'>"
                    f"{token_str}</span>"
                )
            parts.append("</div>")
            return "".join(parts)
        except Exception as exc:
            return f"<p style='color:gray;'>SHAP unavailable: {exc}</p>"

    @staticmethod
    def make_confidence_chart(scores: dict[str, float]) -> plt.Figure:
        labels = list(scores.keys())
        values = [scores[label] for label in labels]
        colors = [EMOTION_COLORS.get(label, "#888") for label in labels]
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_title("Emotion Scores", fontsize=11, fontweight="bold")
        for bar, value in zip(bars, values):
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    @staticmethod
    def make_trend_chart(emotion_history: list[tuple[int, str, float]]) -> plt.Figure:
        if not emotion_history:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No conversation yet", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.axis("off")
            return fig

        turns = [item[0] for item in emotion_history]
        emotions = [item[1] for item in emotion_history]
        confidences = [item[2] for item in emotion_history]
        colors = [EMOTION_COLORS.get(emotion, "#888") for emotion in emotions]

        fig, ax = plt.subplots(figsize=(max(6, len(turns) * 1.2), 3))
        for turn, emotion, confidence, color in zip(turns, emotions, confidences, colors):
            ax.scatter(turn, confidence, color=color, s=160, zorder=5)
            ax.annotate(
                emotion,
                (turn, confidence),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
        if len(turns) > 1:
            ax.plot(turns, confidences, color="#BDC3C7", linewidth=1.2, zorder=1)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(turns)
        ax.set_xticklabels([f"Turn {turn}" for turn in turns], fontsize=8)
        ax.set_ylabel("Confidence", fontsize=9)
        ax.set_title("Emotion Trend", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig
