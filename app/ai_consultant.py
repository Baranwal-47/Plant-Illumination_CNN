"""Gemini consultation client for second-opinion reports."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.disease_info import format_label, get_disease_info

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


@dataclass
class GeminiConfig:
    api_key: str
    base_url: str
    model: str


class ConsultationError(Exception):
    """Raised when an AI consultation cannot be completed."""


def _load_env_file(path=".env"):
    if not os.path.exists(path):
        return {}

    values = {}
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
                continue
            key, value = clean_line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def load_gemini_config():
    """Load Gemini config from the local .env file or process environment."""
    file_values = _load_env_file()

    api_key = file_values.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    base_url = (
        file_values.get("GEMINI_BASE_URL")
        or os.getenv("GEMINI_BASE_URL")
        or DEFAULT_GEMINI_BASE_URL
    )
    model = (
        file_values.get("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )

    if not api_key:
        raise ConsultationError("Gemini API key was not found. Add GEMINI_API_KEY to .env.")

    return GeminiConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
    )


def _gemini_generate_url(config):
    return f"{config.base_url}/models/{config.model}:generateContent"


def _format_gemini_error(exc):
    detail = exc.read().decode("utf-8", errors="ignore")
    try:
        payload = json.loads(detail)
        message = payload.get("error", {}).get("message") or detail
    except json.JSONDecodeError:
        message = detail
    return f"Gemini request failed ({exc.code}): {message}"


def build_consultation_prompt(prediction, assessment, user_notes):
    disease_info = get_disease_info(prediction.prediction) or {}
    top_predictions = [
        {
            "class": class_name,
            "label": format_label(class_name),
            "confidence": round(confidence, 4),
        }
        for class_name, confidence in zip(
            prediction.top3_classes,
            prediction.top3_confidences,
        )
    ]

    return {
        "detected_class": prediction.prediction,
        "detected_label": format_label(prediction.prediction),
        "confidence": round(prediction.confidence, 4),
        "confidence_margin": round(assessment.confidence_margin, 4),
        "status": assessment.status,
        "uncertainty_reasons": assessment.reasons,
        "top_predictions": top_predictions,
        "known_disease_info": disease_info,
        "user_notes": user_notes.strip(),
    }


def _build_gemini_prompt(prompt_payload):
    return (
        "You are an agricultural AI assistant for a plant disease detection app. "
        "Use only the provided CNN prediction context and disease notes. Do not claim "
        "a definitive diagnosis. Give cautious, practical guidance and mention when "
        "expert review or lab testing is needed.\n\n"
        "Create a second-consultation report using these sections:\n"
        "1. Likely interpretation\n"
        "2. Why the model may be uncertain\n"
        "3. Immediate next steps\n"
        "4. When to ask an expert\n\n"
        f"Prediction context:\n{json.dumps(prompt_payload, indent=2)}"
    )


def request_ai_consultation(prediction, assessment, user_notes=""):
    """Request a short second-opinion report from Gemini."""
    config = load_gemini_config()
    prompt_payload = build_consultation_prompt(prediction, assessment, user_notes)

    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "text": _build_gemini_prompt(prompt_payload),
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 7000,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        _gemini_generate_url(config),
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": config.api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ConsultationError(_format_gemini_error(exc)) from exc
    except urllib.error.URLError as exc:
        raise ConsultationError(f"Could not reach Gemini: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ConsultationError("Gemini request timed out.") from exc

    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ConsultationError("Gemini returned an unexpected response format.") from exc
