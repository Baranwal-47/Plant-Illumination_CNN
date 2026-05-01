"""AgentRouter consultation client for second-opinion reports."""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.disease_info import format_label, get_disease_info

DEFAULT_AGENT_ROUTER_BASE_URL = "https://agentrouter.org/v1"
DEFAULT_AGENT_ROUTER_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class AgentRouterConfig:
    api_key: str
    base_url: str
    model: str


class ConsultationError(Exception):
    """Raised when an AI consultation cannot be completed."""


def _load_env_file(path=".env"):
    if not os.path.exists(path):
        return {}

    values = {}
    raw_key = None
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue
            if "=" in clean_line:
                key, value = clean_line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
            elif raw_key is None:
                raw_key = clean_line

    if raw_key:
        values.setdefault("RAW_AGENT_ROUTER_KEY", raw_key)
    return values


def load_agent_router_config():
    """Load AgentRouter config from process env or the local .env file."""
    file_values = _load_env_file()

    api_key = (
        os.getenv("AGENT_ROUTER_TOKEN")
        or os.getenv("AGENT_ROUTER_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or file_values.get("AGENT_ROUTER_TOKEN")
        or file_values.get("AGENT_ROUTER_API_KEY")
        or file_values.get("ANTHROPIC_API_KEY")
        or file_values.get("RAW_AGENT_ROUTER_KEY")
    )
    base_url = (
        os.getenv("AGENT_ROUTER_BASE_URL")
        or file_values.get("AGENT_ROUTER_BASE_URL")
        or DEFAULT_AGENT_ROUTER_BASE_URL
    )
    model = (
        os.getenv("AGENT_ROUTER_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or file_values.get("AGENT_ROUTER_MODEL")
        or file_values.get("ANTHROPIC_MODEL")
        or DEFAULT_AGENT_ROUTER_MODEL
    )

    if not api_key:
        raise ConsultationError(
            "AgentRouter API key was not found. Add it to .env or set AGENT_ROUTER_TOKEN."
        )

    return AgentRouterConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
    )


def _chat_completions_url(base_url):
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


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


def request_ai_consultation(prediction, assessment, user_notes=""):
    """Request a short second-opinion report from AgentRouter."""
    config = load_agent_router_config()
    prompt_payload = build_consultation_prompt(prediction, assessment, user_notes)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an agricultural AI assistant. Give cautious, concise plant-health "
                "guidance from the provided CNN prediction context only. Do not claim a "
                "definitive diagnosis. Mention when expert review or lab testing is needed."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create a second-consultation report for this plant leaf analysis. "
                "Use sections: Likely interpretation, Why uncertain, Immediate next steps, "
                "When to ask an expert.\n\n"
                f"{json.dumps(prompt_payload, indent=2)}"
            ),
        },
    ]
    body = json.dumps(
        {
            "model": config.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 600,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        _chat_completions_url(config.base_url),
        data=body,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ConsultationError(f"AgentRouter request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConsultationError(f"Could not reach AgentRouter: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ConsultationError("AgentRouter request timed out.") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ConsultationError("AgentRouter returned an unexpected response format.") from exc
