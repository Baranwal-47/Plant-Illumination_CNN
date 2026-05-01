"""Prediction uncertainty helpers."""

from dataclasses import dataclass


DEFAULT_MARGIN_THRESHOLD = 0.15


@dataclass
class UncertaintyAssessment:
    is_uncertain: bool
    status: str
    confidence_margin: float
    reasons: list


def assess_prediction(prediction, confidence, top_confidences, threshold, margin_threshold):
    """Assess whether a model prediction needs human review."""
    reasons = []
    top_confidences = list(top_confidences)
    confidence_margin = (
        top_confidences[0] - top_confidences[1] if len(top_confidences) > 1 else 1.0
    )

    if confidence < threshold:
        reasons.append(
            f"confidence is below the selected threshold ({confidence:.2%} < {threshold:.2%})"
        )

    if confidence_margin < margin_threshold:
        reasons.append(
            f"top predictions are close together (gap {confidence_margin:.2%} < {margin_threshold:.2%})"
        )

    if reasons:
        return UncertaintyAssessment(
            is_uncertain=True,
            status="Needs Expert Review",
            confidence_margin=confidence_margin,
            reasons=reasons,
        )

    return UncertaintyAssessment(
        is_uncertain=False,
        status="Healthy" if "healthy" in prediction.lower() else "Disease Detected",
        confidence_margin=confidence_margin,
        reasons=[],
    )
