"""Model loading and prediction utilities."""

import json
import pickle
from dataclasses import dataclass

import torch
import streamlit as st

from plant_disease_model import PlantDiseaseModel


@dataclass
class PredictionResult:
    prediction: str
    confidence: float
    top3_classes: list
    top3_confidences: list
    predicted_index: int


@st.cache_resource
def load_model():
    """Load the trained model and inference artifacts once per Streamlit session."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("class_names.json") as f:
        classes = json.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)

    model = PlantDiseaseModel(len(classes))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    return model, transform, device, encoder, classes


def predict_image(image, model, transform, device, encoder):
    """Predict disease for a single PIL image."""
    img = image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        top3_prob, top3_idx = torch.topk(probabilities, 3)

    top3_classes = [encoder.inverse_transform([idx.item()])[0] for idx in top3_idx[0]]
    top3_confidences = [prob.item() for prob in top3_prob[0]]

    return PredictionResult(
        prediction=encoder.inverse_transform([pred.item()])[0],
        confidence=confidence.item(),
        top3_classes=top3_classes,
        top3_confidences=top3_confidences,
        predicted_index=pred.item(),
    )
