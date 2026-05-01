"""Prediction history helpers."""

from datetime import datetime

import pandas as pd


def ensure_prediction_history(session_state):
    if "prediction_history" not in session_state:
        session_state.prediction_history = []


def add_prediction_history(session_state, filename, prediction, confidence, analysis_type):
    ensure_prediction_history(session_state)
    session_state.prediction_history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence,
            "status": "Healthy" if "healthy" in prediction.lower() else "Disease Detected",
            "analysis_type": analysis_type,
        }
    )


def filter_history(history, selected_type, selected_status):
    filtered_history = history.copy()
    if selected_type != "All":
        filtered_history = [
            item
            for item in filtered_history
            if item.get("analysis_type", "Unknown") == selected_type
        ]
    if selected_status != "All":
        filtered_history = [
            item for item in filtered_history if item["status"] == selected_status
        ]
    return filtered_history


def history_dataframe(history):
    history_df = pd.DataFrame(history)
    column_order = [
        "timestamp",
        "filename",
        "analysis_type",
        "prediction",
        "confidence",
        "status",
    ]
    available_columns = [col for col in column_order if col in history_df.columns]
    history_df = history_df[available_columns]

    if "confidence" in history_df.columns:
        history_df["confidence_display"] = history_df["confidence"].apply(
            lambda value: f"{value:.2%}" if isinstance(value, (int, float)) else str(value)
        )
        history_df = history_df.drop("confidence", axis=1)
        history_df = history_df.rename(columns={"confidence_display": "confidence"})

    return history_df
