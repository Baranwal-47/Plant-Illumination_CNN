"""Plotly chart builders for the Streamlit app."""

import pandas as pd
import plotly.express as px

from app.disease_info import format_label


def create_confidence_chart(top3_classes, top3_confidences):
    """Create a confidence visualization chart."""
    df = pd.DataFrame(
        {
            "Disease": [format_label(class_name) for class_name in top3_classes],
            "Confidence": [confidence * 100 for confidence in top3_confidences],
        }
    )

    fig = px.bar(
        df,
        x="Confidence",
        y="Disease",
        orientation="h",
        title="Top 3 Predictions",
        color="Confidence",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(height=300)
    return fig


def create_status_distribution_chart(history):
    status_counts = pd.Series([item["status"] for item in history]).value_counts()
    return px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Health Status Distribution",
    )


def create_analysis_type_chart(history):
    type_counts = pd.Series(
        [item.get("analysis_type", "Unknown") for item in history]
    ).value_counts()
    return px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Analysis Type Distribution",
    )


def create_confidence_distribution_chart(history):
    numerical_confidences = [
        item["confidence"]
        for item in history
        if isinstance(item["confidence"], (int, float))
    ]
    if not numerical_confidences:
        return None

    return px.histogram(
        x=numerical_confidences,
        title="Confidence Score Distribution",
        nbins=20,
        labels={"x": "Confidence Score", "y": "Count"},
    )
