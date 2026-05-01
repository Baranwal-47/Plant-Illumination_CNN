from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from app.charts import (
    create_analysis_type_chart,
    create_confidence_chart,
    create_confidence_distribution_chart,
    create_status_distribution_chart,
)
from app.ai_consultant import ConsultationError, request_ai_consultation
from app.disease_info import format_label, get_disease_info
from app.uncertainty import DEFAULT_MARGIN_THRESHOLD, assess_prediction
from app.history import (
    add_prediction_history,
    ensure_prediction_history,
    filter_history,
    history_dataframe,
)
from app.gradcam import create_gradcam_overlay
from app.model_utils import load_model, predict_image


st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon=":herb:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def display_disease_info(disease_name):
    """Display detailed information about the detected disease."""
    info = get_disease_info(disease_name)
    if not info:
        st.info("No detailed disease information is available for this class yet.")
        return

    with st.expander(f"About {format_label(disease_name)}", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Description:**")
            st.write(info["description"])
            st.write("**Symptoms:**")
            st.write(info["symptoms"])

        with col2:
            st.write("**Treatment:**")
            st.write(info["treatment"])
            st.write("**Prevention:**")
            st.write(info["prevention"])


def display_prediction_result(prediction, confidence, threshold, assessment):
    """Display the primary prediction status and confidence meter."""
    formatted_prediction = format_label(prediction)

    if assessment.is_uncertain:
        st.warning(f"Needs expert review: {formatted_prediction}")
        for reason in assessment.reasons:
            st.write(f"- {reason}")
    elif confidence >= threshold:
        if "healthy" in prediction.lower():
            st.success("Healthy plant detected")
            st.balloons()
        else:
            st.error(f"Disease detected: {formatted_prediction}")
    else:
        st.warning(f"Low confidence prediction: {formatted_prediction}")
        st.warning(f"Confidence ({confidence:.2%}) is below threshold ({threshold:.2%})")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Confidence Level", f"{confidence:.2%}")
    with metric_col2:
        st.metric("Top-1 vs Top-2 Gap", f"{assessment.confidence_margin:.2%}")
    st.progress(confidence, text=f"Confidence: {confidence:.2%}")


def display_ai_consultation(prediction, assessment):
    """Render the optional AgentRouter second-consultation panel."""
    with st.expander("AI Second Consultation", expanded=assessment.is_uncertain):
        user_notes = st.text_area(
            "Optional field notes",
            placeholder="Example: crop age, weather, watering pattern, visible spots, location...",
        )
        if st.button("Ask AI for second consultation", width="stretch"):
            with st.spinner("Asking AgentRouter for a cautious second opinion..."):
                try:
                    consultation = request_ai_consultation(
                        prediction,
                        assessment,
                        user_notes=user_notes,
                    )
                    st.markdown(consultation)
                except ConsultationError as exc:
                    st.warning(f"AI consultation unavailable: {exc}")
                except Exception as exc:
                    st.warning(f"AI consultation unavailable: {exc}")


def render_sidebar(classes, device):
    """Render controls and return selected options."""
    with st.sidebar:
        st.header("Controls")
        st.info(
            f"""
            **Model Statistics:**
            - Classes: {len(classes)}
            - Device: {device}
            - Status: Ready
            """
        )

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for reliable predictions",
        )
        margin_threshold = st.slider(
            "Close Prediction Gap",
            min_value=0.0,
            max_value=0.5,
            value=DEFAULT_MARGIN_THRESHOLD,
            step=0.01,
            help="Flag results when the top two predictions are too close",
        )

        if st.checkbox("Show Available Disease Classes"):
            st.write("**Detectable Diseases:**")
            for i, class_name in enumerate(classes, 1):
                st.write(f"{i}. {format_label(class_name)}")

    return confidence_threshold, margin_threshold


def render_single_image_tab(
    model,
    transform,
    device,
    encoder,
    confidence_threshold,
    margin_threshold,
):
    st.header("Single Image Analysis")

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a plant leaf for disease detection",
    )

    if not uploaded_file:
        return

    image = Image.open(uploaded_file)
    file_size = getattr(uploaded_file, "size", None)
    if file_size is None:
        file_size = len(uploaded_file.getbuffer())
    analysis_key = f"{uploaded_file.name}:{file_size}"

    preview_col, detail_col = st.columns([1.25, 0.75])

    with preview_col:
        st.image(image, caption="Uploaded Image", width="stretch")

    with detail_col:
        st.write("**Image Details:**")
        st.write(f"- Size: {image.size}")
        st.write(f"- Mode: {image.mode}")
        st.write(f"- Format: {uploaded_file.type}")

        analyze_clicked = st.button("Analyze Disease", type="primary", width="stretch")

    if analyze_clicked:
        with st.spinner("Analyzing image and generating Grad-CAM..."):
            try:
                prediction = predict_image(image, model, transform, device, encoder)
                assessment = assess_prediction(
                    prediction.prediction,
                    prediction.confidence,
                    prediction.top3_confidences,
                    confidence_threshold,
                    margin_threshold,
                )
                heatmap = create_gradcam_overlay(
                    image,
                    model,
                    transform,
                    device,
                    prediction.predicted_index,
                )
                add_prediction_history(
                    st.session_state,
                    uploaded_file.name,
                    prediction.prediction,
                    prediction.confidence,
                    "Single Image",
                    status=assessment.status,
                )
                st.session_state.single_image_analysis = {
                    "key": analysis_key,
                    "prediction": prediction,
                    "assessment": assessment,
                    "heatmap": heatmap,
                }
            except Exception as exc:
                st.error(f"Error during prediction: {exc}")
                return

    saved_analysis = st.session_state.get("single_image_analysis")
    if saved_analysis and saved_analysis.get("key") == analysis_key:
        prediction = saved_analysis["prediction"]
        assessment = saved_analysis["assessment"]
        heatmap = saved_analysis["heatmap"]

        st.divider()
        st.write("### Analysis Results")
        display_prediction_result(
            prediction.prediction,
            prediction.confidence,
            confidence_threshold,
            assessment,
        )

        result_col, heatmap_col = st.columns([0.95, 1.05])
        with result_col:
            st.plotly_chart(
                create_confidence_chart(
                    prediction.top3_classes,
                    prediction.top3_confidences,
                )
            )
            display_disease_info(prediction.prediction)

        with heatmap_col:
            compare_left, compare_right = st.columns(2)
            with compare_left:
                st.image(image, caption="Original image", width="stretch")
            with compare_right:
                st.image(
                    heatmap,
                    caption="Grad-CAM focus overlay",
                    width="stretch",
                )
            st.caption(
                "Warmer colors show regions that contributed more strongly to the CNN prediction."
            )

        display_ai_consultation(prediction, assessment)


def render_batch_tab(
    model,
    transform,
    device,
    encoder,
    confidence_threshold,
    margin_threshold,
):
    st.header("Batch Analysis")
    st.info("Upload multiple images for batch processing")

    uploaded_files = st.file_uploader(
        "Choose multiple plant images...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis",
    )

    if not uploaded_files or not st.button("Analyze All Images", type="primary"):
        return

    results_data = []
    ensure_prediction_history(st.session_state)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")

        try:
            image = Image.open(file)
            prediction = predict_image(image, model, transform, device, encoder)
            assessment = assess_prediction(
                prediction.prediction,
                prediction.confidence,
                prediction.top3_confidences,
                confidence_threshold,
                margin_threshold,
            )

            results_data.append(
                {
                    "Filename": file.name,
                    "Prediction": format_label(prediction.prediction),
                    "Confidence": f"{prediction.confidence:.2%}",
                    "Status": assessment.status,
                    "Reliable": "Yes"
                    if not assessment.is_uncertain
                    else "No",
                }
            )
            add_prediction_history(
                st.session_state,
                file.name,
                prediction.prediction,
                prediction.confidence,
                "Batch Analysis",
                status=assessment.status,
            )
        except Exception:
            results_data.append(
                {
                    "Filename": file.name,
                    "Prediction": "Error",
                    "Confidence": "N/A",
                    "Status": "Error",
                    "Reliable": "No",
                }
            )
            st.session_state.prediction_history.append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": file.name,
                    "prediction": "Error",
                    "confidence": 0.0,
                    "status": "Error",
                    "analysis_type": "Batch Analysis (Error)",
                }
            )

        progress_bar.progress((i + 1) / len(uploaded_files))

    status_text.text("Analysis complete!")
    df = pd.DataFrame(results_data)
    st.dataframe(df, width="stretch")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Healthy Plants", sum(1 for row in results_data if "healthy" in row["Status"].lower()))
    with col2:
        st.metric("Diseased Plants", sum(1 for row in results_data if "disease" in row["Status"].lower()))
    with col3:
        st.metric("Reliable Predictions", sum(1 for row in results_data if row["Reliable"] == "Yes"))
    with col4:
        st.metric("Total Images", len(results_data))

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"plant_disease_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_history_tab():
    st.header("Results History")

    if "prediction_history" not in st.session_state or not st.session_state.prediction_history:
        st.info("No prediction history available. Start analyzing images to see results here.")
        return

    history = st.session_state.prediction_history
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_types = ["All"] + list(
            set(item.get("analysis_type", "Unknown") for item in history)
        )
        selected_type = st.selectbox("Filter by Analysis Type:", analysis_types)

    with col2:
        statuses = ["All"] + list(set(item["status"] for item in history))
        selected_status = st.selectbox("Filter by Status:", statuses)

    with col3:
        st.metric("Total Predictions", len(history))

    filtered_history = filter_history(history, selected_type, selected_status)
    if not filtered_history:
        st.info("No results match the selected filters.")
        return

    history_df = history_dataframe(filtered_history)
    st.dataframe(history_df, width="stretch", hide_index=True)

    st.subheader("Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Healthy Plants", sum(1 for item in filtered_history if "healthy" in item["status"].lower()))
    with col2:
        st.metric("Diseased Plants", sum(1 for item in filtered_history if "disease" in item["status"].lower()))
    with col3:
        st.metric("Single Analysis", sum(1 for item in filtered_history if item.get("analysis_type", "").startswith("Single")))
    with col4:
        st.metric("Batch Analysis", sum(1 for item in filtered_history if item.get("analysis_type", "").startswith("Batch")))

    if len(filtered_history) > 1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_status_distribution_chart(filtered_history)
            )
        with col2:
            st.plotly_chart(
                create_analysis_type_chart(filtered_history)
            )

        confidence_fig = create_confidence_distribution_chart(filtered_history)
        if confidence_fig:
            st.plotly_chart(confidence_fig)

    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered History as CSV",
        data=csv,
        file_name=f"plant_disease_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.markdown("---")
    _, clear_col = st.columns([3, 1])
    with clear_col:
        if st.button("Clear All History", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()


def main():
    st.title("Leaf Disease Detection System")
    st.markdown("### ML-Powered Plant Health Analysis")
    st.markdown("Upload images of plant leaves to detect diseases using the trained CNN model.")

    try:
        model, transform, device, encoder, classes = load_model()
        st.success("Model loaded successfully.")
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return

    confidence_threshold, margin_threshold = render_sidebar(classes, device)
    tab1, tab2, tab3 = st.tabs(
        ["Single Image Analysis", "Batch Analysis", "Results History"]
    )

    with tab1:
        render_single_image_tab(
            model,
            transform,
            device,
            encoder,
            confidence_threshold,
            margin_threshold,
        )
    with tab2:
        render_batch_tab(
            model,
            transform,
            device,
            encoder,
            confidence_threshold,
            margin_threshold,
        )
    with tab3:
        render_history_tab()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Plant Disease Detection System | Built with Streamlit & PyTorch</p>
            <p>Model trained on PlantVillage dataset with 98%+ accuracy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
