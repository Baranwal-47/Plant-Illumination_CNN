import streamlit as st
import torch
from PIL import Image
import pickle
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64
from plant_disease_model import PlantDiseaseModel

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disease information database
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is a common disease affecting pepper plants.",
        "symptoms": "Dark brown spots with yellow halos on leaves, fruit lesions",
        "treatment": "Apply copper-based bactericides, improve air circulation",
        "prevention": "Use disease-free seeds, avoid overhead watering, crop rotation"
    },
    "Pepper__bell___healthy": {
        "description": "Healthy pepper plant with no visible disease symptoms.",
        "symptoms": "Green, vibrant leaves without spots or discoloration",
        "treatment": "No treatment needed - maintain good care",
        "prevention": "Continue proper watering, fertilization, and monitoring"
    },
    "Potato___Early_blight": {
        "description": "Early blight is a fungal disease affecting potato plants.",
        "symptoms": "Brown spots with concentric rings, yellowing leaves",
        "treatment": "Apply fungicides, remove affected leaves",
        "prevention": "Proper spacing, avoid overhead watering, crop rotation"
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious fungal disease of potatoes.",
        "symptoms": "Water-soaked lesions, white fuzzy growth on leaf undersides",
        "treatment": "Apply preventive fungicides, remove affected plants",
        "prevention": "Use resistant varieties, avoid wet conditions"
    },
    "Potato___healthy": {
        "description": "Healthy potato plant showing normal growth.",
        "symptoms": "Green foliage without lesions or discoloration",
        "treatment": "No treatment needed",
        "prevention": "Maintain proper care and monitoring"
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot affects tomato leaves and fruit.",
        "symptoms": "Small dark spots on leaves, fruit lesions",
        "treatment": "Copper sprays, remove affected parts",
        "prevention": "Use certified seeds, avoid overhead watering"
    },
    "Tomato_Early_blight": {
        "description": "Early blight is a common tomato fungal disease.",
        "symptoms": "Brown spots with target-like rings",
        "treatment": "Fungicide applications, proper plant spacing",
        "prevention": "Mulching, avoiding wet foliage"
    },
    "Tomato_Late_blight": {
        "description": "Late blight can devastate tomato crops quickly.",
        "symptoms": "Water-soaked lesions, rapid plant death",
        "treatment": "Preventive fungicides, immediate removal of affected plants",
        "prevention": "Use resistant varieties, monitor weather conditions"
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold thrives in humid greenhouse conditions.",
        "symptoms": "Yellow spots on upper leaf surface, fuzzy growth below",
        "treatment": "Improve ventilation, apply fungicides",
        "prevention": "Reduce humidity, increase air circulation"
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes gradual defoliation.",
        "symptoms": "Small brown spots with dark borders and light centers",
        "treatment": "Fungicide applications, remove lower leaves",
        "prevention": "Mulching, proper plant spacing"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites are tiny pests that damage leaves.",
        "symptoms": "Stippled leaves, fine webbing, yellowing",
        "treatment": "Miticides, increase humidity, beneficial insects",
        "prevention": "Regular monitoring, avoid water stress"
    },
    "Tomato__Target_Spot": {
        "description": "Target spot creates distinctive ring patterns on leaves.",
        "symptoms": "Brown spots with concentric rings",
        "treatment": "Fungicide applications, improve air circulation",
        "prevention": "Avoid overhead watering, crop rotation"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies.",
        "symptoms": "Upward curling leaves, yellowing, stunted growth",
        "treatment": "No cure - remove infected plants, control whiteflies",
        "prevention": "Use resistant varieties, control whitefly populations"
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Mosaic virus causes distinctive leaf patterns.",
        "symptoms": "Mottled green and yellow leaf patterns",
        "treatment": "No cure - remove infected plants",
        "prevention": "Use virus-free seeds, control aphid vectors"
    },
    "Tomato_healthy": {
        "description": "Healthy tomato plant with normal growth.",
        "symptoms": "Vibrant green leaves, normal growth patterns",
        "treatment": "No treatment needed",
        "prevention": "Continue proper care and monitoring"
    }
}

@st.cache_resource
def load_model():
    """Load the trained model and artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load artifacts
    with open("class_names.json") as f:
        classes = json.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)
    
    # Load model
    model = PlantDiseaseModel(len(classes))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    return model, transform, device, encoder, classes

def predict_image(image, model, transform, device, encoder):
    """Predict disease for a single image."""
    img = image.convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_classes = [encoder.inverse_transform([idx.item()])[0] for idx in top3_idx[0]]
        top3_confidences = [prob.item() for prob in top3_prob[0]]
    
    return encoder.inverse_transform([pred.item()])[0], confidence.item(), top3_classes, top3_confidences

def display_disease_info(disease_name):
    """Display detailed information about the detected disease."""
    if disease_name in DISEASE_INFO:
        info = DISEASE_INFO[disease_name]
        
        with st.expander(f"ℹ️ About {disease_name.replace('_', ' ').title()}", expanded=True):
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

def create_confidence_chart(top3_classes, top3_confidences):
    """Create a confidence visualization chart."""
    df = pd.DataFrame({
        'Disease': [cls.replace('_', ' ').replace('  ', ' - ') for cls in top3_classes],
        'Confidence': [conf * 100 for conf in top3_confidences]
    })
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Disease',
        orientation='h',
        title="Top 3 Predictions",
        color='Confidence',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=300)
    return fig

def main():
    # Title and description
    st.title("🌱 Leaf Disease Detection System")
    st.markdown("### ML-Powered Plant Health Analysis")
    st.markdown("Upload images of plant leaves to detect diseases using our trained CNN model.")
    
    # Load model
    try:
        model, transform, device, encoder, classes = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Model info
        st.info(f"""
        **Model Statistics:**
        - Classes: {len(classes)}
        - Device: {device}
        - Status: Ready
        """)
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.05,
            help="Minimum confidence for reliable predictions"
        )
        
        # Show disease list
        if st.checkbox("Show Available Disease Classes"):
            st.write("**Detectable Diseases:**")
            for i, cls in enumerate(classes, 1):
                st.write(f"{i}. {cls.replace('_', ' ').replace('  ', ' - ')}")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Single Image Analysis", "📊 Batch Analysis", "📈 Results History"])
    
    with tab1:
        st.header("Single Image Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload a clear image of a plant leaf for disease detection"
        )
        
        if uploaded_file:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width='stretch')
                
                # Image info
                st.write(f"**Image Details:**")
                st.write(f"- Size: {image.size}")
                st.write(f"- Mode: {image.mode}")
                st.write(f"- Format: {uploaded_file.type}")
            
            with col2:
                if st.button("🔍 Analyze Disease", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        try:
                            # Make prediction
                            result, confidence, top3_classes, top3_confidences = predict_image(
                                image, model, transform, device, encoder
                            )
                            
                            # Store in session state for history
                            if 'prediction_history' not in st.session_state:
                                st.session_state.prediction_history = []
                            
                            prediction_data = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': uploaded_file.name,
                                'prediction': result,
                                'confidence': confidence,
                                'status': 'Healthy' if 'healthy' in result.lower() else 'Disease Detected',
                                'analysis_type': 'Single Image'
                            }
                            st.session_state.prediction_history.append(prediction_data)
                            
                            # Display results
                            st.write("### 🎯 Analysis Results")
                            
                            # Main prediction
                            if confidence >= confidence_threshold:
                                if 'healthy' in result.lower():
                                    st.success(f"✅ **Healthy Plant Detected**")
                                    st.balloons()
                                else:
                                    st.error(f"⚠️ **Disease Detected: {result.replace('_', ' ').replace('  ', ' - ')}**")
                            else:
                                st.warning(f"⚡ **Low Confidence Prediction: {result.replace('_', ' ').replace('  ', ' - ')}**")
                                st.warning(f"Confidence ({confidence:.2%}) is below threshold ({confidence_threshold:.2%})")
                            
                            # Confidence meter
                            st.metric("Confidence Level", f"{confidence:.2%}")
                            
                            # Confidence bar
                            progress_color = "normal" if confidence >= confidence_threshold else "inverse"
                            st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                            
                            # Top 3 predictions chart
                            st.plotly_chart(
                                create_confidence_chart(top3_classes, top3_confidences),
                                width='stretch'
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {e}")
                
                # Disease information
                if uploaded_file and 'result' in locals():
                    display_disease_info(result)
    
    with tab2:
        st.header("Batch Analysis")
        st.info("Upload multiple images for batch processing")
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose multiple plant images...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True,
            help="Upload multiple images for batch analysis"
        )
        
        if uploaded_files and st.button("🔍 Analyze All Images", type="primary"):
            results_data = []
            
            # Initialize prediction history if it doesn't exist
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                try:
                    image = Image.open(file)
                    result, confidence, _, _ = predict_image(image, model, transform, device, encoder)
                    
                    # Add to results table
                    results_data.append({
                        'Filename': file.name,
                        'Prediction': result.replace('_', ' ').replace('  ', ' - '),
                        'Confidence': f"{confidence:.2%}",
                        'Status': 'Healthy' if 'healthy' in result.lower() else 'Disease Detected',
                        'Reliable': 'Yes' if confidence >= confidence_threshold else 'No'
                    })
                    
                    # Add to prediction history (for Results History tab)
                    prediction_data = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': file.name,
                        'prediction': result,
                        'confidence': confidence,
                        'status': 'Healthy' if 'healthy' in result.lower() else 'Disease Detected',
                        'analysis_type': 'Batch Analysis'  # New field to distinguish batch vs single
                    }
                    st.session_state.prediction_history.append(prediction_data)
                    
                except Exception as e:
                    results_data.append({
                        'Filename': file.name,
                        'Prediction': 'Error',
                        'Confidence': 'N/A',
                        'Status': 'Error',
                        'Reliable': 'No'
                    })
                    
                    # Add error to history as well
                    prediction_data = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': file.name,
                        'prediction': 'Error',
                        'confidence': 0.0,
                        'status': 'Error',
                        'analysis_type': 'Batch Analysis (Error)'
                    }
                    st.session_state.prediction_history.append(prediction_data)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Analysis complete!")
            
            # Display results table
            df = pd.DataFrame(results_data)
            st.dataframe(df, width='stretch')
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                healthy_count = sum(1 for r in results_data if 'healthy' in r['Status'].lower())
                st.metric("Healthy Plants", healthy_count)
            
            with col2:
                diseased_count = sum(1 for r in results_data if 'disease' in r['Status'].lower())
                st.metric("Diseased Plants", diseased_count)
            
            with col3:
                reliable_count = sum(1 for r in results_data if r['Reliable'] == 'Yes')
                st.metric("Reliable Predictions", reliable_count)
            
            with col4:
                total_count = len(results_data)
                st.metric("Total Images", total_count)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"plant_disease_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("Results History")
        
        if 'prediction_history' in st.session_state and st.session_state.prediction_history:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Analysis type filter
                analysis_types = ['All'] + list(set([item.get('analysis_type', 'Unknown') for item in st.session_state.prediction_history]))
                selected_type = st.selectbox("Filter by Analysis Type:", analysis_types)
            
            with col2:
                # Status filter
                statuses = ['All'] + list(set([item['status'] for item in st.session_state.prediction_history]))
                selected_status = st.selectbox("Filter by Status:", statuses)
            
            with col3:
                # Show count
                total_predictions = len(st.session_state.prediction_history)
                st.metric("Total Predictions", total_predictions)
            
            # Filter the data
            filtered_history = st.session_state.prediction_history.copy()
            
            if selected_type != 'All':
                filtered_history = [item for item in filtered_history if item.get('analysis_type', 'Unknown') == selected_type]
            
            if selected_status != 'All':
                filtered_history = [item for item in filtered_history if item['status'] == selected_status]
            
            if filtered_history:
                # Display filtered history table
                history_df = pd.DataFrame(filtered_history)
                
                # Reorder columns for better display
                column_order = ['timestamp', 'filename', 'analysis_type', 'prediction', 'confidence', 'status']
                # Only include columns that exist
                available_columns = [col for col in column_order if col in history_df.columns]
                history_df = history_df[available_columns]
                
                # Format confidence as percentage
                if 'confidence' in history_df.columns:
                    history_df['confidence_display'] = history_df['confidence'].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else str(x))
                    history_df = history_df.drop('confidence', axis=1)
                    history_df = history_df.rename(columns={'confidence_display': 'confidence'})
                
                st.dataframe(history_df, width='stretch', hide_index=True)
                
                # Summary statistics for filtered data
                st.subheader("📊 Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    healthy_count = sum(1 for item in filtered_history if 'healthy' in item['status'].lower())
                    st.metric("Healthy Plants", healthy_count)
                
                with col2:
                    diseased_count = sum(1 for item in filtered_history if 'disease' in item['status'].lower())
                    st.metric("Diseased Plants", diseased_count)
                
                with col3:
                    single_count = sum(1 for item in filtered_history if item.get('analysis_type', '').startswith('Single'))
                    st.metric("Single Analysis", single_count)
                
                with col4:
                    batch_count = sum(1 for item in filtered_history if item.get('analysis_type', '').startswith('Batch'))
                    st.metric("Batch Analysis", batch_count)
                
                # Summary charts
                if len(filtered_history) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Status distribution
                        status_counts = pd.Series([item['status'] for item in filtered_history]).value_counts()
                        fig_status = px.pie(
                            values=status_counts.values,
                            names=status_counts.index,
                            title="Health Status Distribution"
                        )
                        st.plotly_chart(fig_status, width='stretch')
                    
                    with col2:
                        # Analysis type distribution
                        type_counts = pd.Series([item.get('analysis_type', 'Unknown') for item in filtered_history]).value_counts()
                        fig_type = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Analysis Type Distribution"
                        )
                        st.plotly_chart(fig_type, width='stretch')
                    
                    # Confidence distribution (only for numerical confidence values)
                    numerical_confidences = [item['confidence'] for item in filtered_history 
                                           if isinstance(item['confidence'], (int, float))]
                    if numerical_confidences:
                        fig_conf = px.histogram(
                            x=numerical_confidences,
                            title="Confidence Score Distribution",
                            nbins=20,
                            labels={'x': 'Confidence Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_conf, width='stretch')
                
                # Download filtered results
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered History as CSV",
                    data=csv,
                    file_name=f"plant_disease_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No results match the selected filters.")
            
            # Clear history button
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Clear All History", type="secondary"):
                    st.session_state.prediction_history = []
                    st.rerun()
        else:
            st.info("No prediction history available. Start analyzing images to see results here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🌱 Plant Disease Detection System | Built with Streamlit & PyTorch</p>
        <p>Model trained on PlantVillage dataset with 98%+ accuracy</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
