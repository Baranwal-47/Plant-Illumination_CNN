# Plant Disease Detection using CNN

An AI-powered plant disease detection system that uses a custom PyTorch CNN to classify plant leaf diseases from images. The Streamlit app supports single-image analysis, batch analysis, prediction history, Grad-CAM explainability, uncertainty handling, and an optional AgentRouter AI second consultation.

## Features

### Machine Learning
- Custom CNN architecture with 1,688,079 trainable parameters.
- 15 supported classes across pepper, potato, and tomato leaves.
- Confidence scoring with adjustable reliability threshold.
- Top-3 prediction chart for uncertainty awareness.
- Grad-CAM heatmap overlay to show which image regions influenced the CNN prediction.

### Streamlit Application
- Single image analysis with prediction, confidence, top-3 classes, disease information, and Grad-CAM comparison.
- Batch image analysis with reliability flags and CSV export.
- Results history with filters, summary metrics, charts, and CSV export.
- `Needs Expert Review` status when confidence is low or the top predictions are too close.
- Optional AI second consultation through AgentRouter for uncertain cases.

### AI Second Consultation
The consultation feature uses an OpenAI-compatible AgentRouter chat completion endpoint.

Default configuration:

```text
Base URL: https://agentrouter.org/v1
Model: claude-haiku-4-5-20251001
```

The app reads the API key from `.env` or environment variables using any of these names:

```text
AGENT_ROUTER_TOKEN
AGENT_ROUTER_API_KEY
ANTHROPIC_API_KEY
```

It also supports a single raw key line in `.env` for convenience. The `.env` file is ignored by Git.

## Supported Disease Classes

- Pepper bell: bacterial spot, healthy
- Potato: early blight, late blight, healthy
- Tomato: bacterial spot, early blight, late blight, leaf mold, Septoria leaf spot, spider mites, target spot, yellow leaf curl virus, mosaic virus, healthy

## Tech Stack

- PyTorch and torchvision for model training and inference
- Streamlit for the web application
- Plotly for charts
- scikit-learn for label encoding and data splitting
- Pandas and NumPy for data handling
- Pillow for image processing

## Project Structure

```text
Plant-Disease-Detection-using-CNN/
|-- app/
|   |-- ai_consultant.py      # AgentRouter second-opinion client
|   |-- charts.py             # Plotly chart helpers
|   |-- disease_info.py       # Disease metadata and label formatting
|   |-- gradcam.py            # Grad-CAM heatmap generation
|   |-- history.py            # Prediction history helpers
|   |-- model_utils.py        # Model loading and prediction helpers
|   `-- uncertainty.py        # Needs Expert Review logic
|-- plant_disease_model.py    # CNN architecture and dataset pipeline
|-- train.py                  # Training script
|-- inference.py              # Command-line prediction script
|-- streamlit_app.py          # Streamlit UI entrypoint
|-- test_model.py             # Quick model loading/inference test
|-- requirements.txt          # Python dependencies
|-- best_model.pth            # Trained model weights
|-- class_names.json          # Class labels
|-- label_encoder.pkl         # Label encoder artifact
`-- inference_transform.pkl   # Inference transform artifact
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

For Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

## Running Inference from the Command Line

```bash
python inference.py path/to/leaf_image.jpg
```

## Training

Place the PlantVillage dataset in a `PlantVillage/` directory using this structure:

```text
PlantVillage/
|-- Pepper__bell___Bacterial_spot/
|-- Pepper__bell___healthy/
|-- Potato___Early_blight/
`-- ...
```

Then run:

```bash
python train.py
```

Training saves:

- `best_model.pth`
- `class_names.json`
- `label_encoder.pkl`
- `inference_transform.pkl`
- `learning_curves.png`

## Verification

Run the model smoke test:

```bash
python test_model.py
```

On Windows, if the console cannot print emoji from the existing script, run:

```powershell
$env:PYTHONIOENCODING='utf-8'
python test_model.py
```

## Notes

- Grad-CAM uses the trained CNN's final convolution block (`conv_block4`) and the predicted class score.
- AI consultation is advisory only. It should not be treated as a definitive agricultural diagnosis.
- Low confidence or close top predictions are flagged as `Needs Expert Review`.
- The app does not retrain the model during inference.
