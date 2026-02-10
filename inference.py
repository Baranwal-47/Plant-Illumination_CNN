import torch
from PIL import Image
import pickle
import json
import sys
import os
from torchvision import transforms
from plant_disease_model import PlantDiseaseModel

def predict_image(image_path, model, transform, device, encoder):
    """Predict the disease class for a given plant image."""
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(img)
        probabilities = torch.nn.functional.softmax(out, dim=1)
        confidence, pred = torch.max(probabilities, 1)
    return encoder.inverse_transform([pred.item()])[0], confidence.item()

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py rust_fungus-min_1024x1024.webp")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    print(f"🔍 Analyzing image: {image_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Load artifacts
    with open("class_names.json") as f:
        classes = json.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("inference_transform.pkl", "rb") as f:
        transform = pickle.load(f)
    
    # Load and prepare model
    model = PlantDiseaseModel(len(classes))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    print("✅ Model loaded successfully!")
    
    # Make prediction
    try:
        result, confidence = predict_image(image_path, model, transform, device, encoder)
        print(f"\n🎯 Prediction Results:")
        print(f"   Disease Class: {result}")
        print(f"   Confidence: {confidence:.2%}")
        
        # Interpret the result
        if "healthy" in result.lower():
            print(f"   Status: ✅ Plant appears healthy!")
        else:
            print(f"   Status: ⚠️  Disease detected - {result}")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
