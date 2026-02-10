#!/usr/bin/env python3
"""Quick test to verify the trained model loads correctly."""

import torch
import json
from plant_disease_model import PlantDiseaseModel

try:
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"✅ Found {len(class_names)} classes: {', '.join(class_names[:5])}...")
    
    # Create and load model
    model = PlantDiseaseModel(len(class_names))
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model has {total_params:,} parameters")
    
    # Test a dummy prediction
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_input)
        predictions = torch.nn.functional.softmax(outputs, dim=1)
        top_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][top_class].item()
    
    print(f"✅ Model inference test passed!")
    print(f"   Sample prediction: {class_names[top_class]} ({confidence:.2%} confidence)")
    print("\n🎉 Your model is ready for inference!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
