"""Disease metadata and label formatting helpers."""

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is a common disease affecting pepper plants.",
        "symptoms": "Dark brown spots with yellow halos on leaves, fruit lesions",
        "treatment": "Apply copper-based bactericides, improve air circulation",
        "prevention": "Use disease-free seeds, avoid overhead watering, crop rotation",
    },
    "Pepper__bell___healthy": {
        "description": "Healthy pepper plant with no visible disease symptoms.",
        "symptoms": "Green, vibrant leaves without spots or discoloration",
        "treatment": "No treatment needed - maintain good care",
        "prevention": "Continue proper watering, fertilization, and monitoring",
    },
    "Potato___Early_blight": {
        "description": "Early blight is a fungal disease affecting potato plants.",
        "symptoms": "Brown spots with concentric rings, yellowing leaves",
        "treatment": "Apply fungicides, remove affected leaves",
        "prevention": "Proper spacing, avoid overhead watering, crop rotation",
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious fungal disease of potatoes.",
        "symptoms": "Water-soaked lesions, white fuzzy growth on leaf undersides",
        "treatment": "Apply preventive fungicides, remove affected plants",
        "prevention": "Use resistant varieties, avoid wet conditions",
    },
    "Potato___healthy": {
        "description": "Healthy potato plant showing normal growth.",
        "symptoms": "Green foliage without lesions or discoloration",
        "treatment": "No treatment needed",
        "prevention": "Maintain proper care and monitoring",
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot affects tomato leaves and fruit.",
        "symptoms": "Small dark spots on leaves, fruit lesions",
        "treatment": "Copper sprays, remove affected parts",
        "prevention": "Use certified seeds, avoid overhead watering",
    },
    "Tomato_Early_blight": {
        "description": "Early blight is a common tomato fungal disease.",
        "symptoms": "Brown spots with target-like rings",
        "treatment": "Fungicide applications, proper plant spacing",
        "prevention": "Mulching, avoiding wet foliage",
    },
    "Tomato_Late_blight": {
        "description": "Late blight can devastate tomato crops quickly.",
        "symptoms": "Water-soaked lesions, rapid plant death",
        "treatment": "Preventive fungicides, immediate removal of affected plants",
        "prevention": "Use resistant varieties, monitor weather conditions",
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold thrives in humid greenhouse conditions.",
        "symptoms": "Yellow spots on upper leaf surface, fuzzy growth below",
        "treatment": "Improve ventilation, apply fungicides",
        "prevention": "Reduce humidity, increase air circulation",
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes gradual defoliation.",
        "symptoms": "Small brown spots with dark borders and light centers",
        "treatment": "Fungicide applications, remove lower leaves",
        "prevention": "Mulching, proper plant spacing",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites are tiny pests that damage leaves.",
        "symptoms": "Stippled leaves, fine webbing, yellowing",
        "treatment": "Miticides, increase humidity, beneficial insects",
        "prevention": "Regular monitoring, avoid water stress",
    },
    "Tomato__Target_Spot": {
        "description": "Target spot creates distinctive ring patterns on leaves.",
        "symptoms": "Brown spots with concentric rings",
        "treatment": "Fungicide applications, improve air circulation",
        "prevention": "Avoid overhead watering, crop rotation",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease transmitted by whiteflies.",
        "symptoms": "Upward curling leaves, yellowing, stunted growth",
        "treatment": "No cure - remove infected plants, control whiteflies",
        "prevention": "Use resistant varieties, control whitefly populations",
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Mosaic virus causes distinctive leaf patterns.",
        "symptoms": "Mottled green and yellow leaf patterns",
        "treatment": "No cure - remove infected plants",
        "prevention": "Use virus-free seeds, control aphid vectors",
    },
    "Tomato_healthy": {
        "description": "Healthy tomato plant with normal growth.",
        "symptoms": "Vibrant green leaves, normal growth patterns",
        "treatment": "No treatment needed",
        "prevention": "Continue proper care and monitoring",
    },
}


def format_label(class_name):
    """Turn a training class name into readable app text."""
    return class_name.replace("_", " ").replace("  ", " - ").strip()


def get_disease_info(class_name):
    """Return disease metadata for a class name, if available."""
    return DISEASE_INFO.get(class_name)
