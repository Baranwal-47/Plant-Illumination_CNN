import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import json
import pickle

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(PlantDiseaseModel, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx])
        except Exception as e:
            print(f"Warning: Could not load image {self.image_paths[idx]}: {e}")
            # Return a blank image if current image is corrupted
            img = Image.new('RGB', (256, 256), color='black')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(self.labels[idx])

def load_images(data_dir):
    image_paths, labels = [], []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    skipped_files = 0
    
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            if img_name.startswith("."):
                continue
            
            # Check if file has valid extension
            file_ext = os.path.splitext(img_name.lower())[1]
            if file_ext not in valid_extensions:
                skipped_files += 1
                print(f"Skipping invalid file: {img_name}")
                continue
                
            img_path = os.path.join(class_path, img_name)
            
            # Try to verify the image can be opened
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify it's a valid image
                image_paths.append(img_path)
                labels.append(class_dir)
            except Exception as e:
                skipped_files += 1
                print(f"Skipping corrupted image: {img_path} - {e}")
    
    print(f"Loaded {len(image_paths)} valid images, skipped {skipped_files} invalid files")
    return image_paths, labels

def prepare_data(data_dir, image_size=(256, 256), batch_size=32, test_size=0.3, valid_ratio=0.5):
    from sklearn.model_selection import train_test_split
    image_paths, labels = load_images(data_dir)
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open("class_names.json", "w") as f:
        json.dump(list(encoder.classes_), f)
    
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels_encoded, test_size=test_size, stratify=labels_encoded)
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=valid_ratio, stratify=temp_labels)
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor()
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    with open("inference_transform.pkl", "wb") as f:
        pickle.dump(valid_transform, f)

    train_ds = PlantDataset(train_paths, train_labels, train_transform)
    valid_ds = PlantDataset(valid_paths, valid_labels, valid_transform)
    test_ds = PlantDataset(test_paths, test_labels, valid_transform)
    
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(valid_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
        len(encoder.classes_)
    )

