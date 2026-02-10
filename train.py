import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from plant_disease_model import *

def save_learning_curves(train_losses, valid_losses, valid_accuracies):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(valid_accuracies, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.close()

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device):
    train_losses, val_losses, val_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item()
                _, preds = torch.max(out, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_losses.append(val_loss / len(valid_loader))
        acc = (correct / total) * 100
        val_accs.append(acc)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Val Loss={val_losses[-1]:.4f}, Val Acc={acc:.2f}%")
    save_learning_curves(train_losses, val_losses, val_accs)
    torch.save(model.state_dict(), "best_model.pth")

def main():
    train_loader, valid_loader, test_loader, num_classes = prepare_data("PlantVillage")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, 40, device)

if __name__ == "__main__":
    main()

