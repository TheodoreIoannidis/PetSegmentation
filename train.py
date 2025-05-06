import numpy as np
from dataset import ISICDataset
from inception import InceptionSegment 
from unet import UNet
from utils import lr_scheduler, early_stop
from torch.nn import functional as F
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, jaccard_score, f1_score


# ====== Config ======
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.2
IMG_DIR = './data/images'
MASK_DIR = './data/masks'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Resize((225, 225)),
    transforms.ToTensor()
])

# ====== Dataset ======
dataset = ISICDataset(IMG_DIR, MASK_DIR, transform)
val_len = int(len(dataset) * VAL_SPLIT)
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ====== Model & Optimizer ======
model = InceptionSegment(num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ====== Train Loop ======
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ====== Validation ======
def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)
            loss = loss_fn(output, masks)
            val_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            y_true.append(masks.cpu().numpy().flatten())
            y_pred.append(preds.cpu().numpy().flatten())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return (
        val_loss / len(loader),
        accuracy_score(y_true, y_pred),
        jaccard_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    )

# Training Function
def train(model, train_loader, val_loader, optimizer, epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        optimizer = lr_scheduler(optimizer, epoch=epoch, interval=10)
        # Training Phase
        loop = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{epochs}]", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            loss = F.cross_entropy(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            val_loop = tqdm(
                val_loader, desc=f"Val Epoch [{epoch+1}/{epochs}]", leave=True
            )
            for images, masks in val_loop:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)["out"]
                loss = F.cross_entropy(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)


        if early_stop(avg_val_loss, patience=5):
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | best val loss: {min(val_losses):.6f}"
        )

    return train_losses, val_losses
