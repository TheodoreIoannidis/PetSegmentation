import numpy as np
from dataset import ISICDataset
from utils import *
from torch.nn import functional as F
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_scheduler(optimizer, epoch=0, start=0, interval=10):
    # decreases optimizer's learning rate
    if epoch > start and epoch % interval == 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.5
        print("Learning rate decreased")
    return optimizer

def early_stop(val_loss, val_list, epoch=0, patience=6):
    # early stopping
    if val_loss >= min(val_list) and (epoch - np.argmin(np.array(val_list))) > patience:
        print(f"Training stopped at epoch: {epoch}\n")
        return True
    return False

def train(model, img_dir, mask_dir, epochs=10, batch_size=4):
    # ====== Transforms & Dataset ======
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = ISICDataset(img_dir, mask_dir, transform)
    val_len = int(len(dataset) * 0.2)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers = os.cpu_count() // 2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers = os.cpu_count() // 2, pin_memory=True)

    # ====== Model & Optimizer ======
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        optimizer = lr_scheduler(optimizer, epoch=epoch, interval=10)

        loop = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{epochs}]", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):  # for DeepLabV3
                outputs = outputs["out"]

            loss = F.cross_entropy(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # ====== Validation ======
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]

                loss = F.cross_entropy(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                y_true.append(masks.cpu().numpy().flatten())
                y_pred.append(preds.cpu().numpy().flatten())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        acc = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Acc: {acc:.4f} | IoU: {iou:.4f} | F1: {f1:.4f}")

        # ====== Save best model ======
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'best_model_epoch{epoch+1}.pt')

        # ====== Early stopping ======
        if early_stop(avg_val_loss, val_losses, epoch=epoch, patience=5):
            print(f"Early stopping at epoch {epoch+1}")
            break