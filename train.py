import numpy as np
import torch
from utils import *
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import ISICDataset
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def lr_scheduler(optimizer, epoch=0, start=0, interval=5):
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

    train_dataset = ISICDataset(img_dir, mask_dir, transform, file_list='./splits/train.txt')
    val_dataset = ISICDataset(img_dir, mask_dir, transform, file_list='./splits/val.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ====== Model & Optimizer ======
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        optimizer = lr_scheduler(optimizer, epoch=epoch, interval=2)

        loop = tqdm(train_loader, desc=f"Train Epoch [{epoch+1}/{epochs}]", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get("logits") or outputs.get("out")

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
                    outputs = outputs.get("logits") or outputs.get("out")
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
            torch.save(model.state_dict(), f'./{model.__class__.__name__.lower()}.pt')

        # ====== Early stopping ======
        if early_stop(avg_val_loss, val_losses, epoch=epoch, patience=5):
            print(f"Early stopping at epoch {epoch+1}")
            break

def test_model(model, loader):
    """
    Predict masks for test set.
    Returns: pred_masks, gt_masks, original images
    """
    model.eval()
    gt_masks, pred_masks, image_list = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs.get("logits") or outputs.get("out")
            preds = torch.argmax(outputs, dim=1)
            gt_masks.extend(masks.cpu().numpy())
            pred_masks.extend(preds.cpu().numpy())
            image_list.extend((images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8))

    # Compute per-sample metrics
    accs, ious, f1s = [], [], []
    for pred, gt in zip(pred_masks, gt_masks):
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        accs.append(accuracy_score(gt_flat, pred_flat))
        ious.append(jaccard_score(gt_flat, pred_flat))
        f1s.append(f1_score(gt_flat, pred_flat))

    # Plot metric distributions
    metrics = {'Accuracy': accs, 'IoU': ious, 'F1 Score': f1s}
    plt.figure(figsize=(10, 6))
    for name, values in metrics.items():
        plt.plot([name]*len(values), values, 'o', alpha=0.4, label=name)
    plt.title("Per-sample Metric Distributions")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pred_masks, gt_masks, image_list