import numpy as np
from dataset import ISICDataset
from utils import *
from torch.nn import functional as F
from tqdm import tqdm
import torch
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
            torch.save(model.state_dict(), f'./{model.__class__.__name__.lower()}_model.pt')

        # ====== Early stopping ======
        if early_stop(avg_val_loss, val_losses, epoch=epoch, patience=5):
            print(f"Early stopping at epoch {epoch+1}")
            break


def predict_and_visualize_single(model, image_path, mask_path=None, alpha=0.5):
    """
    Predicts a segmentation mask for a single image and visualizes the result.
    """

    # === Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_np = np.array(image.resize((128,128)))  # for visualization
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

    # === Run model prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):  # e.g., DeepLabV3
            output = output["out"]
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # shape: [H, W]

    # === Load GT mask if provided
    gt_mask = None
    if mask_path:
        gt = Image.open(mask_path).convert('L').resize((128, 128))
        gt_mask = (np.array(gt) > 0).astype(np.uint8)

    # === Visualization
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_np)
    plt.title("Original Image")
    plt.axis("off")

    # Ground Truth
    if gt_mask is not None:
        gt_overlay = overlay_mask(original_np, gt_mask, color=(0, 255, 0), alpha=alpha)
        plt.subplot(1, 3, 2)
        plt.imshow(gt_overlay)
        plt.title("Ground Truth Overlay (Green)")
        plt.axis("off")
    else:
        plt.subplot(1, 3, 2)
        plt.imshow(original_np)
        plt.title("No Ground Truth Provided")
        plt.axis("off")

    # Prediction
    pred_overlay = overlay_mask(original_np, pred_mask, color=(255, 0, 0), alpha=alpha)
    plt.subplot(1, 3, 3)
    plt.imshow(pred_overlay)
    plt.title("Prediction Overlay (Red)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def test_model(model, loader):
    """
    Predict masks for test set.
    Returns: pred_masks, gt_masks, original images
    """
    model.eval()
    y_true, y_pred = [], []
    gt_masks, pred_masks, images = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]
            preds = torch.argmax(outputs, dim=1)

            y_true.append(masks.cpu().numpy().flatten())
            y_pred.append(preds.cpu().numpy().flatten())

            gt_masks.extend(masks.cpu().numpy())
            pred_masks.extend(preds.cpu().numpy())
            images.extend(images.cpu().permute(0, 2, 3, 1).numpy() * 255)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\nTest Set Metrics:")
    print(f" - Accuracy: {acc:.4f}")
    print(f" - IoU (Jaccard): {iou:.4f}")
    print(f" - F1 Score (Dice): {f1:.4f}")

    return pred_masks, gt_masks, images
