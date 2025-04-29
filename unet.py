import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, jaccard_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DeepLabv3 Pretrained
model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True)
model.classifier[-1] = torch.nn.Conv2d(
    256, 2, kernel_size=1
)  # Binary segmentation: background vs lesion
model = model.to(device)

# Define Transformation
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize all images and masks to 256x256
        transforms.ToTensor(),
    ]
)


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = []
        for img in os.listdir(image_dir):
            if img.endswith(".jpg"):
                mask_name = img.replace(".jpg", "_segmentation.png")
                if os.path.exists(os.path.join(mask_dir, mask_name)):
                    self.images.append(img)

        print(f"Loaded {len(self.images)} images with masks.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(
            self.mask_dir, self.images[idx].replace(".jpg", "_segmentation.png")
        )

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).long().squeeze(0)  # Normalize to 0/1, shape (H,W)

        return image, mask


# Paths (change to your correct folders)
image_dir = "./data/images/"  # path where ISIC images are
mask_dir = "./data/masks/"  # path where ISIC masks are

full_dataset = ISICDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

dataset_size = len(full_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training Function
def train(model, train_loader, val_loader, optimizer, epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

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

        # 📋 Print epoch summary
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses


# Evaluation Function
def evaluate(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1)

            y_true.append(masks.cpu().numpy().flatten())
            y_pred.append(preds.cpu().numpy().flatten())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"IoU (Jaccard): {jaccard_score(y_true, y_pred):.4f}")
    print(f"F1 Score (Dice): {f1_score(y_true, y_pred):.4f}")


# Train the model
train(model, train_loader, val_loader, optimizer, epochs=1)

# Final Evaluation on Test set
print("\nTest Metrics:")
evaluate(model, test_loader)
