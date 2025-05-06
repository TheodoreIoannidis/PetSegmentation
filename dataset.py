import os
from torch.utils.data import Dataset
from PIL import Image

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
        mask = (mask > 0).long().squeeze(0)  # normalize to 0/1, shape (H,W)

        return image, mask