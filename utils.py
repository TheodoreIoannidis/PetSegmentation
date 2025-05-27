import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from supervised import UNet, Segformer, Inception
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision import transforms
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def postprocess(masks, mode="open", kernel_size=5, iters=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if mode == "open":
        new_masks = [cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iters) for mask in masks]
    elif mode == "close":
        new_masks = [cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iters) for mask in masks]
    elif mode == "erosion":
        new_masks = [cv2.erode(mask.astype(np.uint8), kernel, iterations=iters) for mask in masks]
    elif mode == "dilation":
        new_masks = [cv2.dilate(mask.astype(np.uint8), kernel, iterations=iters) for mask in masks]
    else:
        new_masks = masks
    return new_masks

def fix_labels(pred_masks, gt_masks, lesion_positive=True):
    """
    Flip predicted masks if needed based on GT, and ensure lesion is 1.
    If lesion_positive=True, final output has lesion as 1.
    """
    fixed_preds = []

    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred.astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        # Flatten for metric comparison
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        # Try both label assignments
        iou_0 = jaccard_score(gt_flat, (pred_flat == 0))
        iou_1 = jaccard_score(gt_flat, (pred_flat == 1))

        # Flip if label 0 gives better IoU
        if iou_0 > iou_1:
            pred = 1 - pred

        # Optional: ensure lesion is positive (class 1)
        if lesion_positive:
            # If GT has more lesion pixels than background, make sure pred does too
            gt_lesion_ratio = np.sum(gt) / gt.size
            pred_lesion_ratio = np.sum(pred) / pred.size

            if pred_lesion_ratio < 0.5 and gt_lesion_ratio > 0.5:
                pred = 1 - pred

        fixed_preds.append(pred)

    return fixed_preds

def evaluate_masks(pred_masks, gt_masks):
    """
    Evaluate predicted masks.
    Returns mean metrics (accuracy, iou, f1).
    """
    acc_list = []
    iou_list = []
    f1_list = []
    cm = np.zeros((2, 2), dtype=int)
    for pred, gt in zip(pred_masks, gt_masks):
        pred_flat = pred.flatten()
        gt_flat = (gt.flatten() > 0).astype(np.uint8)

        acc0 = accuracy_score(gt_flat, (pred_flat == 0))
        acc1 = accuracy_score(gt_flat, (pred_flat == 1))

        acc = accuracy_score(gt_flat, pred_flat)
        iou = jaccard_score(gt_flat, pred_flat)
        f1 = f1_score(gt_flat, pred_flat)

        acc_list.append(acc)
        iou_list.append(iou)
        f1_list.append(f1)
        cm += confusion_matrix(gt_flat, pred_flat, labels=[0, 1])

    mean_acc = np.mean(acc_list)
    mean_iou = np.mean(iou_list)
    mean_f1 = np.mean(f1_list)

    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Mean IoU (Jaccard): {mean_iou:.4f}")
    print(f"Mean F1 Score (Dice): {mean_f1:.4f}")

    disp = ConfusionMatrixDisplay(cm, display_labels=["Background", "Lesion"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Aggregated)")
    plt.show()

    # Plot histograms
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(acc_list, bins=10, color='r', alpha=0.6, edgecolor='black')
    plt.title("Accuracy Distribution")

    plt.subplot(1, 3, 2)
    plt.hist(iou_list, bins=10, color='g', alpha=0.6, edgecolor='black')
    plt.title("IoU Distribution")

    plt.subplot(1, 3, 3)
    plt.hist(f1_list, bins=10, color='skyblue', alpha=0.6, edgecolor='black')
    plt.title("F1 Score Distribution")

    plt.tight_layout()
    plt.show()

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlay a binary mask on top of an image.
    - image: (H, W, 3) numpy array, RGB
    - mask: (H, W) numpy array, 0/1 values or 0/255
    - color: RGB tuple for mask color
    - alpha: transparency factor (0=transparent, 1=opaque)
    """
    image = image.copy()

    # Make sure mask is binary 0 or 1
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = color[0]
    colored_mask[:, :, 1] = color[1]
    colored_mask[:, :, 2] = color[2]

    # Apply mask
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    overlay = np.where(mask_3d, (1 - alpha) * image + alpha * colored_mask, image)

    return overlay.astype(np.uint8)


def visualize_overlay(image, gt_mask, pred_mask, post_mask=None, alpha=0.5):
    """
    Plot original image + overlay GT mask and Predicted mask.
    """
    plt.figure(figsize=(18, 6))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Ground Truth Overlay
    overlay_gt = overlay_mask(image, gt_mask, color=(0, 255, 0), alpha=alpha)
    plt.subplot(1, 3, 2)
    plt.imshow(overlay_gt)
    plt.title("Ground Truth Overlay (Green)")
    plt.axis("off")

    # Predicted Overlay
    overlay_pred = overlay_mask(image, pred_mask, color=(255, 0, 0), alpha=alpha)
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_pred)
    plt.title("Prediction Overlay (Red)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def predict_and_visualize_single(model, image_path, postprocess_mode='none', alpha=0.5, device='cpu'):
    image = Image.fromarray(image_path).convert('RGB')
    original_np = np.array(image.resize((128, 128)))

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    if isinstance(model, (UNet, Segformer, Inception)):
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, dict):
                output = output.get("logits") or output.get("out")
            pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    elif isinstance(model, (KMeans, GaussianMixture)):
        model.fit(original_np.reshape(-1, 3))
        pred_mask = model.predict(original_np.reshape(-1, 3)).reshape(128, 128)

    if postprocess_mode != 'none':
        pred_mask = postprocess([pred_mask], mode=postprocess_mode)[0]

    bw_mask = (pred_mask * 255).astype(np.uint8)
    overlay = overlay_mask(original_np, pred_mask, color=(255, 0, 0), alpha=alpha)
    # Resize outputs to 384x384
    bw_mask = cv2.resize(pred_mask.astype(np.uint8) * 255, (256, 256), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.resize(overlay_mask(original_np, pred_mask, color=(255, 0, 0), alpha=alpha),
        (256, 256),
        interpolation=cv2.INTER_LINEAR
    )

    return bw_mask, overlay