import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils import *

def predict_dataset(model, image_dir, mask_dir, file_list=None):
    """
    Predict masks over dataset.
    Returns: predicted masks list, ground truth masks list.
    """
    
    if file_list:
        with open(file_list, "r") as f:
            images = [line.strip() for line in f.readlines()]
    else:   
        images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    pred_masks = []
    gt_masks = []
    img_np_list = []

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(
            mask_dir, img_name.replace(".jpg", "_segmentation.png")
        )

        image_np = np.array(Image.open(img_path).convert("RGB").resize((128, 128)))
        mask_np = np.array(Image.open(mask_path).convert("L").resize((128, 128)))
        mask_np = (mask_np > 0).astype(np.uint8)

        h, w, _ = image_np.shape
        pixels = image_np.reshape((-1, 3)) / 255
        model.fit(pixels)

        if isinstance(model, KMeans):
            segmented = model.labels_.reshape((h, w))
        else:
            segmented = model.predict(pixels).reshape((h, w))

        pred_masks.append(segmented.astype(np.uint8))
        gt_masks.append(mask_np)
        img_np_list.append(image_np)

    return pred_masks, gt_masks, img_np_list

def run_unsupervised(model_name='kmeans', image_dir='./data/images/', mask_dir='./data/masks/', file_list='./splits/test.txt', post='none', visualize=True):
    """
    Run unsupervised segmentation using KMeans or GMM on test set.
    """

    if model_name == 'kmeans':
        model = KMeans(n_clusters=2, random_state=0)
    elif model_name == 'gmm':
        model = GaussianMixture(n_components=2, random_state=0)
    else:
        raise ValueError("Model name must be 'kmeans' or 'gmm'.")

    print(f"\nRunning {model_name.upper()} segmentation...")
    pred_masks, gt_masks, images = predict_dataset(model, image_dir, mask_dir, file_list=file_list)
    pred_masks = fix_labels(pred_masks, gt_masks)

    return pred_masks, gt_masks, images