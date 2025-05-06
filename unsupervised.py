import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils import *

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from utils import predict_dataset, fix_labels, postprocess, evaluate_masks, visualize_overlay

def run_unsupervised(model_name='kmeans', image_dir='./data/images/', mask_dir='./data/masks/', num_samples=10, visualize=True):
    """
    Run unsupervised segmentation using KMeans or GMM.
    """

    if model_name == 'kmeans':
        model = KMeans(n_clusters=2, random_state=0)
    elif model_name == 'gmm':
        model = GaussianMixture(n_components=2, random_state=0)

    print(f"\nRunning {model_name.upper()} segmentation...")
    pred_masks, gt_masks, images = predict_dataset(model, image_dir, mask_dir, num_samples=num_samples)
    pred_masks = fix_labels(pred_masks, gt_masks)

    evaluate_masks(pred_masks, gt_masks)

    for mode in [None, 'close', 'erosion', 'dilation']:
        post_masks = postprocess(pred_masks, mode=mode, iters=3)
        print(f"\n Postprocessed ({mode}) Results:")
        evaluate_masks(post_masks, gt_masks)

        if visualize:
            for image, gt, pred in zip(images[:2], gt_masks[:2], post_masks[:2]):
                visualize_overlay(image, gt, pred, alpha=0.5)
                plt.close()

# Optional CLI entry
if __name__ == "__main__":
    run_unsupervised(model_name='kmeans')
