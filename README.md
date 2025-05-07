# ISIC 2018 Skin Lesion Segmentation

This project explores unsupervised and supervised image segmentation methods applied to the **ISIC 2018 skin lesion dataset**. It compares simple segmentation techniques like **KMeans** and **Gaussian Mixture Models (GMM)** against deep learning models (Unet and Inception-based architecture). The deep models are trained on ISIC data, evaluated on the test set and its performance is compared with the baseline models.

## Goals

- Segment skin lesions from dermoscopic images.
- Compare baseline unsupervised methods (KMeans, GMM) with Unet and a custom Inception-based network.
- Evaluate masks using standard metrics: **IoU**, **Dice**, **Accuracy**.
- Visualize results with overlays (predictions vs. ground truth).
- Explore which morphological operations can improve the quality of the segmentation.
    (erosion, dilation, opening, closing)

## Dataset

- **ISIC 2018 Challenge - Task 1**
- ~2,600 dermoscopic images and ground truth lesion masks.
- Downloaded from [ISIC Archive](https://challenge.isic-archive.com/data/#2018).
The images and masks are stored in different folders.
