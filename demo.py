import gradio as gr
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils import *
from supervised import *
# from unsupervised import run_single_unsupervised

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== Load Models =====
models = {
    "unet": UNet(num_classes=2).to(device),
    "segformer": Segformer(num_classes=2).to(device),
    "inception": Inception(num_classes=2).to(device) 
}
models["unet"].load_state_dict(torch.load("unet.pt", map_location=device))
models["segformer"].load_state_dict(torch.load("segformer.pt", map_location=device))
models["inception"].load_state_dict(torch.load("inception.pt", map_location=device))

for model in models.values():
    model.eval()

# ===== Gradio UI =====
def inference(image, model_name, postprocess_mode):
    model = models[model_name]
    bw_mask, overlay = predict_and_visualize_single(model, image, postprocess_mode=postprocess_mode)
    return overlay, bw_mask

demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(type='numpy', label="Upload Image"),
        gr.Radio(choices=["unet", "segformer", "inception"], label="Model"),
        gr.Radio(choices=["none", "open", "close", "erosion", "dilation"], label="Postprocessing", value="none")
    ],
    outputs=[
        gr.Image(type='numpy', label="Overlay"),
        gr.Image(type='numpy', label="Predicted Mask"),
    ],
    title="Skin Lesion Segmentation Demo",
    description="Upload a skin image, choose model and optional postprocessing to visualize the predicted segmentation mask."
)

demo.launch(share=True)
