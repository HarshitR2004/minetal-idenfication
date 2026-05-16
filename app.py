import io
import os
import numpy as np
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Config (mirrors notebook)
IMG_SIZE = 224
MODEL_PATH = "model.pth"
CLASS_NAMES = [
    "biotite",
    "bornite",
    "chrysocolla",
    "malachite",
    "muscovite",
    "pyrite",
    "quartz",
]


@st.cache_resource
def load_model(path: str, device: torch.device):
    # load state dict first to infer output classes
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        return None, None

    state = torch.load(path, map_location="cpu")

    # build base EfficientNet-B3
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)

    # detect output classes from state_dict classifier weights
    classifier_weight_keys = [k for k in state.keys() if k.startswith("classifier.") and k.endswith(".weight")]
    out_features = None
    if classifier_weight_keys:
        # choose highest index key
        try:
            idxs = [int(k.split('.')[1]) for k in classifier_weight_keys]
            max_idx = max(idxs)
            last_key = f"classifier.{max_idx}.weight"
            out_features = state[last_key].shape[0]
        except Exception:
            # fallback: take first
            out_features = state[classifier_weight_keys[0]].shape[0]

    # fallback default
    if out_features is None:
        out_features = 10

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, out_features)
    )

    # load state dict (allow missing keys if any)
    try:
        model.load_state_dict(state)
    except Exception:
        # try loading strict=False
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    return model, out_features


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_and_cam(model, device, pil_image: Image.Image):
    transform = get_transform()
    img_rgb = pil_image.convert("RGB")
    cam_image = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    input_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        pred_idx = int(pred.item())
        conf_val = float(confidence.item())

    # Grad-CAM
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # Grad-CAM overlay must use the same spatial size as the model input.
    rgb_for_cam = np.array(cam_image).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_for_cam, grayscale_cam, use_rgb=True, image_weight=0.6)

    class_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"

    return pred_idx, class_name, conf_val, visualization


def main():
    st.title("Mineral Classifier")
    st.write("Upload an image and the app will predict the mineral class and show Grad-CAM.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, out_features = load_model(MODEL_PATH, device)
    if model is None:
        st.stop()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    image_data = uploaded.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    st.image(pil_image, caption="Input image", use_column_width=True)

    if st.button("Run inference"):
        with st.spinner("Running model and generating Grad-CAM..."):
            try:
                pred_idx, class_name, conf_val, vis = predict_and_cam(model, device, pil_image)

                st.subheader("Prediction")
                st.write(f"Mineral: {class_name} — Confidence: {conf_val*100:.2f}%")
                st.caption(f"Class index: {pred_idx}")

                st.subheader("Grad-CAM Visualization")
                st.image(vis, use_column_width=True)
            except Exception as e:
                st.error(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
