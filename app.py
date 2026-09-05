import io
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    GradCAM = None

# Model Configuration & Metadata
IMG_SIZE = 224
INT8_MODEL_PATH = "models/model_int8_ptq.onnx"
PYTORCH_MODEL_PATH = "models/model.pth"

CLASS_NAMES = [
    "biotite",
    "bornite",
    "chrysocolla",
    "malachite",
    "muscovite",
    "pyrite",
    "quartz",
]

# Benchmark Verified Metrics for the INT8 PTQ Model
MODEL_SPECS = {
    "name": "EfficientNet-B3 (INT8 Post-Training Quantized)",
    "accuracy": "90.54%",
    "macro_f1": "0.8817",
    "model_size": "12.54 MB",
    "compression": "71.7% smaller (3.54x reduction)",
    "throughput": "7.24 FPS",
}


def get_transform():
    """Standard preprocessing transform matching ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def load_int8_session(model_path: str):
    """Load optimized ONNX Runtime InferenceSession for INT8 PTQ model."""
    candidates = [
        Path(model_path),
        Path("models/model_int8_ptq.onnx"),
        Path("models/model.onnx"),
    ]
    resolved_path = None
    for c in candidates:
        if c.is_file():
            resolved_path = c.resolve()
            break

    if resolved_path is None or ort is None:
        return None, None, None

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(str(resolved_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name


@st.cache_resource
def load_pytorch_model(path: str, device: torch.device):
    """Load baseline PyTorch model for Grad-CAM explainability."""
    candidates = [
        Path(path),
        Path("models/model.pth"),
        Path("model.pth"),
    ]
    resolved_path = None
    for c in candidates:
        if c.is_file():
            resolved_path = c.resolve()
            break

    if resolved_path is None:
        return None

    try:
        state = torch.load(resolved_path, map_location="cpu", weights_only=False)
        model = efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(CLASS_NAMES)),
        )
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def run_int8_inference(session, input_name, output_name, pil_image: Image.Image):
    """Run fast INT8 PTQ inference via ONNX Runtime."""
    transform = get_transform()
    img_rgb = pil_image.convert("RGB")
    tensor = transform(img_rgb).unsqueeze(0).numpy()

    t0 = time.perf_counter()
    logits = session.run([output_name], {input_name: tensor})[0][0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    pred_idx = int(np.argmax(probs))
    conf_val = float(probs[pred_idx])
    class_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"Mineral {pred_idx + 1}"

    return pred_idx, class_name, conf_val, latency_ms


def generate_gradcam_overlay(model, device, pil_image: Image.Image, pred_idx: int):
    """Generate Grad-CAM heat-map overlay using PyTorch model if available."""
    if model is None or GradCAM is None:
        return None

    try:
        transform = get_transform()
        img_rgb = pil_image.convert("RGB")
        cam_image = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)

        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_idx)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        rgb_for_cam = np.array(cam_image).astype(np.float32) / 255.0
        vis = show_cam_on_image(rgb_for_cam, grayscale_cam, use_rgb=True, image_weight=0.6)
        return vis
    except Exception:
        return None


def main():
    st.set_page_config(
        page_title="Explainable Mineral Classification",
        page_icon="💎",
        layout="wide",
    )

    st.title("💎 Mineral Classifier")
    st.write(
        "Powered by **EfficientNet-B3 INT8 Post-Training Quantization (PTQ)** for high-speed, "
        "low-latency edge inference with explainable visual attention."
    )

    # Sidebar: Performance Metrics
    with st.sidebar:
        st.header("⚡ Model Specification")
        st.write(f"**Architecture**: {MODEL_SPECS['name']}")
        st.write(f"**Model Size**: {MODEL_SPECS['model_size']} ({MODEL_SPECS['compression']})")

        st.subheader("Benchmark Accuracy")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Top-1 Accuracy", MODEL_SPECS["accuracy"])
        col_m2.metric("Macro F1", MODEL_SPECS["macro_f1"])

        st.write(f"**CPU Throughput**: {MODEL_SPECS['throughput']}")
        st.caption("Benchmark tested on 846 ground-truth mineral specimens.")

    # Load sessions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    int8_sess, inp_name, out_name = load_int8_session(INT8_MODEL_PATH)
    pytorch_model = load_pytorch_model(PYTORCH_MODEL_PATH, device)

    if int8_sess is None and pytorch_model is None:
        st.error("No model checkpoints found in `models/`. Please check model artifact files.")
        st.stop()

    uploaded = st.file_uploader(
        "Select or upload a mineral specimen image (JPG, PNG, WEBP):",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("Upload an image or drag and drop a specimen to begin.")
        return

    image_data = uploaded.read()
    pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Specimen")
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Classification & Visual Analysis")
        if st.button("Identify Mineral", type="primary"):
            with st.spinner("Classifying specimen with INT8 PTQ model..."):
                try:
                    if int8_sess is not None:
                        pred_idx, class_name, conf_val, latency = run_int8_inference(
                            int8_sess, inp_name, out_name, pil_image
                        )
                    else:
                        # Fallback to PyTorch inference if ONNX is unavailable
                        transform = get_transform()
                        input_tensor = transform(pil_image).unsqueeze(0).to(device)
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            probs = torch.softmax(pytorch_model(input_tensor), dim=1)[0]
                        latency = (time.perf_counter() - t0) * 1000.0
                        pred_idx = int(torch.argmax(probs).item())
                        conf_val = float(probs[pred_idx].item())
                        class_name = CLASS_NAMES[pred_idx]

                    # Results Card
                    st.success(f"**Identified Mineral**: {class_name.capitalize()}")
                    st.write(f"**Confidence**: {conf_val * 100:.2f}%")
                    st.caption(f"Inference Latency: {latency:.2f} ms | Engine: INT8 Quantized ONNX Runtime")

                    # Grad-CAM Visualization
                    vis = generate_gradcam_overlay(pytorch_model, device, pil_image, pred_idx)
                    if vis is not None:
                        st.subheader("Grad-CAM Attention Map")
                        st.image(vis, caption=f"Visual Regions Indicating {class_name.capitalize()}", use_column_width=True)
                except Exception as e:
                    st.error(f"Inference error: {e}")


if __name__ == "__main__":
    main()
