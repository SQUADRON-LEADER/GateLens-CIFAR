from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from model import BaselineMLP, PrunableMLP


PROJECT_NAME = "GateLens CIFAR"
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@st.cache_data
def list_checkpoints(output_dir: str) -> List[Path]:
    base = Path(output_dir)
    if not base.exists():
        return []
    return sorted(base.glob("*.pt"))


@st.cache_resource
def load_model(checkpoint_path: str, hidden_dims: Tuple[int, ...]) -> torch.nn.Module:
    path = Path(checkpoint_path)
    if path.name.startswith("baseline"):
        model: torch.nn.Module = BaselineMLP(input_dim=32 * 32 * 3, hidden_dims=hidden_dims, num_classes=10)
    else:
        model = PrunableMLP(input_dim=32 * 32 * 3, hidden_dims=hidden_dims, num_classes=10)

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def load_cifar_test_set(data_dir: str) -> datasets.CIFAR10 | None:
    root = Path(data_dir)
    if not root.exists():
        return None
    try:
        return datasets.CIFAR10(root=str(root), train=False, download=False)
    except RuntimeError:
        return None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    tensor = transform(image.convert("RGB"))
    return tensor.view(1, -1)


def predict(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
    return probs.squeeze(0)


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(dims) < 2 or len(dims) > 3:
        raise ValueError("Use 2 or 3 hidden dims, for example: 512,256,128")
    return dims


def show_topk(probs: torch.Tensor, k: int = 5) -> None:
    top_probs, top_idx = torch.topk(probs, k=k)
    rows = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        rows.append({"class": CIFAR10_CLASSES[idx], "probability": f"{100.0 * p:.2f}%"})
    st.table(rows)


def main() -> None:
    st.set_page_config(page_title=PROJECT_NAME, page_icon="🧠", layout="wide")

    st.title("GateLens CIFAR")
    st.caption("Interactive frontend for your self-pruning CIFAR-10 MLP checkpoints")

    with st.sidebar:
        st.header("Model Setup")
        output_dir = st.text_input("Checkpoint folder", value="outputs")
        hidden_dims_input = st.text_input("Hidden dims", value="512,256,128")
        data_dir = st.text_input("CIFAR data folder", value="data")

    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        st.error("No .pt checkpoints found. Train first or point to a valid outputs folder.")
        st.stop()

    checkpoint_name = st.selectbox("Choose checkpoint", [p.name for p in checkpoints], index=0)
    checkpoint_path = str(Path(output_dir) / checkpoint_name)

    try:
        hidden_dims = parse_hidden_dims(hidden_dims_input)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    model = load_model(checkpoint_path, hidden_dims)

    if isinstance(model, PrunableMLP):
        sparsity = model.sparsity_level(threshold=1e-2)
        st.info(f"Loaded prunable model | sparsity (<1e-2): {sparsity:.2f}%")
    else:
        st.info("Loaded baseline model (non-prunable)")

    source = st.radio("Image source", ["Upload image", "CIFAR test sample"], horizontal=True)

    input_image: Image.Image | None = None
    true_label: str | None = None

    if source == "Upload image":
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            input_image = Image.open(uploaded).convert("RGB")
    else:
        test_set = load_cifar_test_set(data_dir)
        if test_set is None:
            st.warning("CIFAR test set was not found in the selected data folder.")
        else:
            idx = st.slider("Sample index", min_value=0, max_value=len(test_set) - 1, value=0, step=1)
            sample, label_idx = test_set[idx]
            input_image = sample
            true_label = CIFAR10_CLASSES[label_idx]

    if input_image is None:
        st.stop()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Input Image")
        st.image(input_image.resize((256, 256)), caption="Resized preview", use_container_width=False)
        if true_label is not None:
            st.write(f"True label: **{true_label}**")

    with col2:
        st.subheader("Prediction")
        probs = predict(model, preprocess_image(input_image))
        pred_idx = int(torch.argmax(probs).item())
        pred_label = CIFAR10_CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx].item()) * 100.0

        st.metric("Top-1 class", pred_label)
        st.metric("Confidence", f"{pred_conf:.2f}%")
        show_topk(probs, k=5)

        if true_label is not None:
            is_correct = pred_label == true_label
            if is_correct:
                st.success("Prediction matches the ground-truth label.")
            else:
                st.error("Prediction does not match the ground-truth label.")


if __name__ == "__main__":
    main()
