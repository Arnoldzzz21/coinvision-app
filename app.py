import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import json
import pandas as pd
import re

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="CoinVision", page_icon="🪙", layout="centered")

# ---------------------------
# Constants
# ---------------------------
NUM_CLASSES = 211
NUM_GROUPS = 32
IMG_SIZE = 300
MODEL_PATH = "models/best_model_hierarchical.pth"
CAT_TO_NAME_PATH = "cat_to_name.json"
LABEL_MAPPING_PATH = "label_mapping.json"
EXCHANGE_RATES_PATH = "exchange_rates.csv"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Maps the exact currency names used in this coin dataset (cat_to_name.json)
# to the ISO codes used in exchange_rates.csv. Confirmed against the actual
# file — all 32 currencies in the dataset have a matching ISO code.
CURRENCY_ALIASES = {
    "Australian dollar": "AUD",
    "Brazilian Real": "BRL",
    "British Pound": "GBP",
    "Canadian Dollar": "CAD",
    "Chilean Peso": "CLP",
    "Chinese Yuan Renminbi": "CNY",
    "Czech Koruna": "CZK",
    "Danish Krone": "DKK",
    "Euro": "EUR",
    "Hong Kong dollar": "HKD",
    "Hungarian Forint": "HUF",
    "Indian Rupee": "INR",
    "Indonesian Rupiah": "IDR",
    "Israeli New Shekel": "ILS",
    "Japanese Yen": "JPY",
    "Korean Won": "KRW",
    "Malaysian Ringgit": "MYR",
    "Mexican peso": "MXN",
    "New Zealand dollar": "NZD",
    "Norwegian Krone": "NOK",
    "Pakistan Rupee": "PKR",
    "Philipine peso": "PHP",
    "Polish Zloty": "PLN",
    "Russian Ruble": "RUB",
    "Singapore Dollar": "SGD",
    "South African Rand": "ZAR",
    "Swedish Krona": "SEK",
    "Swiss Franc": "CHF",
    "Thai Baht": "THB",
    "Turkish Lira": "TRY",
    "US Dollar": "USD",
    "taiwan Dollar": "TWD",
}


# ---------------------------
# Model definition — same architecture as Phase 11 (HierarchicalCoinNet).
# Both heads load so the checkpoint matches exactly, but at inference we
# only use class_head (flat argmax). Masking is NOT used: Phase 11 found the
# group_head plateaued at ~73% val accuracy, not enough for hard masking to
# beat the flat classifier (68.13% vs 52.13% test accuracy).
# ---------------------------
class HierarchicalCoinNet(nn.Module):
    def __init__(self, num_classes, num_groups):
        super().__init__()
        base = models.efficientnet_b3(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[1].in_features
        self.dropout = nn.Dropout(p=0.3)
        self.class_head = nn.Linear(in_features, num_classes)
        self.group_head = nn.Linear(in_features, num_groups)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.class_head(x), self.group_head(x)


# ---------------------------
# Grad-CAM — hooks on the shared backbone's last conv block
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        class_logits, _ = self.model(input_tensor)
        score = class_logits[0, class_idx]
        score.backward()

        weights = self.gradients[0].mean(dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:])
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


# ---------------------------
# Load resources (cached so they only load/register once per server process,
# not once per script rerun — Streamlit reruns the whole script on every
# widget interaction, so model + Grad-CAM hooks are built together in one
# cached function. Registering the hooks outside a cache would re-attach a
# new forward/backward hook to model.features[-1] on every rerun, leaking
# hooks for the life of the session.)
# ---------------------------
@st.cache_resource
def load_model_and_gradcam():
    model = HierarchicalCoinNet(NUM_CLASSES, NUM_GROUPS)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    grad_cam = GradCAM(model, target_layer=model.features[-1])
    return model, grad_cam


@st.cache_resource
def load_mappings():
    with open(CAT_TO_NAME_PATH, "r", encoding="utf-8") as f:
        cat_to_name = json.load(f)
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    # label_mapping.json stores class_to_idx nested under that key.
    class_to_idx = {str(k): int(v) for k, v in label_mapping["class_to_idx"].items()}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return cat_to_name, idx_to_class


@st.cache_resource
def load_exchange_rates():
    # exchange_rates.csv is a daily time series (one row per currency per
    # date), with 'value' = how many units of that currency equal 1 EUR.
    # Some currencies stop appearing after a certain date (the feed later
    # only tracks 7 major currencies), so instead of requiring one shared
    # date, we take each currency's own most recent available rate.
    df = pd.read_csv(EXCHANGE_RATES_PATH)
    df["date_parsed"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    latest = df.sort_values("date_parsed").groupby("currency").tail(1)
    return dict(zip(latest["currency"], latest["value"]))


model, grad_cam = load_model_and_gradcam()
cat_to_name, idx_to_class = load_mappings()
exchange_rates = load_exchange_rates()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------
# USD conversion
#
# exchange_rates give "units of currency X per 1 EUR". To go from a local
# amount to USD: first convert local -> EUR (divide by the local rate),
# then EUR -> USD (multiply by the USD rate). Example: 100 JPY, rate[JPY]
# ~178, rate[USD] ~1.16 -> (100 / 178) * 1.16 ~= $0.65
# ---------------------------
def parse_denomination(name):
    """Extract the leading numeric value from a denomination string, e.g. '10 Kurus' -> 10.0

    A few classes in cat_to_name.json write fractions as two space-separated
    integers instead of a slash (e.g. '1 2 Dollar' means 1/2 Dollar, '1 4 Dollar'
    means 1/4 Dollar): '1 2 New Sheqel' (id 88), '1 2 Franc' (id 181),
    '1 2 Dollar' (ids 185 and 210), '1 4 Dollar' (id 209). Those must be caught
    before the general parser below, which would otherwise stop at the first
    number and read them as a whole unit (1.0) instead of 0.5 / 0.25.
    """
    name = name.strip()

    frac_match = re.match(r"^(\d+)\s+(\d+)\s+[A-Za-z]", name)
    if frac_match:
        num, den = frac_match.groups()
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None

    match = re.match(r"[\d.,/]+", name)
    if not match:
        return None
    value_str = match.group().replace(",", ".")
    if "/" in value_str:
        num, den = value_str.split("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(value_str)
    except ValueError:
        return None


def convert_to_usd(denomination_str, currency_name, rates):
    value = parse_denomination(denomination_str)
    iso_code = CURRENCY_ALIASES.get(currency_name)
    if value is None or iso_code is None or iso_code not in rates or "USD" not in rates:
        return None
    value_in_eur = value / rates[iso_code]
    value_in_usd = value_in_eur * rates["USD"]
    return value_in_usd


# ---------------------------
# Prediction — flat argmax on class_head only (no group masking)
# ---------------------------
def predict_coin(image, top_k=3):
    input_tensor = transform(image).unsqueeze(0)

    class_logits, _ = model(input_tensor)
    probs = torch.softmax(class_logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        class_id = str(idx_to_class[idx.item()])
        full_name = cat_to_name.get(class_id, class_id)
        results.append((full_name, prob.item()))

    pred_class = top_idxs[0].item()
    cam = grad_cam.generate(input_tensor, pred_class)
    cam_resized = cv2.resize(cam, image.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_resized = np.array(image.resize(image.size))
    overlay = (orig_resized * 0.5 + heatmap * 0.5).astype(np.uint8)

    return results, overlay


# ---------------------------
# UI
# ---------------------------
st.title("🪙 CoinVision")
st.caption(
    "Identifica monedas de 211 clases distintas con un modelo EfficientNet-B3, "
    "con explainability vía Grad-CAM."
)

uploaded_file = st.file_uploader("Sube una foto de una moneda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen subida", use_container_width=True)

    with st.spinner("Analizando moneda..."):
        results, overlay = predict_coin(image)

    with col2:
        st.image(overlay, caption="Grad-CAM: región usada por el modelo", use_container_width=True)

    st.subheader("Predicciones")
    for i, (name, prob) in enumerate(results):
        parts = [p.strip() for p in name.split(",")]
        denomination = parts[0] if len(parts) > 0 else name
        currency = parts[1] if len(parts) > 1 else ""
        country = parts[2] if len(parts) > 2 else ""

        label = f"**{denomination}** — {currency} ({country})"

        if i == 0:
            st.success(f"🥇 {label} — {prob:.1%} de confianza")
            usd_value = convert_to_usd(denomination, currency, exchange_rates)
            if usd_value is not None:
                st.metric("Valor estimado en USD", f"${usd_value:.4f}")
            else:
                st.caption("Conversión a USD no disponible para esta moneda.")
        else:
            st.write(f"{i + 1}. {label} — {prob:.1%}")
else:
    st.info("Sube una imagen de una moneda para comenzar.")

st.divider()
st.caption("Modelo: EfficientNet-B3 (multi-task, class_head + group_head auxiliar) · Test accuracy 68.13% sobre 211 clases · Proyecto de portafolio de Data Science.")
