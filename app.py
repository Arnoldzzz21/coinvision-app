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

# The model is a closed-set classifier: it always picks one of the 211
# known classes (32 countries), even for a coin it was never trained on
# (e.g. an Argentine peso). When its own top-1 confidence is this low,
# that's a sign the coin is probably out of its known scope rather than
# a coin it actually recognizes with low certainty, so instead of
# asserting a likely-wrong country/currency we show a clear "not
# recognized" message. Tune this threshold based on real-world testing.
LOW_CONFIDENCE_THRESHOLD = 0.35
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

# Many coin denominations in this dataset are written in a currency's named
# minor subunit (e.g. "1 euro Cent" = 0.01 Euro, "1 Rappen" = 0.01 Swiss
# Franc, "1 Jiao" = 0.10 Yuan) rather than its major unit. parse_denomination()
# only reads the leading number, so without this table those coins would be
# valued as if that number were whole units of the major currency (e.g.
# "1 euro Cent" priced the same as "1 Euro"). Maps the subunit word
# (lowercase, as it appears right after the number in cat_to_name.json,
# singular/plural forms included) to its fraction of one major unit. Built
# by checking every denomination word actually present in the 211 classes;
# anything not listed here (Euro, Dollar, Peso, Krona, Forint, Yen, Won,
# Rupiah, ...) is a major unit and keeps a fraction of 1.0.
SUBUNIT_FRACTIONS = {
    "cent": 0.01, "cents": 0.01,
    "centavo": 0.01, "centavos": 0.01,
    "pence": 0.01, "penny": 0.01,
    "paise": 0.01,
    "agorot": 0.01,
    "sen": 0.01,
    "sentimo": 0.01, "sentimos": 0.01,
    "grosz": 0.01, "grosze": 0.01, "groszy": 0.01,
    "kopek": 0.01, "kopeks": 0.01,
    "ore": 0.01,
    "hellers": 0.01,
    "kurus": 0.01,
    "satang": 0.01,
    "rappen": 0.01,
    "euro cent": 0.01,
    "dime": 0.1,   # US dime = 1/10 dollar
    "jiao": 0.1,   # Chinese jiao = 1/10 yuan
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


def subunit_fraction(denomination_str):
    """Fraction of the currency's major unit that this denomination's word
    represents: 1.0 for a major-unit coin (Euro, Dollar, Peso, ...), or the
    looked-up fraction for a named subunit (Cent, Rappen, Hellers, Kurus,
    ... — see SUBUNIT_FRACTIONS above)."""
    name = denomination_str.strip()
    # Same "strip the leading number" shape as parse_denomination, but keep
    # the trailing word(s) instead of the number.
    match = re.match(r"^[\d.,/\s]+(.*)$", name)
    unit_word = (match.group(1) if match else name).strip().lower()
    return SUBUNIT_FRACTIONS.get(unit_word, 1.0)


def convert_to_usd(denomination_str, currency_name, rates):
    value = parse_denomination(denomination_str)
    iso_code = CURRENCY_ALIASES.get(currency_name)
    if value is None or iso_code is None or iso_code not in rates or "USD" not in rates:
        return None
    value_in_major_unit = value * subunit_fraction(denomination_str)
    value_in_eur = value_in_major_unit / rates[iso_code]
    value_in_usd = value_in_eur * rates["USD"]
    return value_in_usd


def split_coin_name(full_name):
    """Split a cat_to_name.json entry ('10 Kurus, Turkish Lira, turkey') into
    (denomination, currency, country)."""
    parts = [p.strip() for p in full_name.split(",")]
    denomination = parts[0] if len(parts) > 0 else full_name
    currency = parts[1] if len(parts) > 1 else ""
    country = parts[2] if len(parts) > 2 else ""
    return denomination, currency, country


# ---------------------------
# Coin value table — USD value of every one of the 211 known classes,
# precomputed once so we can find coins of similar purchasing power to a
# given prediction (used instead of showing the model's low-confidence
# runner-up guesses, which were confusing: e.g. a 1 Euro prediction next to
# "2 Zlote — 3.2%" reads as if the model is unsure between those, when
# really it's just what a flat top-3 softmax happens to rank next).
# ---------------------------
def build_coin_value_table(cat_to_name_map, rates):
    table = []
    for class_id, full_name in cat_to_name_map.items():
        denomination, currency, country = split_coin_name(full_name)
        usd_value = convert_to_usd(denomination, currency, rates)
        if usd_value is not None:
            table.append({
                "class_id": class_id,
                "denomination": denomination,
                "currency": currency,
                "country": country,
                "usd_value": usd_value,
            })
    return table


def find_similar_value_coins(usd_value, exclude_class_id, value_table, n=2):
    """Return the n known coins whose USD value is closest to usd_value,
    excluding the predicted coin itself."""
    candidates = [c for c in value_table if c["class_id"] != exclude_class_id]
    candidates.sort(key=lambda c: abs(c["usd_value"] - usd_value))
    return candidates[:n]


coin_value_table = build_coin_value_table(cat_to_name, exchange_rates)


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
        results.append((class_id, full_name, prob.item()))

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
    "Identifies coins from 211 different classes using an EfficientNet-B3 "
    "model, with explainability via Grad-CAM."
)

uploaded_file = st.file_uploader("Upload a photo of a coin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing coin..."):
        results, overlay = predict_coin(image)

    with col2:
        st.image(overlay, caption="Grad-CAM: region used by the model", use_container_width=True)

    st.subheader("Predictions")

    top1_prob = results[0][1]

    if top1_prob < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "⚠️ Coin not recognized with confidence. This may not be one of "
            "the coins this model was trained on (it recognizes 211 "
            "denominations from 32 countries). No country or USD value is "
            "shown, since it would most likely be wrong."
        )
        with st.expander("Show closest matches anyway (low confidence)"):
            for i, (class_id, name, prob) in enumerate(results):
                denomination, currency, country = split_coin_name(name)
                st.write(f"{i + 1}. **{denomination}** — {currency} ({country}) — {prob:.1%}")
    else:
        class_id, name, prob = results[0]
        denomination, currency, country = split_coin_name(name)
        label = f"**{denomination}** — {currency} ({country})"

        st.success(f"🥇 {label} — {prob:.1%} confidence")

        usd_value = convert_to_usd(denomination, currency, exchange_rates)
        if usd_value is not None:
            st.metric("Estimated value in USD", f"${usd_value:.4f}")

            similar_coins = find_similar_value_coins(usd_value, class_id, coin_value_table, n=2)
            if similar_coins:
                st.caption("Other known coins worth about the same:")
                for coin in similar_coins:
                    coin_label = f"**{coin['denomination']}** — {coin['currency']} ({coin['country']})"
                    st.write(f"{coin_label} — ≈ ${coin['usd_value']:.4f}")
        else:
            st.caption("USD conversion not available for this coin.")
else:
    st.info("Upload an image of a coin to get started.")

st.divider()
st.caption("Model: EfficientNet-B3 (multi-task, class_head + auxiliary group_head) · Test accuracy 68.13% across 211 classes · Data Science portfolio project."
)