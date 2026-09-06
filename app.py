import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import pandas as pd
import re

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="CoinVision", page_icon="🪙", layout="centered")

# Streamlit Community Cloud's free tier gives this app a small, fixed memory
# ceiling (it crashed with "This app has gone over its resource limits" in
# production). PyTorch on CPU spins up one worker thread per available core
# by default for intra-op parallelism, and each of those threads keeps its
# own scratch buffers alive for the life of the process — on a
# resource-constrained, effectively single-core container that costs memory
# for no real speed benefit. Pinning it to 1 thread removes that overhead.
torch.set_num_threads(1)

# ---------------------------
# Constants
# ---------------------------
NUM_CLASSES = 231
NUM_GROUPS = 41
IMG_SIZE = 300
MODEL_PATH = "models/best_model_hierarchical.pth"

# The model is a closed-set classifier: it always picks one of the 231
# known classes (36 countries), even for a coin it was never trained on
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

    # --- Fase 2 (2026-09-05): 5 monedas vigentes agregadas al dataset,
    # todas ya presentes en exchange_rates.csv bajo estos codigos ISO. ---
    "Salvadoran Colon": "SVC",
    "Panamanian Balboa": "PAB",
    "Moroccan Dirham": "MAD",
    "Tunisian Dinar": "TND",
    "Libyan Dinar": "LYD",
}

# Fase 1 (peseta, escudo, guilder, lira) agrego 4 monedas que dejaron de
# existir antes de que exchange_rates.csv empezara a registrar tasas
# (ninguna de las 4 aparece en el CSV bajo ningun codigo ISO, verificado).
# En vez de una tasa de mercado, usamos la tasa fija e irrevocable que la
# Union Europea fijo al introducir el euro para cada pais -- es una
# constante legal, no cambia nunca, y es la conversion correcta para
# monedas fisicas que ya no cotizan. Unidades de la moneda local por 1 EUR.
FIXED_EUR_LEGACY_RATES = {
    "Spanish Peseta": 166.386,
    "Portuguese Escudo": 200.482,
    "Dutch Guilder": 2.20371,
    "Italian Lira": 1936.27,
}

# Many coin denominations in this dataset are written in a currency's named
# minor subunit (e.g. "1 euro Cent" = 0.01 Euro, "1 Rappen" = 0.01 Swiss
# Franc, "1 Jiao" = 0.10 Yuan) rather than its major unit. parse_denomination()
# only reads the leading number, so without this table those coins would be
# valued as if that number were whole units of the major currency (e.g.
# "1 euro Cent" priced the same as "1 Euro"). Maps the subunit word
# (lowercase, as it appears right after the number in cat_to_name.json,
# singular/plural forms included) to its fraction of one major unit. Built
# by checking every denomination word actually present in the 231 classes;
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

    # --- Fase 2 (2026-09-05) ---
    "centesimo": 0.01, "centesimos": 0.01,   # Panama (also written centésimo)
    "centésimo": 0.01, "centésimos": 0.01,
    "centime": 0.01, "centimes": 0.01,       # Morocco
    "millime": 0.001, "millimes": 0.001,     # Tunisia: 1 dinar = 1000 millimes,
                                              # NOT 100 -- do not default this to 0.01.
}

# "Dirham" is ambiguous on its own: it is Morocco's MAJOR unit (1 Dirham =
# 1 Dirham, fraction 1.0 like Dollar/Peso/Euro) but Libya's named SUBUNIT
# (1 Libyan Dinar = 1000 Dirham, so a Libyan "50 Dirham" coin is worth
# 0.05 Dinar). SUBUNIT_FRACTIONS keys only on the word, so it cannot hold
# both meanings at once -- this table overrides it by currency, checked
# first in subunit_fraction() below.
SUBUNIT_FRACTIONS_BY_CURRENCY = {
    "Libyan Dinar": {"dirham": 0.001, "dirhams": 0.001},
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
#
# generate() takes the class_logits already produced by a forward pass the
# caller ran (see predict_coin below) instead of running the model again,
# and gets its gradient with torch.autograd.grad() targeted directly at the
# hooked activation instead of model.zero_grad() + score.backward(). The
# original score.backward() had no way to stop early: autograd walks the
# whole graph back to every leaf that requires grad, so it was recomputing
# and storing a gradient for all ~11M backbone parameters (in
# model.<param>.grad, kept around until the next call) on every single
# uploaded coin just to read the gradient at one intermediate layer.
# torch.autograd.grad(score, self.activations, ...) computes only the
# gradient at that one tensor and stops the backward pass there — it never
# touches the earlier conv layers or leaves anything in .grad.
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        # Keep the graph attached here (no .detach()) — generate() needs it
        # to compute torch.autograd.grad() against this exact tensor.
        self.activations = output

    def generate(self, class_logits, class_idx):
        score = class_logits[0, class_idx]
        gradients = torch.autograd.grad(score, self.activations, retain_graph=False)[0]
        activations = self.activations.detach()

        weights = gradients[0].mean(dim=(1, 2))
        cam = torch.zeros(activations.shape[2:])
        for i, w in enumerate(weights):
            cam += w * activations[0, i]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


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


def subunit_fraction(denomination_str, currency_name=None):
    """Fraction of the currency's major unit that this denomination's word
    represents: 1.0 for a major-unit coin (Euro, Dollar, Peso, ...), or the
    looked-up fraction for a named subunit (Cent, Rappen, Hellers, Kurus,
    ... — see SUBUNIT_FRACTIONS above). currency_name resolves words that
    mean different things in different currencies (see
    SUBUNIT_FRACTIONS_BY_CURRENCY, e.g. "Dirham")."""
    name = denomination_str.strip()
    # Same "strip the leading number" shape as parse_denomination, but keep
    # the trailing word(s) instead of the number.
    match = re.match(r"^[\d.,/\s]+(.*)$", name)
    unit_word = (match.group(1) if match else name).strip().lower()
    override_table = SUBUNIT_FRACTIONS_BY_CURRENCY.get(currency_name)
    if override_table and unit_word in override_table:
        return override_table[unit_word]
    return SUBUNIT_FRACTIONS.get(unit_word, 1.0)


def convert_to_usd(denomination_str, currency_name, rates):
    value = parse_denomination(denomination_str)
    if value is None or "USD" not in rates:
        return None
    value_in_major_unit = value * subunit_fraction(denomination_str, currency_name)

    fixed_eur_rate = FIXED_EUR_LEGACY_RATES.get(currency_name)
    if fixed_eur_rate is not None:
        # Discontinued pre-euro currency: fixed legal rate, not a market one.
        value_in_eur = value_in_major_unit / fixed_eur_rate
    else:
        iso_code = CURRENCY_ALIASES.get(currency_name)
        if iso_code is None or iso_code not in rates:
            return None
        value_in_eur = value_in_major_unit / rates[iso_code]

    return value_in_eur * rates["USD"]


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
# Grad-CAM heatmap rendering — plain PIL/numpy, no opencv.
#
# opencv-python(-headless) was only used here for a resize, a Jet colormap
# lookup and a channel swap — all things PIL/numpy already do. Dropping it
# removes a large C++ library (and its import-time memory footprint) from a
# process that's already tight on Streamlit Community Cloud's free-tier
# resource limit, permanently rather than only under load.
# ---------------------------
_JET_CONTROL_POINTS = np.array([
    [0, 0, 0.50],   # dark blue
    [0, 0, 1.00],   # blue
    [0, 0.50, 1.00],  # cyan-blue
    [0, 1.00, 1.00],  # cyan
    [0.50, 1.00, 0.50],  # green
    [1.00, 1.00, 0],  # yellow
    [1.00, 0.50, 0],  # orange
    [1.00, 0, 0],  # red
    [0.50, 0, 0],   # dark red
], dtype=np.float32)


def _resize_grayscale_array(arr, size):
    """Resize a 2D float array in [0, 1] to `size` = (width, height)."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _apply_jet_colormap(gray_uint8):
    """Map an HxW uint8 array to an HxWx3 RGB array, approximating
    matplotlib/OpenCV's "Jet" colormap via linear interpolation between a
    handful of control points — enough fidelity for a visual explainability
    overlay without pulling in opencv or matplotlib as a dependency."""
    xs = np.linspace(0, 255, len(_JET_CONTROL_POINTS))
    flat = gray_uint8.ravel().astype(np.float32)
    channels = [np.interp(flat, xs, _JET_CONTROL_POINTS[:, c]) for c in range(3)]
    rgb = np.stack(channels, axis=-1).reshape(gray_uint8.shape + (3,))
    return (rgb * 255).astype(np.uint8)


# ---------------------------
# Prediction — flat argmax on class_head only (no group masking)
# ---------------------------
def predict_coin(image, top_k=3):
    input_tensor = transform(image).unsqueeze(0)

    # Single forward pass, reused for both the classification result and
    # the Grad-CAM gradient below — the model used to run twice per upload
    # (once here, once again inside grad_cam.generate()) for no benefit.
    class_logits, _ = model(input_tensor)
    probs = torch.softmax(class_logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        class_id = str(idx_to_class[idx.item()])
        full_name = cat_to_name.get(class_id, class_id)
        results.append((class_id, full_name, prob.item()))

    pred_class = top_idxs[0].item()
    cam = grad_cam.generate(class_logits, pred_class)
    cam_resized = _resize_grayscale_array(cam, image.size)
    heatmap = _apply_jet_colormap(np.uint8(255 * cam_resized))
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

    top1_prob = results[0][2]

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
st.caption("Model: EfficientNet-B3 (multi-task, class_head + auxiliary group_head) · Test accuracy 66.25% across 231 classes · Data Science portfolio project."
)
