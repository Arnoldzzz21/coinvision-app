# CoinVision

Coin classifier that identifies the country and denomination of a coin from a photo, powered by a fine-tuned EfficientNet-B3 model with Grad-CAM explainability and automatic USD value conversion.

**Live app:** https://coinvision-arnoldo.streamlit.app/

## Features

- Upload a coin photo (JPG or PNG) and get an instant prediction.
- Top-3 predicted classes with confidence scores, out of 231 coin classes from 36 currencies.
- Grad-CAM heatmap overlay showing which part of the image the model focused on.
- Automatic conversion of the predicted denomination into an estimated USD value using daily exchange rates.

## Model

The classifier is a HierarchicalCoinNet built on top of an EfficientNet-B3 backbone (transfer learning + fine-tuning), pre-trained on ImageNet. It uses two output heads trained jointly: a class_head that predicts the exact coin class (231 classes) and an auxiliary group_head that predicts a coarser currency group (41 groups). At inference time only the class_head is used (flat argmax) -- the auxiliary head helped regularize training but plateaued at a lower validation accuracy, so hard masking by group was not used in the final model. The final model reaches 66.25% test accuracy across the 231 classes. Grad-CAM hooks are attached to the last convolutional block of the backbone to generate a heatmap showing which region of the image most influenced the prediction.

See [`training/`](training/) for the full training pipeline: dataset build (sourced entirely from Wikimedia Commons), EDA, base training, fine-tuning, Grad-CAM, and a documented hierarchical two-head experiment that wasn't used in production.

## Dataset

Each coin class corresponds to a denomination, currency, and country (for example "10 Kurus, Turkish Lira, Turkey"), stored in cat_to_name.json. label_mapping.json holds the class-to-index mapping used to align model outputs with class names. exchange_rates.csv is a daily time series of exchange rates (units of currency per 1 EUR) for the 36 currencies represented in the dataset, used to convert the predicted denomination into an estimated USD value.

## Tech stack

Python, PyTorch, Torchvision, OpenCV, Streamlit, Pandas, and NumPy.

## Project structure

```
coinvision-app/
├── app.py                  # Streamlit app: model, Grad-CAM, UI
├── models/
│   └── best_model_hierarchical.pth
├── cat_to_name.json         # class id -> "denomination, currency, country"
├── label_mapping.json       # class_to_idx mapping
├── exchange_rates.csv       # daily exchange rates for USD conversion
├── requirements.txt
└── training/                # full training pipeline (notebook, dataset builder, docs)
```

## Running locally

```bash
git clone https://github.com/Arnoldzzz21/coinvision-app.git
cd coinvision-app
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud, running on Python 3.11 with CPU-only PyTorch and Torchvision wheels (see requirements.txt).

## Author

Built by Arnoldo Cuellar as part of a personal Data Science portfolio.
