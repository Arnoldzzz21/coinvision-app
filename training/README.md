# CoinVision — Training Notebook

This folder holds the full training pipeline behind the [CoinVision app](../README.md): the Jupyter notebook used to build the dataset manifest, run EDA, train the model, fine-tune it, and evaluate it. It's included here so the whole process is reproducible and auditable, not just the final deployed weights.

The deployed app only needs `models/best_model_hierarchical.pth`, `cat_to_name.json`, and `label_mapping.json` (already in the repo root) — you do **not** need anything in this folder to run the app itself. This is for anyone who wants to see how the model was built, reproduce it, or extend it to new coin classes.

## What's in here

| File | Purpose |
|---|---|
| `Codevision.ipynb` | The full pipeline: EDA, dataset build, training, fine-tuning, Grad-CAM, hierarchical experiment, evaluation. Includes executed outputs/plots from the actual run. |
| `build_new_classes.py` | Downloads coin images for new classes from Wikimedia Commons (see "Dataset" below). |
| `coin_dataset.py` | Shared `Dataset`/transform classes (`CoinDataset`, `AddGaussianNoise`), imported by the notebook — kept in a real `.py` module (not defined inline) so PyTorch `DataLoader` workers can pickle them on Windows. |
| `requirements-training.txt` | Python dependencies for training (separate from the app's CPU-only `requirements.txt`). |

## Dataset

The dataset is **not included in this repo** — it's ~7,500 coin photos across 231 classes, sourced entirely from [Wikimedia Commons](https://commons.wikimedia.org/) (public domain / freely licensed images), so there's no copyrighted or scraped-from-marketplaces content involved.

To rebuild it:

1. Use `build_new_classes.py` to download images per class from Wikimedia Commons. It queries the Commons API by category/search term, downloads candidate images, filters out corrupted/duplicate files, and splits each class into `data/train/<class_id>/` and `data/test/<class_id>/`.
2. Maintain a `cat_to_name.json` (class id → "denomination, currency, country") and `new_classes_manifest.csv` (class id → search terms/categories to query) to drive the downloader.
3. Point the notebook's `TRAIN_DIR`/`TEST_DIR` paths at your local `data/` folder and run it top to bottom.

Classes with very few available photos on Commons are handled specially in the notebook (singleton classes skip the train/validation stratified split and go straight to train; classes with fewer than ~3 total images get 0 held out for test). Five currencies (El Salvador, Panama, Morocco, Tunisia, Libya) were deliberately excluded from the deployed model because Commons has too few photos of them to classify reliably — they're flagged in the notebook if you want to pick that work back up once more images are available.

## Running the notebook

```bash
cd training
pip install -r requirements-training.txt
jupyter notebook Codevision.ipynb
```

Requirements:
- A GPU is strongly recommended (training + fine-tuning + the hierarchical experiment together took ~20 minutes on an NVIDIA GPU; CPU-only would take considerably longer). `requirements-training.txt` defaults to a CUDA build of PyTorch — swap the index URL for your own CUDA version, or drop it for CPU-only.
- The dataset built per the section above, with `data/train/` and `data/test/` populated.
- `cat_to_name.json` and `new_classes_manifest.csv` in the working directory (copies live in the repo root and in `..`).

The notebook runs sequentially top to bottom: dataset load → EDA → manifest/split → base training (frozen backbone) → evaluation → fine-tuning (last 2 blocks unfrozen) → evaluation → Grad-CAM visualization → a hierarchical (class + currency-group) two-head experiment for comparison. The final deployed model uses the fine-tuned flat classifier (the hierarchical variant underperformed with hard group-masking, so it's kept as a documented experiment rather than used in production).

## Results

- Final flat (fine-tuned) model: ~66% test accuracy across the 231 deployed classes.
- The hierarchical group-masked variant plateaued lower and was not used in production — see the last few cells of the notebook for the comparison.

## License / attribution

All training images come from Wikimedia Commons under their respective free-use licenses. No copyrighted or third-party proprietary images are used anywhere in this dataset or repo.
