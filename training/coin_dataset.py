"""Shared Dataset/transform classes for the CoinVision training notebook
(Codevision.ipynb).

These used to be defined directly inside notebook cells. They were moved here
so that PyTorch DataLoader worker processes (num_workers > 0) can pickle and
re-import them. On Windows, multiprocessing uses the "spawn" start method: a
worker process needs to import every class used by the Dataset/transform
pipeline from a real module. A class defined inside a Jupyter cell lives in
`__main__`, which the ipykernel process cannot re-import from a worker, so
num_workers had to stay 0 (single-threaded data loading -> the main cost of
the ~20 min training runs). With these classes here instead, the notebook can
use num_workers > 0 safely.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class CoinDataset(Dataset):
    def __init__(self, manifest_df, split, transform=None):
        self.data = manifest_df[manifest_df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["file_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["label_idx"]
        return image, label
