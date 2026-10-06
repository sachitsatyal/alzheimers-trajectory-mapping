"""Load the saved tensors (data/processed) for PyTorch training."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FLOAT_KEYS = ["X", "M", "tau", "dt_next", "MU", "PI"]
LONG_KEYS = ["Y", "S"]


def load_meta(processed_dir) -> dict:
    return json.loads((Path(processed_dir) / "meta.json").read_text())


class TARCCSequences(Dataset):
    """One item = one patient's padded sequence, as a dict of tensors."""

    def __init__(self, processed_dir, split: str):
        arrays = np.load(Path(processed_dir) / f"{split}.npz", allow_pickle=True)
        self.tensors = {k: torch.from_numpy(arrays[k]).float() for k in FLOAT_KEYS}
        self.tensors.update({k: torch.from_numpy(arrays[k]).long() for k in LONG_KEYS})
        self.patid = arrays["PATID"]

    def __len__(self):
        return self.tensors["X"].shape[0]

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.tensors.items()}


def make_loader(processed_dir, split: str, batch_size=32, shuffle=None, seed=0) -> DataLoader:
    """Training batches are shuffled (reproducibly); val/test keep their order."""
    shuffle = (split == "train") if shuffle is None else shuffle
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TARCCSequences(processed_dir, split), batch_size=batch_size,
                      shuffle=shuffle, generator=generator)