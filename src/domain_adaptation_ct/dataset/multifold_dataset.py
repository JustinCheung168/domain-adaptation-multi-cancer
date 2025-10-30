from domain_adaptation_ct.dataset.image_dataset import BaseImageDataset
from domain_adaptation_ct.logging.log_mixin import LogMixin

import torch

class MultifoldDataset(torch.utils.data.Dataset, LogMixin):
    """
    A wrapper that combines multiple BaseImageDataset-like objects and
    provides a unified interface for __getitem__ and __len__.

    Behaves as if you concatenate the input datasets, without needing copies.
    """
    def __init__(self, datasets: list[BaseImageDataset]):
        if not datasets:
            raise ValueError("No datasets provided.")
        self.datasets = datasets

        # Precompute cumulative lengths for indexing
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)

    def __len__(self) -> int:
        """Combined length of all datasets included in this multifold dataset."""
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve item idx from the appropriate dataset."""
        ds_idx, local_idx = self._locate_dataset(idx)
        return self.datasets[ds_idx][local_idx]

    def _locate_dataset(self, idx: int):
        for dataset_num, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                if dataset_num == 0:
                    prev_cum = 0
                else:
                    prev_cum = self.cumulative_lengths[dataset_num - 1]
                return dataset_num, idx - prev_cum
        raise IndexError("Index out of range")