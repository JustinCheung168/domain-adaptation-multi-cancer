
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models, transforms
import pandas as pd
import os
# ------------------------------------
# CData
# ------------------------------------
class FlatImageDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, subsample_ratio=None, seed=42):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Encode class labels as integers
        self.label_to_index = {label: i for i, label in enumerate(sorted(self.data['label'].unique()))}
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        self.data['label_index'] = self.data['label'].map(self.label_to_index)

        # Subsample with fixed seed
        if subsample_ratio is not None:
            # self.data = self.data.sample(frac=subsample_ratio, random_state=seed).reset_index(drop=True)
            self.data = (
            self.data
            .groupby('label', group_keys=False)
            .sample(frac=subsample_ratio, random_state=seed)
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = row['label_index']

        if self.transform:
            image = self.transform(image)

        return image, label
    
    
    
