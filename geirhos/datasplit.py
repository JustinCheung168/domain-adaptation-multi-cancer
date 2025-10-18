import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from dataset import *


#load full dataset
df = pd.read_csv("flat_train_labels.csv")

#split ratios
train_frac = 0.60
val_frac = 0.20
test_frac = 0.20

#stratify by 60 first. 40 leftover
train_df = df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(frac=train_frac, random_state=42)
)
#remove training data
remaining_df = df.drop(train_df.index)

#split 40 into 20/20
val_df = remaining_df.groupby("label", group_keys=False).apply(
    lambda x: x.sample(frac=0.5, random_state=42)
)

#remainder 20
test_df = remaining_df.drop(val_df.index)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

#save to CSV
train_df.to_csv("flat_train_60percent.csv", index=False)
val_df.to_csv("flat_val_20percent.csv", index=False)
test_df.to_csv("flat_test_20percent.csv", index=False)
