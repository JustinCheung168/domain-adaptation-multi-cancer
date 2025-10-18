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


# if __name__ == '__main__':

#     # Device selection: CUDA > MPS > CPU

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")
#     print(f"Using device: {device}")



    
#     # Transforms (no distortion, just normalization)

#     base_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
    
    
#     transform_gray = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5],
#                             [0.5, 0.5, 0.5])
#     ])

#     #Dataset & DataLoader
#     # 'cropped_224': transform_rgb,
#     # 'cropped_224_rotated': base_transform,
#     # 'cropped_contrast_5': base_transform,
#     folders = {
#         'cropped_224': base_transform,
#         'cropped_224_rotated': base_transform,
#         'cropped_contrast_5': base_transform,
#         'highpass_0.7': base_transform,
#         'lowpass_7': base_transform,
#         'phase_scrambled_90': base_transform,
#         'saltpepper_0.2': base_transform,
#         'uniform_035': base_transform,

#     }
#             # 'cropped_224_gray': transform_gray,
#     # BATCH SIZE******************************
#     batch_size = 200
    
#     for name, transform in folders.items():
#         dataset = FlatImageDataset(
#             csv_path="flat_train_60percent.csv",
#             img_dir=name,
#             transform=transform,
#             subsample_ratio=0.3176,
#             seed=42
#         )
        
#         val_dataset = FlatImageDataset(
#             csv_path="flat_val_20percent.csv",
#             img_dir=name,
#             transform=transform,
#             subsample_ratio=0.19057,
#             seed=337  # Different seed for val split
#         )
        
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12)
        
#      # Model

#         model = models.resnet50(weights=None)
#         model.fc = nn.Linear(model.fc.in_features, len(dataset.label_to_index))
#         model.to(device)

#         # Loss, Optimizer, LR Scheduler
    
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


#         # Training Loop

#         epochs = 100
#         for epoch in range(epochs):
#             model.train()
#             running_loss = 0.0
#             correct = 0

#             for images, labels in dataloader:
#                 images, labels = images.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item() * images.size(0)
#                 preds = outputs.argmax(dim=1)
#                 correct += (preds == labels).sum().item()

#             epoch_loss = running_loss / len(dataset)
#             train_acc = correct / len(dataset)
#             print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")
#             scheduler.step()

#             # Eval every 10 epochs
#             if (epoch + 1) % 10 == 0:
#                 model.eval()
#                 val_correct = 0
#                 val_total = 0
#                 with torch.no_grad():
#                     for val_images, val_labels in val_loader:
#                         val_images, val_labels = val_images.to(device), val_labels.to(device)
#                         val_outputs = model(val_images)
#                         val_preds = val_outputs.argmax(dim=1)
#                         val_correct += (val_preds == val_labels).sum().item()
#                         val_total += val_labels.size(0)
#                 val_acc = val_correct / val_total
#                 print(f"[{name}] Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
#                 with open("log.txt", "a") as file:
#                     file.write(f"{name}, Epoch {epoch+1}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n")

#                 checkpoint_path = f"{name}_20ercent_epoch{epoch+1}.pth"
#                 torch.save(model.state_dict(), checkpoint_path)
#                 print(f"{name} Checkpoint saved at {checkpoint_path}")
if __name__ == '__main__':

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Define transforms
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5])
    ])

    folders = {
        'cropped_224': base_transform,
        'cropped_224_gray': transform_gray,
        'cropped_224_rotated': base_transform,
        'cropped_contrast_5': base_transform,
        'highpass_0.7': base_transform,
        'lowpass_7': base_transform,
        'phase_scrambled_90': base_transform,
        'saltpepper_0.2': base_transform,
        'uniform_035': base_transform,
    }

    batch_size = 200

    # Loop through each checkpoint (trained on one distortion)
    for train_distortion, transform in folders.items():
        print(f"\n=== Evaluating model trained on: {train_distortion} ===")

        # Load the model
        dummy_dataset = FlatImageDataset(
            csv_path="flat_val_20percent.csv",
            img_dir=train_distortion,
            transform=transform,
            subsample_ratio=0.3176,
            seed=337  # Matches val split
        )
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(dummy_dataset.label_to_index))

        checkpoint_path = f"{train_distortion}_20percent_epoch100.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        # Evaluate on all distortion test sets
        for test_distortion, test_transform in folders.items():
            test_dataset = FlatImageDataset(
                csv_path="flat_val_20percent.csv",
                img_dir=test_distortion,
                transform=test_transform,
                subsample_ratio=0.19057,  # match val setup
                seed=337
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total
            print(f"Train [{train_distortion}] → Test [{test_distortion}] — Acc: {acc:.4f}")
            with open("cross_eval_log.txt", "a") as f:
                f.write(f"{train_distortion},{test_distortion},{acc:.4f}\n")




#BELOW IS FOR TEST SETS
# if __name__ == '__main__':

#     # Device selection
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")
#     print(f"Using device: {device}")

#     # Define transforms
#     base_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])
    
#     transform_gray = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5],
#                             [0.5, 0.5, 0.5])
#     ])

#     folders = {
#         'cropped_224': base_transform,
#         'cropped_224_gray': transform_gray,
#         'cropped_224_rotated': base_transform,
#         'cropped_contrast_5': base_transform,
#         'highpass_0.7': base_transform,
#         'lowpass_7': base_transform,
#         'phase_scrambled_90': base_transform,
#         'saltpepper_0.2': base_transform,
#         'uniform_035': base_transform,
#     }

#     batch_size = 200

#     # Loop through each checkpoint (trained on one distortion)
#     for train_distortion, transform in folders.items():
#         print(f"\n=== Evaluating model trained on: {train_distortion} ===")

#         # Load the model
#         dummy_dataset = FlatImageDataset(
#             csv_path="flat_val_20percent.csv",
#             img_dir=train_distortion,
#             transform=transform,
#             subsample_ratio=0.3176,
#             seed=337  # Matches val split
#         )
#         model = models.resnet50(weights=None)
#         model.fc = nn.Linear(model.fc.in_features, len(dummy_dataset.label_to_index))

#         checkpoint_path = f"{train_distortion}_20percent_epoch100.pth"
#         model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#         model.to(device)
#         model.eval()

#         # Evaluate on all distortion test sets
#         for test_distortion, test_transform in folders.items():
#             test_dataset = FlatImageDataset(
#                 csv_path="flat_test_20percent.csv",
#                 img_dir=test_distortion,
#                 transform=test_transform,
#                 subsample_ratio=0.19057,  # match val setup
#                 seed=337
#             )
#             test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for images, labels in test_loader:
#                     images, labels = images.to(device), labels.to(device)
#                     outputs = model(images)
#                     preds = outputs.argmax(dim=1)
#                     correct += (preds == labels).sum().item()
#                     total += labels.size(0)

#             acc = correct / total
#             print(f"Train [{train_distortion}] → Test [{test_distortion}] — Acc: {acc:.4f}")
#             with open("cross_eval_log.txt", "a") as f:
#                 f.write(f"{train_distortion},{test_distortion},{acc:.4f}\n")
