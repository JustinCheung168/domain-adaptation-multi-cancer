import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import Trainer, TrainingArguments, PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from dataclasses import dataclass
from typing import Optional
from medmnist import OrganAMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# [1] Yaroslav Ganin, & Victor Lempitsky. (2015). Unsupervised Domain Adaptation by Backpropagation.
# from gradient_reversal import GradientReversal

# Define the model output structure
@dataclass
class BranchedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

# Define the branched model
class ResNetForMultiLabel(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config, num_classes=11):
        super().__init__(config)
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, pixel_values, labels=None):
        features = self.resnet(pixel_values).pooler_output
        x = self.pre_branch(features)
        logits = self.branch1(x)

        loss = self.loss_fn(logits, labels) if labels is not None else None
        return BranchedOutput(loss=loss, logits=logits)

# Prepare OrganAMNIST Dataset with dummy secondary labels for now
class OrganAMNISTDataset(Dataset):
    def __init__(self, split="train"):
        dataset = OrganAMNIST(split=split, size=224, download=True)
        self.data = dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {
            "pixel_values": self.transform(image),
            "labels": int(label)
        }

class CustomImageDataset(Dataset):
    def __init__(self, images, labels1, labels2=None, transform=None):
        self.images = images  # Should be torch.Tensor of shape [N, 3, 224, 224]
        self.labels1 = labels1
        # self.labels2 = labels2
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Grayscale to 3-channel
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label1 = int(self.labels1[idx])
        # label2 = int(self.labels2[idx]) if self.labels2 is not None else 0

        if self.transform:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "labels": int(label1) if torch.is_tensor(label1) else label1,
            # "labels2": int(label2) if torch.is_tensor(label2) else label2,
        }

def evaluate_model(eval_dataset, model, output_dir="./results", num_epochs=3, batch_size=32):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        # log_level="info",
        logging_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_f1",
        # greater_is_better=True,            
        learning_rate=0.1,
        weight_decay=1e-4,
        seed=42,
        optim="sgd"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    metrics = trainer.evaluate()
    return metrics


def train_model(train_dataset, eval_dataset, model, output_dir="./results", num_epochs=3, batch_size=32):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        # log_level="info",
        logging_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_f1",
        # greater_is_better=True,            
        learning_rate=0.1,
        weight_decay=1e-4,
        seed=42,
        optim="sgd"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer





