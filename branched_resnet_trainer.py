
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from dataclasses import dataclass
from typing import Optional
from medmnist import OrganAMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define the model output structure
@dataclass
class BranchedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    branch1_logits: torch.FloatTensor = None
    branch2_logits: torch.FloatTensor = None

# Define the branched model
class ResNetForMultiLabel(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config, num_d1_classes=11, num_d2_classes=5):
        super().__init__(config)
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_d1_classes) # Branch for original labels
        self.branch2 = torch.nn.Linear(512, num_d2_classes) # Branch for distortion labels

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.post_init()

    def forward(self, pixel_values, labels1=None, labels2=None):
        features = self.resnet(pixel_values).pooler_output
        features = features.flatten(1)

        x = self.pre_branch(features)
        logits1 = self.branch1(x)
        logits2 = self.branch2(x)

        loss = None
        if labels1 is not None and labels2 is not None:
            loss = self.loss_fn(logits1, labels1) + self.loss_fn(logits2, labels2)

        return BranchedOutput(loss=loss, branch1_logits=logits1, branch2_logits=logits2)

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
        image, label1 = self.data[idx]  # Unpack the tuple
        image = self.transform(image)
        label1 = int(label1)
        label2 = label1 % 5  # Dummy secondary label
        return {"pixel_values": image, "labels1": label1, "labels2": label2}


# Load datasets
train_ds = OrganAMNISTDataset("train")
val_ds = OrganAMNISTDataset("val")

# Initialize model
config = ResNetConfig.from_pretrained("microsoft/resnet-50")
model = ResNetForMultiLabel(config, num_d1_classes=11, num_d2_classes=5)

# Trainer setup
training_args = TrainingArguments(
    output_dir="./branched_resnet_organamnist",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    report_to="tensorboard",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False
)

# Metrics function
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    logits1, logits2 = preds

    labels1 = labels[0]
    labels2 = labels[1]

    # Convert to NumPy
    preds1 = np.argmax(logits1, axis=-1)
    preds2 = np.argmax(logits2, axis=-1)

    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    return {
        "accuracy_branch1": accuracy_score(labels1, preds1),
        "precision_branch1": precision_score(labels1, preds1, average="macro", zero_division=0),
        "recall_branch1": recall_score(labels1, preds1, average="macro", zero_division=0),
        "f1_branch1": f1_score(labels1, preds1, average="macro", zero_division=0),

        "accuracy_branch2": accuracy_score(labels2, preds2),
        "precision_branch2": precision_score(labels2, preds2, average="macro", zero_division=0),
        "recall_branch2": recall_score(labels2, preds2, average="macro", zero_division=0),
        "f1_branch2": f1_score(labels2, preds2, average="macro", zero_division=0),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics
)

# Train and save
trainer.train()
model.save_pretrained("branched_resnet_organamnist")

# Evaluate with metrics
test_ds = OrganAMNISTDataset("test")
test_results = trainer.evaluate(test_ds)
print("Test Results:", test_results)
# Save the test results
with open("test_results.txt", "w") as f:
    for key, value in test_results.items():
        f.write(f"{key}: {value}\n")
print("Test results saved to test_results.txt")