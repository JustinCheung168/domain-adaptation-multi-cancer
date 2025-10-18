
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, PreTrainedModel, ResNetConfig, set_seed
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification
from dataclasses import dataclass
from typing import Optional
from medmnist import OrganAMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import datetime
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner

# [1] Yaroslav Ganin, & Victor Lempitsky. (2015). Unsupervised Domain Adaptation by Backpropagation.
from gradient_reversal import GradientReversal

# Define the model output structure
@dataclass
class BranchedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    branch1_logits: torch.FloatTensor = None
    branch2_logits: torch.FloatTensor = None


class FixedGradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=x.dtype, device=x.device))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        alpha_tensor, = ctx.saved_tensors
        return -alpha_tensor * grad_output, None


class FixedGradientReversal(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return FixedGradientReversalFunction.apply(x, self.alpha)


# Define the branched model
class ResNetForMultiLabel(PreTrainedModel):
    config_class = ResNetConfig

    def __init__(self, config, num_d1_classes=11, num_d2_classes=5, loss_fn=torch.nn.CrossEntropyLoss(), lamb = 0.25, ld_scale = 1.0):
        super().__init__(config)
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_d1_classes) # Branch for original labels
        self.grad_reverse = FixedGradientReversal(alpha=lamb)
        self.branch2 = torch.nn.Sequential(
            self.grad_reverse,
            torch.nn.Linear(512, num_d2_classes)
            
        )

        self.ld_scale = ld_scale
        self.loss_fn = loss_fn
        self.post_init()

    def forward(self, pixel_values, labels1=None, labels2=None):
        features = self.resnet(pixel_values).pooler_output
        features = features.flatten(1)

        x = self.pre_branch(features)
        logits1 = self.branch1(x)
        logits2 = self.branch2(x)

        loss = self.loss_fn(logits1, labels1)
        loss = (loss * (1 - labels2)).mean()

        loss2 = self.loss_fn(logits2, labels2)
        total_loss = loss + (self.ld_scale * loss2)

        return BranchedOutput(loss=total_loss, branch1_logits=logits1, branch2_logits=logits2)

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

class CustomImageDataset(Dataset):
    def __init__(self, images, labels1, labels2, transform=None):
        # Remember to shuffle the data outside this class if needed
        self.images = images  # Should be torch.Tensor of shape [N, 3, 224, 224]
        self.labels1 = labels1
        self.labels2 = labels2

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
        label2 = int(self.labels2[idx])

        if self.transform:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "labels1": int(label1) if torch.is_tensor(label1) else label1,
            "labels2": int(label2) if torch.is_tensor(label2) else label2,
                }

from transformers import TrainerCallback

class LambdaUpdateCallback(TrainerCallback):
    def __init__(self, model, lambda_scheduler, total_epochs):
        self.model = model
        self.lambda_scheduler = lambda_scheduler
        self.total_epochs = total_epochs

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = state.epoch if state.epoch is not None else 0
        new_lambda = self.lambda_scheduler(epoch, self.total_epochs)
        if hasattr(self.model, "grad_reverse"):
            self.model.grad_reverse.alpha = float(new_lambda)


class CustomTrainer(Trainer):
    def __init__(self, *args, model=None, lambda_scheduler=None, total_epochs=50, **kwargs):
        super().__init__(*args, model=model, **kwargs)
        self.lambda_scheduler = lambda_scheduler
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def on_epoch_begin(self):
        if self.lambda_scheduler:
            new_lambda = self.lambda_scheduler(self.current_epoch, self.total_epochs)
            if hasattr(self.model, 'grad_reverse'):
                self.model.grad_reverse.alpha = new_lambda
            print(f"Epoch {self.current_epoch}: lambda = {new_lambda:.4f}")

        self.current_epoch += 1

def lambda_scheduler(epoch, total_epochs):

    p = epoch / total_epochs

    lambda_p = 2. / (1. + np.exp(-10 * p)) - 1
    return float(lambda_p)

# Need to expose training args
def train_model(train_dataset, eval_dataset, model, output_dir="./results", num_epochs=3, batch_size=32, train=True):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
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
        compute_metrics=make_metrics_fn(model),
        callbacks=[LambdaUpdateCallback(model, lambda_scheduler, num_epochs)]
    )


    if train:
        trainer.train()
    return trainer


def make_metrics_fn(model):
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
            "lambda": model.grad_reverse.alpha if hasattr(model, 'grad_reverse') else None
        }
    return compute_metrics

def dataset_load(file_path):

    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    labels1 = data['labels1']
    labels2 = data['labels2']

    return CustomImageDataset(images, labels1, labels2)

def combine_npzs(data_dir):
    combined_data = {}
    order = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'last']
    # Process files in the order specified by the 'order' list
    for pos in order:
        for filename in os.listdir(data_dir):
            file_split = filename.split('_')
            if pos in file_split and filename.endswith('.npz'):
                print(f"Processing file: {filename}")
                file_path = os.path.join(data_dir, filename)
                data = np.load(file_path)
                for key in data.files:
                    if key in combined_data:
                        combined_data[key] = np.concatenate((combined_data[key], data[key]), axis=0)
                    else:
                        combined_data[key] = data[key]
    return combined_data

def combine_data(orig_data, new_data):
    combined_data = dict(orig_data)

    combined_data['Ring_Artifact_v1'] = new_data['Ring_Artifact_v1']
    combined_data['ring_labels'] = new_data['label']

    return combined_data

from branched_resnet import CustomImageDataset

def normalize_image(image, mean=0.5, std=0.5):
    """
    Normalize an image tensor to have a mean and standard deviation.
    """
    return (image - mean) / std

def normalize_images(images, mean=0.5, std=0.5):
    """
    Normalize a list of images.
    """
    return [normalize_image(image, mean, std) for image in images]

def preprocess_data(data, distortions = [3], include_original=True, save_data=False, save_path=None):

    keys = list(data.keys())

    if include_original:
        images = [data[keys[0]]]
    else:
        images = []

    for distortion in distortions:
        images.append(data[keys[distortion]])

    labels = data[keys[1]]
    print('Normalizing images...')

    normalized_images = []
    for image in images:
        normalized_images.append(normalize_images(image))
    print('Normalization complete.')

    zero_labels = np.zeros_like(labels)
    one_labels = np.ones_like(labels)

    if include_original:
        domain_label_list = [zero_labels]
        expanded_label_list = [labels]
    else:
        domain_label_list = []
        expanded_label_list = []

    for _ in distortions:
        domain_label_list.append(one_labels)
        expanded_label_list.append(labels)

    # domain_labels = np.concatenate(domain_label_list, axis=0)
    # expanded_labels = np.concatenate(expanded_label_list, axis=0)

    print('Stacking labels...')

    domain_labels = np.vstack(domain_label_list)
    expanded_labels = np.vstack(expanded_label_list)

    print(f"Domain labels shape: {domain_labels.shape}")
    print(f"Expanded labels shape: {expanded_labels.shape}")

    print('Stacking images...')
    # concatenated_images = np.concatenate(normalized_images, axis=0)
    concatenated_images = np.vstack(normalized_images)


    # Shuffle the concatenated images and labels
    print('Shuffling data...')
    seed = 42
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(concatenated_images))
    concatenated_images = concatenated_images[shuffled_indices]
    expanded_labels = expanded_labels[shuffled_indices]
    domain_labels = domain_labels[shuffled_indices]
    print('Data shuffled.')

    if save_data:
        print('Saving preprocessed data...')
        if save_path is None:
            raise ValueError("save_path must be specified if save_data is True")
        np.savez_compressed(save_path, images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)

        print(f"Preprocessed data saved to {save_path}")
    
    print('Creating dataset...')
    dataset = CustomImageDataset(images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)
    print('Dataset created.')

    return dataset

def import_data(directory, save_path=None, save=False):

    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.npz'):
            file_path = os.path.join(directory, filename)
            loaded_data = np.load(file_path)
            data.append(loaded_data)

    # concatenate the data from all files
    all_data = {}
    for key in data[0].keys():
        all_data[key] = np.concatenate([d[key] for d in data], axis=0)

    # check the shape of the concatenated data
    for key, value in all_data.items():
        print(f"{key}: {value.shape}")  

    if save:
        if save_path is None:
            save_path = f'datasets/{directory}_concatenated_data.npz'
        np.savez(save_path, **all_data)
        print(f"Data saved to {save_path}")

    return all_data

from huggingface_hub import hf_hub_download, login

def hf_download(repo_id, filename, hf_token):

    login(token = hf_token)

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

    print(f"{filename} downloaded to {path}")

    return path



# def trainer_trainable(config, train_data, val_data, seed=42):
#     print("Training with config:")
#     for key, value in config.items():
#         print(f"{key}: {value}")
#     print("\n")

#     DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = f"./data/hyperparam_tuning_results/"

#     torch.manual_seed(seed)
#     set_seed(seed)

#     model = ResNetForMultiLabel(config=ResNetConfig(), num_d1_classes=11, num_d2_classes=2, lamb=0)

#     training_args = TrainingArguments(
#         output_dir=f"{output_dir}/results_{DATE}",
#         num_train_epochs=config["epochs"],
#         per_device_train_batch_size=config["batch_size"],
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         logging_dir='./logs',
#         logging_steps=10,
#         load_best_model_at_end=True,
#         learning_rate=config["lr"],
#         weight_decay=config["weight_decay"],
#         seed=seed,
#         optim=config["optimizer"]
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=val_data,
#         compute_metrics=make_metrics_fn(model),
#         callbacks=[LambdaUpdateCallback(model, lambda_scheduler, config["epochs"])]
#     )

#     trainer.train()
#     metrics = trainer.evaluate()
#     print(f"Evaluation metrics: {metrics}")
#     tune.report(
#         accuracy_branch1=metrics["accuracy_branch1"],
#         eval_loss=metrics["eval_loss"] if "eval_loss" in metrics else None
#             )




# import numpy as np
# import os

# def normalize_images_np(images, mean=0.5, std=0.5):
#     """
#     Vectorized normalization for a numpy array of images.
#     Accepts images as a numpy array of shape (N, H, W) or (N, H, W, C).
#     Returns float32 array.
#     """
#     arr = np.asarray(images)
#     # convert to float32 once
#     if arr.dtype != np.float32:
#         arr = arr.astype(np.float32)
#     # If grayscale [N, H, W] -> expand channel last so transforms.ToTensor works later
#     if arr.ndim == 3:
#         # keep shape (N, H, W) here; torchvision ToTensor accepts numpy HxW or HxWxC,
#         # but later in dataset we call transforms.ToTensor per-sample so either is fine.
#         pass
#     # Normalize
#     arr = (arr - mean) / std
#     return arr

# def preprocess_data2(data, distortions=[3], include_original=True, save_data=False, save_path=None, use_memmap=False):
#     """
#     Efficient preprocessing:
#       - data: dict-like with keys: [0]=images_original, [1]=labels, [distortion indices] = distorted images arrays
#       - distortions: list of indices into keys to use as distorted domains
#     """
#     keys = list(data.keys())
#     # canonicalize inputs to numpy arrays
#     labels = np.asarray(data[keys[1]])
#     n_samples = len(labels)
#     domain_count = len(distortions) + (1 if include_original else 0)
#     total = n_samples * domain_count

#     # pick first images array shape to determine image shape
#     # the arrays in data are expected to be shape (N, H, W) or (N, H, W, C)
#     sample_img = np.asarray(data[keys[0]])
#     img_shape = sample_img.shape[1:]  # (H, W) or (H, W, C)
#     out_shape = (total, *img_shape)
#     dtype = np.float32

#     # Option A: use memmap file if requested and small disk available
#     if use_memmap:
#         if save_path is None:
#             raise ValueError("save_path must be set if use_memmap=True")
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         memmap_file = save_path + ".mmap"
#         concatenated_images = np.memmap(memmap_file, mode="w+", dtype=dtype, shape=out_shape)
#     else:
#         concatenated_images = np.empty(out_shape, dtype=dtype)

#     # Prepare labels
#     expanded_labels = np.tile(labels, domain_count)           # shape (total,)
#     domain_labels = np.concatenate([
#         np.full(n_samples, 0, dtype=np.int64) if include_original else np.array([], dtype=np.int64),
#         *[np.full(n_samples, 1, dtype=np.int64) for _ in distortions]
#     ])

#     # Fill preallocated array in chunks, normalizing with vectorized function
#     idx = 0
#     if include_original:
#         imgs = normalize_images_np(data[keys[0]])
#         concatenated_images[idx:idx + n_samples] = imgs
#         idx += n_samples

#     for d in distortions:
#         imgs = normalize_images_np(data[keys[d]])
#         concatenated_images[idx:idx + n_samples] = imgs
#         idx += n_samples

#     # cast labels shapes to 1D to be safe
#     expanded_labels = np.asarray(expanded_labels).reshape(-1)
#     domain_labels = np.asarray(domain_labels).reshape(-1)

#     # Shuffle: create permutation and apply in-place view reordering (creates new views not extra copies)
#     rng = np.random.default_rng(42)
#     perm = rng.permutation(total)
#     concatenated_images = concatenated_images[perm]
#     expanded_labels = expanded_labels[perm]
#     domain_labels = domain_labels[perm]

#     # Optionally save (compressed)
#     if save_data:
#         if save_path is None:
#             raise ValueError("save_path must be specified if save_data is True")
#         np.savez_compressed(save_path, images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)

#     # Return dataset (your CustomImageDataset expects numpy arrays)
#     return CustomImageDataset(images=concatenated_images, labels1=expanded_labels, labels2=domain_labels)
