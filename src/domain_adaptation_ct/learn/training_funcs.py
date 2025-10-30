import datetime
import os
import logging
from typing import Optional, Callable

from domain_adaptation_ct.config.experiment_config import TrainingConfig, EvaluationConfig
from domain_adaptation_ct.dataset.image_dataset import DATASET_REGISTRY
from domain_adaptation_ct.dataset.multifold_dataset import MultifoldDataset
from domain_adaptation_ct.learn.architectures import ARCHITECTURE_REGISTRY
from domain_adaptation_ct.learn.lambda_schedules import LAMBDA_SCHEDULER_REGISTRY, LambdaUpdateCallback
from domain_adaptation_ct.learn.metrics import make_metrics_fn
from domain_adaptation_ct.learn.trainers import TRAINER_REGISTRY
from domain_adaptation_ct.logging.log_mixin import init_logging
from domain_adaptation_ct.logging.epoch_csv_logging import CSVLoggingCallback

import torch
from transformers import Trainer, TrainingArguments

def train_model(
    trainer_cls: type[Trainer],
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    eval_dataset: torch.utils.data.Dataset,
    output_dir: str,
    logging_dir: str,
    num_train_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    optim: str,
    resume_from_checkpoint: bool,
    fold_num: int,
    lambda_scheduler: Optional[Callable[[int, int], float]] = None,
) -> str:
    """
    Wrapper for using one of the above trainers.
    """
    callbacks = []

    if lambda_scheduler is None:
        assert not hasattr(model, "grad_reverse"), "Do not specify a lambda scheduler if there is no `grad_reverse` layer."
        scheduler_name = "no_lambda_scheduler"
    else:
        assert hasattr(model, "grad_reverse"), "Must specify a lambda scheduler with the `grad_reverse` layer."
        callbacks.append(LambdaUpdateCallback(model, lambda_scheduler))
        scheduler_name = lambda_scheduler.__name__

    # Unique identifier for this run
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id_str = f"{scheduler_name}_fold_{fold_num}_{date_str}"

    # Decide where to put results for this training run.
    run_output_dir = os.path.join(output_dir, run_id_str)
    os.makedirs(run_output_dir)

    # Record each metric on each epoch in a tabular format
    training_curves_csv_save_path = os.path.join(run_output_dir, f"training_curves.csv")
    callbacks.append(CSVLoggingCallback(training_curves_csv_save_path))

    output_dir_results = os.path.join(run_output_dir, f"results")
    model_save_path = os.path.join(run_output_dir, f"final_model")

    logging.info(f"Will write training curves CSV to {training_curves_csv_save_path}.")
    logging.info(f"Will write results to {output_dir_results}.")
    logging.info(f"Will write model to {model_save_path}.")

    training_args = TrainingArguments(
        output_dir=output_dir_results,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        seed=42,
        optim=optim,
    )

    logging.info(training_args)

    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        compute_metrics=make_metrics_fn(model),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(model_save_path)

    # Return path to where all the results got saved.
    logging.info(f"Saved results to {run_output_dir}.")
    return run_output_dir


def run_training_from_config_file(config_file: str):
    """Top-level call if using a config file."""
    cfg = TrainingConfig(config_file).config_dict

    output_dir=cfg["output_dir"]

    init_logging(logging_dir=output_dir)

    # Instantiate the model.
    architecture_cls = ARCHITECTURE_REGISTRY[cfg["architecture"]["cls_name"]]
    model = architecture_cls(**cfg["architecture"]["cls_init_args"])
    logging.info(f"Instantiated model {architecture_cls.__name__}. Summary:\n{model}")

    # Load the dataset files.
    dataset_cls = DATASET_REGISTRY[cfg["training"]["dataset"]["cls_name"]]
    fold_file_paths = cfg["training"]["dataset"]["fold_file_paths"]
    fold_datasets = []
    for fold_file_path in fold_file_paths:
        fold_datasets.append(
            dataset_cls.load(
                file_path = fold_file_path,
                convert_grayscale_to_rgb = cfg["training"]["dataset"]["convert_grayscale_to_rgb"]
            )
        )

    # Perform k-fold cross validation training.
    # Does not include the final evaluation of the model.
    folds: list[dict[str, list[int]]] = cfg["training"]["dataset"]["folds"]
    for fold_num in range(len(folds)):
        logging.info(f"Beginning training of fold {fold_num} (#{fold_num+1} out of {len(folds)})")

        # Read which folds comprise the training dataset.
        train_folds = folds[fold_num]["train"]
        train_fold_datasets = [fold_datasets[train_fold] for train_fold in train_folds]
        # Give references to the appropriate datasets to the MultifoldDataset.
        train_dataset = MultifoldDataset(datasets = train_fold_datasets)
        logging.info(f"Instantiated train dataset {dataset_cls.__name__}, length {len(train_dataset)}, comprised of files {[fold_file_paths[train_fold] for train_fold in train_folds]}")

        # Read which folds comprise the validation dataset.
        val_folds = folds[fold_num]["val"]
        val_fold_datasets = [fold_datasets[val_fold] for val_fold in val_folds]
        eval_dataset = MultifoldDataset(datasets = val_fold_datasets)
        logging.info(f"Instantiated eval dataset {dataset_cls.__name__}, length {len(eval_dataset)}, comprised of files {[fold_file_paths[val_fold] for val_fold in val_folds]}")

        trainer_cls = TRAINER_REGISTRY[cfg["training"]["trainer"]["cls_name"]]

        lambda_scheduler = None
        if "lambda_scheduler" in cfg["training"]:
            lambda_scheduler = LAMBDA_SCHEDULER_REGISTRY[cfg["training"]["lambda_scheduler"]]
            logging.info(f"Got lambda scheduler {lambda_scheduler.__name__}")
        else:
            logging.info(f"No lambda scheduler provided.")

        run_output_dir = train_model(
            trainer_cls=trainer_cls,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            logging_dir=output_dir,
            num_train_epochs=cfg["training"]["training_arguments"]["num_train_epochs"],
            batch_size=cfg["training"]["training_arguments"]["batch_size"],
            learning_rate=cfg["training"]["training_arguments"]["learning_rate"],
            weight_decay=cfg["training"]["training_arguments"]["weight_decay"],
            optim=cfg["training"]["training_arguments"]["optim"],
            resume_from_checkpoint=cfg["training"]["training_arguments"]["resume_from_checkpoint"],
            fold_num=fold_num,
            lambda_scheduler=lambda_scheduler,
        )


