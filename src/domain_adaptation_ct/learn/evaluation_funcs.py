import datetime
import os
import logging
from typing import Optional, Callable

from domain_adaptation_ct.config.experiment_config import EvaluationConfig
from domain_adaptation_ct.dataset.image_dataset import DATASET_REGISTRY
from domain_adaptation_ct.dataset.multifold_dataset import MultifoldDataset
from domain_adaptation_ct.learn.architectures import ARCHITECTURE_REGISTRY
from domain_adaptation_ct.learn.lambda_schedules import LAMBDA_SCHEDULER_REGISTRY, LambdaUpdateCallback
from domain_adaptation_ct.learn.metrics import make_metrics_fn
from domain_adaptation_ct.learn.trainers import TRAINER_REGISTRY
from domain_adaptation_ct.logging.log_mixin import init_logging
from domain_adaptation_ct.logging.epoch_csv_logging import EvaluationCSVLoggingCallback

import torch
from transformers import Trainer, TrainingArguments

def evaluate_model(
    evaluator_cls: type[Trainer],
    model: torch.nn.Module,
    eval_dataset: torch.utils.data.Dataset,
    output_dir: str,
    batch_size: int,
    fold_num: int,
) -> str:
    """
    """
    callbacks = []

    # Unique identifier for this run
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id_str = f"evaluation_{fold_num}_{date_str}"

    # Decide where to put results for this evaluation run.
    run_output_dir = os.path.join(output_dir, run_id_str)
    os.makedirs(run_output_dir)

    # Record the evaluation metrics in CSV files.
    # Each metric shall have its own file.
    # Each model shall have its own column.
    # Each domain in the test set shall have its own row.
    # This design is not permanent - needs to be adapted to our current format for test data
    test_metrics_csv_save_path = os.path.join(run_output_dir, f"test_metrics.csv")
    callbacks.append(EvaluationCSVLoggingCallback(test_metrics_csv_save_path))

    evaluation_args = TrainingArguments(
        per_device_eval_batch_size=batch_size,
        seed=42,
        dataloader_drop_last=False,
        report_to="none", # TODO test W&B integration
    )

    logging.info(evaluation_args)

    evaluator = evaluator_cls(
        model=model,
        args=evaluation_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model),
        callbacks=callbacks,
    )

    # Metrics are recorded to CSV via EvaluationCSVLoggingCallback.
    metrics = evaluator.evaluate()
    logging.info(f"Metrics: {metrics}")

    return run_output_dir

def run_evaluation_from_config_file(config_file: str):
    """"""
    cfg = EvaluationConfig(config_file).config_dict

    output_dir=cfg["output_dir"]

    init_logging(logging_dir=output_dir)

    # Instantiate the model.
    architecture_cls = ARCHITECTURE_REGISTRY[cfg["architecture"]["cls_name"]]
    model = architecture_cls.load(cfg["checkpoint_dir"], **cfg["architecture"]["cls_init_args"])
    logging.info(f"Instantiated model {architecture_cls.__name__}. Summary:\n{model}")

    # Load the dataset files.
    dataset_cls = DATASET_REGISTRY[cfg["evaluation"]["dataset"]["cls_name"]]
    fold_file_paths = cfg["evaluation"]["dataset"]["fold_file_paths"]
    assert isinstance(fold_file_paths, list)
    fold_datasets = []
    for fold_file_path in fold_file_paths:
        fold_datasets.append(
            dataset_cls.load(
                file_path = fold_file_path,
                convert_grayscale_to_rgb = cfg["evaluation"]["dataset"]["convert_grayscale_to_rgb"]
            )
        )

    # Assume we only look at one fold at a time. This design needs to be reworked to be less convoluted (TODO)
    folds: list[dict[str, list[int]]] = [{"val": [fold_num]} for fold_num in range(len(fold_file_paths))]
    for fold_num in range(len(folds)):
        logging.info(f"Beginning evaluation of fold {fold_num} (#{fold_num+1} out of {len(folds)})")

        # Read which folds comprise the validation dataset.
        val_folds = folds[fold_num]["val"]
        val_fold_datasets = [fold_datasets[val_fold] for val_fold in val_folds]
        eval_dataset = MultifoldDataset(datasets = val_fold_datasets)
        logging.info(f"Instantiated eval dataset {dataset_cls.__name__}, length {len(eval_dataset)}, comprised of files {[fold_file_paths[val_fold] for val_fold in val_folds]}")

        evaluator_cls = TRAINER_REGISTRY[cfg["evaluation"]["evaluator"]["cls_name"]]

        run_output_dir = evaluate_model(
            evaluator_cls=evaluator_cls,
            model=model,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            batch_size=cfg["evaluation"]["evaluation_arguments"]["batch_size"],
            fold_num=fold_num,
        )

    # TODO - add some logic to merge the resulting CSV's. may want this to just be a separate script which takes all the paths this sent the CSV's to.
