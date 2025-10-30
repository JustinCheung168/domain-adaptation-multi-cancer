
def evaluate_model(
    evaluator_cls: type[Trainer],
    model: torch.nn.Module,
    eval_dataset: torch.utils.data.Dataset,
    output_dir: str,
    batch_size: int,
):
    evaluation_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        seed=42,
    )

    evaluator = evaluator_cls(
        model=model,
        args=evaluation_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=make_metrics_fn(model)
    )

    metrics = evaluator.evaluate()
    return metrics


def run_evaluation_from_config_file(config_file: str):
    """"""

