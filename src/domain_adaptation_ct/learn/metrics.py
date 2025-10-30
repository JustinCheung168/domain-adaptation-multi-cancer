import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers.trainer_utils import EvalPrediction

def make_metrics_fn(model: torch.nn.Module):
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred

        is_branched = isinstance(logits, tuple)
        
        if is_branched:
            logits1, logits2 = logits
            labels1, labels2 = labels
            
            preds1 = np.argmax(logits1, axis=-1)
            preds2 = np.argmax(logits2, axis=-1)
            
            metrics = {
                "accuracy_branch1": accuracy_score(labels1, preds1),
                "precision_branch1": precision_score(labels1, preds1, average="macro", zero_division=0),
                "recall_branch1": recall_score(labels1, preds1, average="macro", zero_division=0),
                "f1_branch1": f1_score(labels1, preds1, average="macro", zero_division=0),
                "accuracy_branch2": accuracy_score(labels2, preds2),
                "precision_branch2": precision_score(labels2, preds2, average="macro", zero_division=0),
                "recall_branch2": recall_score(labels2, preds2, average="macro", zero_division=0),
                "f1_branch2": f1_score(labels2, preds2, average="macro", zero_division=0),
                "lambda": model.grad_reverse.lamb if hasattr(model, 'grad_reverse') else None
            }
        else:
            # Single branch case
            preds = np.argmax(logits, axis=-1)
            metrics = {
                "accuracy_branch1": accuracy_score(labels, preds),
                "precision_branch1": precision_score(labels, preds, average="macro", zero_division=0),
                "recall_branch1": recall_score(labels, preds, average="macro", zero_division=0),
                "f1_branch1": f1_score(labels, preds, average="macro", zero_division=0)
            }
            
        return metrics
    return compute_metrics