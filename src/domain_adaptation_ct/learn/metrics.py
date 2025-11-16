import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from transformers.trainer_utils import EvalPrediction

def make_metrics_fn(model: torch.nn.Module):
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        outputs, labels = eval_pred

        is_branched = isinstance(outputs, tuple)
        
        if is_branched:
            logits1, logits2, loss1, loss2 = outputs
            labels1, labels2 = labels
            
            # Label prediction is multiclass prediction from logits
            preds1 = np.argmax(logits1, axis=-1)

            # Domain classification is binary prediction from logits
            preds2 = (logits2 > 0).astype(int).flatten()

            # Handle confusion matrix with explicit labels to handle single-class cases
            cm = confusion_matrix(y_true=labels2, y_pred=preds2, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel().tolist()
            
            metrics = {
                "accuracy_branch1": accuracy_score(labels1, preds1),
                "precision_branch1": precision_score(labels1, preds1, average="macro", zero_division=0),
                "recall_branch1": recall_score(labels1, preds1, average="macro", zero_division=0),
                "f1_branch1": f1_score(labels1, preds1, average="macro", zero_division=0),
                "accuracy_branch2": accuracy_score(labels2, preds2),
                "precision_branch2": precision_score(labels2, preds2, average="macro", zero_division=0),
                "recall_branch2": recall_score(labels2, preds2, average="macro", zero_division=0),
                "f1_branch2": f1_score(labels2, preds2, average="macro", zero_division=0),
                "lambda": model.grad_reverse.lamb if hasattr(model, 'grad_reverse') else None,
                "tn_branch2": int(tn),
                "fp_branch2": int(fp),
                "fn_branch2": int(fn),
                "tp_branch2": int(tp),
                "loss_branch1": float(loss1.mean()),
                "loss_branch2": float(loss2.mean()),
            }
        else:
            # Single branch case
            preds = np.argmax(outputs, axis=-1)
            metrics = {
                "accuracy_branch1": accuracy_score(labels, preds),
                "precision_branch1": precision_score(labels, preds, average="macro", zero_division=0),
                "recall_branch1": recall_score(labels, preds, average="macro", zero_division=0),
                "f1_branch1": f1_score(labels, preds, average="macro", zero_division=0)
            }
            
        return metrics
    return compute_metrics