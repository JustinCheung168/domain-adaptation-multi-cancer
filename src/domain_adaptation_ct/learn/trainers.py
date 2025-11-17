from transformers import Trainer, TrainerCallback
import torch

from domain_adaptation_ct.logging.log_mixin import LogMixin

class BaselineTrainer(Trainer, LogMixin):
    """
    Trainer for ResNet50Baseline model.
    """

class DANNTrainer(Trainer, LogMixin):
    """
    Trainer for ResNet50DANN model.
    """

class GradientAssertionCallback(TrainerCallback):
    def on_backward_end(self, args, state, control, **kwargs):
        # kwargs contains 'model' and 'outputs'
        outputs = kwargs.get("outputs")
        model = kwargs.get("model")
        # Ensure outputs is BranchedOutput
        if outputs is not None and hasattr(outputs, "branch1_logits") and hasattr(outputs, "branch2_logits"):
            logits1 = outputs.branch1_logits
            logits2 = outputs.branch2_logits
            labels1 = kwargs.get("inputs").get("labels1")
            labels2 = kwargs.get("inputs").get("labels2")
            # Run your assertions
            zero_label_grad_indices, = torch.where(torch.sum(torch.abs(logits1.grad), dim=1) == 0.0)
            zero_domain_grad_indices, = torch.where(logits2.grad == 0.0)
            target_domain_indices, = torch.where(labels2 == 1)
            assert torch.equal(zero_label_grad_indices, target_domain_indices), \
                "Target domain instances should have zero gradient for label predictions."
            assert zero_domain_grad_indices.numel() == 0, \
                "Every instance should affect the domain classifier gradient."
 

# Allow selection of a trainer by its name as a string.
TRAINER_REGISTRY: dict[str, type[Trainer]] = {
    "BaselineTrainer": BaselineTrainer,
    "DANNTrainer": DANNTrainer,
}
