from transformers import Trainer

from domain_adaptation_ct.logging.log_mixin import LogMixin

class BaselineTrainer(Trainer, LogMixin):
    """
    Trainer for ResNet50Baseline model.
    """

class DANNTrainer(Trainer, LogMixin):
    """
    Trainer for ResNet50DANN model.
    """

# Allow selection of a trainer by its name as a string.
TRAINER_REGISTRY: dict[str, type[Trainer]] = {
    "BaselineTrainer": BaselineTrainer,
    "DANNTrainer": DANNTrainer,
}
