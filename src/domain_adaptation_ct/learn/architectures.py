from typing import Optional

from dataclasses import dataclass
import torch
from transformers import PreTrainedModel, ResNetConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

from domain_adaptation_ct.learn.gradient_reversal import GradientReversal
from domain_adaptation_ct.learn.loss import MaskedDomainAdversarialLoss

@dataclass
class BranchedOutput(ModelOutput):
    """Defines the model output structure"""
    loss: Optional[torch.FloatTensor] = None
    branch1_logits: Optional[torch.FloatTensor] = None
    branch2_logits: Optional[torch.FloatTensor] = None

class ResNet50Baseline(PreTrainedModel):
    """
    Baseline ResNet-50 model,
    with the same label predictor as the DANN variant for fairer comparison.
    """
    config_class = ResNetConfig

    def __init__(self, num_classes: int):
        """
        num_classes: Number of possible values for output label y.
        """
        config = ResNet50Baseline.config_class()
        super().__init__(config)
        # ResNet-50
        self.resnet = ResNetForImageClassification(config).resnet

        self.pre_branch = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.branch1 = torch.nn.Linear(512, num_classes)

        # Loss function used only by trainer
        self.loss_fn = torch.nn.CrossEntropyLoss()

        if type(self) is ResNet50Baseline:
            # Children of this class should not call this post_init.
            self.post_init()

    def forward(self, pixel_values: torch.Tensor, labels1: Optional[torch.Tensor] = None) -> BranchedOutput:
        features = self.resnet(pixel_values).pooler_output
        features = self.pre_branch(features)

        logits = self.branch1(features)

        loss = None
        if labels1 is not None:
            loss = self.loss_fn(logits, labels1)

        return BranchedOutput(
            loss = loss,
            branch1_logits = logits,
            branch2_logits = None,
        )

    @classmethod
    def load(cls, file_path: str):
        """In theory should be able to load from checkpoint or safetensors formats"""
        raise NotImplementedError(":)")

class ResNet50DANN(ResNet50Baseline):
    """
    Defines the DANN (domain adversarial neural network) with a ResNet-50 feature extractor.
    This is a branched model.
    """

    def __init__(self, num_classes: int, lamb_initial: float, ld_scale: float):
        """
        num_classes: Number of possible values for output label y.
        lamb_initial: Initial value for lambda hyperparameter for gradient reversal layer.
        """
        # Inherit from ResNet50Baseline
        super().__init__(num_classes)

        self.grad_reverse = GradientReversal(lamb=lamb_initial)
        
        self.branch2 = torch.nn.Sequential(
            self.grad_reverse,
            torch.nn.Linear(512, 1)
        )

        # Loss function used only by trainer.
        self.loss_fn = MaskedDomainAdversarialLoss()

        self.ld_scale = ld_scale

        self.post_init()

    def forward(self, pixel_values, labels1: Optional[torch.Tensor] = None, labels2: Optional[torch.Tensor] = None) -> BranchedOutput:
        # Feature extractor G_f
        features = self.resnet(pixel_values).pooler_output
        features = self.pre_branch(features)

        # Label predictor G_y (branch for original labels)
        logits1 = self.branch1(features)

        # Gradient reversal layer R_lambda and
        # Domain classifier G_d (branch for domain labels)
        logits2 = self.branch2(features)

        loss = None
        if (labels1 is not None) and (labels2 is not None):
            loss = self.loss_fn(logits1, logits2, labels1, labels2.view(-1, 1), self.ld_scale)

        return BranchedOutput(
            loss = loss,
            branch1_logits = logits1,
            branch2_logits = logits2,
        )


ARCHITECTURE_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "ResNet50Baseline": ResNet50Baseline,
    "ResNet50DANN": ResNet50DANN,
}
