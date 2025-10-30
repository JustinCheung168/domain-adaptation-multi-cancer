import torch

class InstanceWeightedCrossEntropyLoss(torch.nn.Module):
    """
    Minibatch loss function where the contribution of each instance to the overall loss can be given a weight.
    """
    def __init__(self):
        super().__init__()
        self.base_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, instance_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Let N represent batch size.
        Let K represent number of possible output classes.

        Args:
            logits: Shape [N, K], dtype float32. Model's predicted logits for classification.
            labels: Shape [N], dtype int64. Ground truth class indices.
            instance_weights: Shape [N], dtype float32. Weights for each instance in the minibatch.
        Returns:
            Shape []. Loss value.
        """
        loss = self.base_loss(logits, labels)
        masked_loss = loss * instance_weights
        reduced_loss = masked_loss.mean()
        return reduced_loss


class MaskedDomainAdversarialLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Branch 1 is trained as a multi-class classifier which ignores loss contributions by providing a weight of 0 to instances whose domain is 1.
        self.branch1_loss_fn = InstanceWeightedCrossEntropyLoss()

        # Branch 2 is trained as a standard binary classifier.
        self.branch2_loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits1: torch.Tensor, logits2: torch.Tensor, labels1: torch.Tensor, labels2: torch.Tensor, ld_scale: float) -> torch.Tensor:
        """
        Compute the loss.

        Let N represent batch size.
        Let K represent number of possible output classes.

        Args:
            logits1: Shape [N, K]. Model predicted logits for label y.
            logits2: Shape [N]. Model predicted logits for domain d.
            labels1: Shape [N]. Ground truth label y.
            labels2: Shape [N]. Ground truth domain d. Assume d=0 is source domain, d=1 is target domain.
            ld_scale: Scale factor for the domain classification loss relative to the label prediction loss.
        Returns:
            Shape []. Loss value.
        """
        # Mask contributions of target domain.
        domain_mask = 1.0 - labels2

        loss1 = self.branch1_loss_fn(logits = logits1, labels = labels1, instance_weights = domain_mask)
        loss2 = self.branch2_loss_fn(input = logits2, target = labels2.float())

        total_loss = loss1 + (ld_scale * loss2)

        return total_loss
