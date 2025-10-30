import torch

class GradientReversalFunction(torch.autograd.Function):
    """Adapted from https://github.com/tadeephuy/GradientReversal/tree/master"""
    @staticmethod
    def forward(ctx, x: torch.Tensor, lamb: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lamb, dtype=x.dtype, device=x.device))
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lamb_tensor, = ctx.saved_tensors
        return -lamb_tensor * grad_output, None


class GradientReversal(torch.nn.Module):
    """Adapted from https://github.com/tadeephuy/GradientReversal/tree/master"""
    def __init__(self, lamb: float):
        super().__init__()
        self.lamb = lamb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lamb)
