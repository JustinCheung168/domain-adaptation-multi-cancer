import numpy as np
from transformers import TrainerCallback

class LambdaUpdateCallback(TrainerCallback):
    def __init__(self, model, lambda_scheduler):
        self.model = model
        self.lambda_scheduler = lambda_scheduler

    def on_epoch_begin(self, args, state, control, **kwargs):
        new_lambda = self.lambda_scheduler(state.epoch, args.num_train_epochs)
        if not hasattr(self.model, "grad_reverse"):
            raise Exception("Could not find `grad_reverse` in model. Lambda hyperparameter is not meaningful without `grad_reverse`.")
        self.model.grad_reverse.lamb = float(new_lambda)

def logistic_increasing_lambda_scheduler(epoch: int, total_epochs: int) -> float:
    """
    Based on Ganin, Y., & Lempitsky, V. (2015, June). Unsupervised domain adaptation by backpropagation.
    """
    progress = epoch / total_epochs
    lambda_p = 2. / (1. + np.exp(-10 * progress)) - 1
    return float(lambda_p)

def linear_increasing_lambda_scheduler(epoch: int, total_epochs: int) -> float:
    return epoch / total_epochs

def linear_decreasing_lambda_scheduler(epoch: int, total_epochs: int) -> float:
    return 1 - (epoch / total_epochs)

def constant_lambda_scheduler(epoch: int, total_epochs: int) -> float:
    return 0.5

def parabolic_increasing_lambda_scheduler(epoch: int, total_epochs: int) -> float:
    START_VALUE=0.0
    END_VALUE=1.0
    progress = epoch / total_epochs
    return START_VALUE + (END_VALUE - START_VALUE) * (progress ** 2)

def parabolic_decreasing_lambda_scheduler(epoch: int, total_epochs: int) -> float:
    START_VALUE=0.0
    END_VALUE=1.0
    progress = epoch / total_epochs
    return END_VALUE - (END_VALUE - START_VALUE) * (progress ** 2)

# Allow selection of a lambda scheduler by its name as a string.
LAMBDA_SCHEDULER_REGISTRY = {
    "logistic_increasing_lambda_scheduler": logistic_increasing_lambda_scheduler,
    "linear_increasing_lambda_scheduler": linear_increasing_lambda_scheduler,
    "linear_decreasing_lambda_scheduler": linear_decreasing_lambda_scheduler,
    "constant_lambda_scheduler": constant_lambda_scheduler,
    "parabolic_increasing_lambda_scheduler": parabolic_increasing_lambda_scheduler,
    "parabolic_decreasing_lambda_scheduler": parabolic_decreasing_lambda_scheduler,
}
