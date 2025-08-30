

import torch
import torch.nn as nn
from neural_methods.loss.PhysNetUncertaintyLoss import neg_log_likelihood_function

class HRClassifierQuantileLoss(nn.Module):
    """
    Quantile loss for heart rate prediction with optional regularization using MAE.

    This loss combines:
        1. Quantile loss: penalizes under- and overestimation according to a specified quantile (alpha).
        2. Mean Absolute Error (MAE): regularization term to encourage accurate heart rate predictions.

    Args:
        alpha (float): Quantile level (default: 0.01). Determines asymmetry of quantile loss.
        lambda_value (float): Weight for the MAE term (default: 1.0).

    Forward Inputs:
        hr_lower (torch.Tensor): Predicted lower bound of heart rate, shape [batch_size].
        hr_upper (torch.Tensor): Predicted upper bound of heart rate, shape [batch_size].
        hr_labels (torch.Tensor): Ground truth heart rate, shape [batch_size].

    Returns:
        torch.Tensor: Scalar loss value combining quantile loss and weighted MAE.
    """
    
    def __init__(self, alpha=0.01, lambda_value=1.0):
        super(HRClassifierQuantileLoss, self).__init__()

        self.alpha = alpha
        self.lambda_value = lambda_value

    def forward(self, hr_lower, hr_upper, hr_labels):

        loss_lower = torch.max((self.alpha * (hr_labels - hr_lower)), ((self.alpha - 1) * (hr_labels - hr_lower)))
        loss_upper = torch.max((self.alpha * (hr_upper - hr_labels)), ((self.alpha - 1) * (hr_upper - hr_labels)))
        loss_quantile = torch.mean(loss_lower) + torch.mean(loss_upper)

        predictions = (hr_lower + hr_upper) / 2  # Average of lower and upper bounds
        loss_hr = torch.mean(torch.abs(predictions - hr_labels))  # Mean absolute error for heart rate

        loss = loss_quantile + self.lambda_value * loss_hr

        return loss