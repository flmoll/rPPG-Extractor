import torch
import torch.nn as nn
from neural_methods.loss.PhysNetUncertaintyLoss import neg_log_likelihood_function

class HRClassifierUncertaintyLoss(nn.Module):
    """
    Loss function for heart rate prediction with uncertainty estimation.

    This loss combines:
        1. Negative log-likelihood (NLL) loss: encourages predicted uncertainty to match prediction errors.
        2. Mean Absolute Error (MAE): regularization term to encourage accurate heart rate predictions.

    Args:
        lambda_value (float): Weight for the MAE term (default: 100.0).

    Forward Inputs:
        hr_pred (torch.Tensor): Predicted heart rate, shape [batch_size].
        hr_uncertainty_pred (torch.Tensor): Predicted uncertainty of the heart rate, shape [batch_size].
        hr_labels (torch.Tensor): Ground truth heart rate, shape [batch_size].

    Returns:
        torch.Tensor: Scalar loss value combining NLL and weighted MAE.
    """

    def __init__(self, lambda_value=100.0):
        super(HRClassifierUncertaintyLoss, self).__init__()

        self.lambda_value = lambda_value

    def forward(self, hr_pred, hr_uncertainty_pred, hr_labels): 
        
        likelihood_loss = torch.mean(neg_log_likelihood_function(hr_pred, hr_uncertainty_pred, hr_labels))
        hr_loss = torch.mean(torch.abs(hr_pred - hr_labels))

        loss = likelihood_loss + self.lambda_value * hr_loss

        return loss