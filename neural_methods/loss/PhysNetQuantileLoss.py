from __future__ import print_function, division
import sys
import scipy
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn
from scipy.stats import norm
from torch.autograd import grad
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson


# Quantile regression loss
class PhysNetQuantileRegressionLoss(nn.Module):
    """
    Quantile regression loss for rPPG signal prediction combined with
    negative Pearson correlation.

    This loss encourages the predicted quantile bounds to:
        1. Capture the specified quantile of the true signal (via quantile loss).
        2. Match the temporal pattern of the ground truth signal (via negative Pearson correlation).

    Args:
        alpha (float): Quantile level for the lower/upper bounds (default: 0.01).
                       Typically, small alpha corresponds to a high-confidence lower bound.
        lambda_value (float): Weighting factor for the quantile loss relative to the Pearson loss (default: 1.0).

    Forward Inputs:
        preds_lower (torch.Tensor): Predicted lower bound of rPPG signal, shape [batch_size, signal_length].
        preds_upper (torch.Tensor): Predicted upper bound of rPPG signal, shape [batch_size, signal_length].
        labels (torch.Tensor): Ground truth rPPG signals, shape [batch_size, signal_length].

    Returns:
        torch.Tensor: Scalar loss value combining quantile regression loss and negative Pearson correlation.
    """
    
    def __init__(self, alpha=0.01, lambda_value=1.0):
        super(PhysNetQuantileRegressionLoss, self).__init__()
        self.alpha = alpha
        self.lambda_value = lambda_value
        self.pearson_loss = Neg_Pearson()

    def forward(self, preds_lower, preds_upper, labels): 
        loss_lower = torch.max((self.alpha * (labels - preds_lower)), ((self.alpha - 1) * (labels - preds_lower)))
        loss_upper = torch.max((self.alpha * (preds_upper - labels)), ((self.alpha - 1) * (preds_upper - labels)))
        loss_quantile = torch.mean(loss_lower) + torch.mean(loss_upper)

        loss_neg_pearson = self.pearson_loss((preds_lower + preds_upper) / 2, labels)
        loss = loss_neg_pearson + self.lambda_value * loss_quantile

        return loss



