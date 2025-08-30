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

def neg_log_likelihood_function(preds, uncertainty_pred, labels):
    return (torch.pow(preds - labels, 2) / (2 * torch.pow(uncertainty_pred, 2))) + torch.log(uncertainty_pred) + math.log(math.sqrt(2 * math.pi))

# neg log likelihood loss
class PhysNetNegLogLikelihoodLoss(nn.Module):
    """
    Combined loss for rPPG signal prediction that incorporates both
    negative Pearson correlation and Gaussian negative log-likelihood.

    The loss encourages the predicted rPPG signal to:
        1. Match the temporal pattern of the ground truth signal (via negative Pearson correlation).
        2. Reflect uncertainty in predictions through a Gaussian likelihood model.

    Args:
        sampling_rate (float): Sampling rate of the rPPG signal in Hz (default: 30).
        bandpass_upper (float): Upper cutoff frequency for expected heart rate in Hz (default: 2.5).

    Forward Inputs:
        preds (torch.Tensor): Predicted rPPG signals, shape [batch_size, signal_length].
        uncertainty_pred (torch.Tensor): Predicted uncertainty per sample, shape [batch_size, signal_length] or [batch_size].
        labels (torch.Tensor): Ground truth rPPG signals, shape [batch_size, signal_length].

    Returns:
        torch.Tensor: Scalar loss value combining negative Pearson correlation and uncertainty-weighted negative log-likelihood.
    """
    
    def __init__(self, sampling_rate=30, bandpass_upper=2.5):
        super(PhysNetNegLogLikelihoodLoss, self).__init__()
        self.sampling_rate = sampling_rate
        self.bandpass_upper = bandpass_upper
        self.lambda_value = 0.1
        self.neg_pearson_loss = Neg_Pearson()

    def forward(self, preds, uncertainty_pred, labels): 
        neg_log_likelihood = neg_log_likelihood_function(preds, uncertainty_pred, labels)
        neg_pearson_loss = self.neg_pearson_loss(preds, labels)

        loss = neg_pearson_loss + self.lambda_value * torch.mean(neg_log_likelihood)
        return loss



