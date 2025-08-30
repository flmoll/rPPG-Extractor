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


class PhysNetLifenessLoss(nn.Module):
    """
    Loss function for rPPG signal prediction with lifeness supervision.

    The loss combines:
        1. Negative Pearson correlation-based loss between predicted and true rPPG signals.
           - Weighted by (1 - shuffled_labels) to reduce contribution from shuffled samples.
        2. Binary cross-entropy (BCE) loss on the lifeness prediction.
           - Encourages the model to correctly identify valid signals vs. shuffled ones.
        3. Lambda weighting controls the contribution of lifeness loss to total loss.

    Args:
        lambda_value (float): Weight for the lifeness BCE loss (default: 0.1).

    Forward Inputs:
        preds (torch.Tensor): Predicted rPPG signals, shape [batch_size, signal_length].
        lifeness_pred (torch.Tensor): Predicted lifeness scores, shape [batch_size].
        labels (torch.Tensor): Ground truth rPPG signals, shape [batch_size, signal_length].
        shuffled_labels (torch.Tensor): Binary labels indicating shuffled samples (1 if shuffled, 0 if valid), shape [batch_size].

    Returns:
        torch.Tensor: Scalar loss value combining Pearson correlation loss and weighted lifeness BCE loss.
    """
    
    def __init__(self, lambda_value=0.1):
        super(PhysNetLifenessLoss, self).__init__()
        self.lambda_value = lambda_value
        self.lifeness_loss = nn.BCELoss()
        self.neg_pearson_loss = Neg_Pearson()

    def forward(self, preds, lifeness_pred, labels, shuffled_labels):   
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])             
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += (1 - pearson) * (1 - shuffled_labels[i])
            
        loss = loss/preds.shape[0]
        lifeness_loss = self.lifeness_loss(lifeness_pred, (1 - shuffled_labels))
        loss += self.lambda_value * lifeness_loss

        return loss
