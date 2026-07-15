import torch
import torch.nn as nn

class NegativePearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Computes Negative Pearson Correlation Loss.
        preds: (B, T)
        targets: (B, T)
        """
        # Subtract mean along time dimension
        preds_mean = preds - torch.mean(preds, dim=-1, keepdim=True)
        targets_mean = targets - torch.mean(targets, dim=-1, keepdim=True)
        
        # Compute covariance
        cov = torch.sum(preds_mean * targets_mean, dim=-1)
        
        # Compute standard deviation
        preds_std = torch.sqrt(torch.sum(preds_mean ** 2, dim=-1) + 1e-8)
        targets_std = torch.sqrt(torch.sum(targets_mean ** 2, dim=-1) + 1e-8)
        
        # Pearson correlation
        pearson = cov / (preds_std * targets_std)
        
        # Loss is 1 - pearson (so 0 is perfect correlation, 2 is perfect anti-correlation)
        return 1.0 - torch.mean(pearson)

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Computes 1 - Concordance Correlation Coefficient.
        Optimizes for both correlation and exact value agreement.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        if preds.shape[0] < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        preds_mean = torch.mean(preds)
        targets_mean = torch.mean(targets)
        
        preds_var = torch.var(preds, unbiased=False)
        targets_var = torch.var(targets, unbiased=False)
        
        cov = torch.mean((preds - preds_mean) * (targets - targets_mean))
        
        ccc = (2 * cov) / (preds_var + targets_var + (preds_mean - targets_mean)**2 + 1e-8)
        
        return 1.0 - ccc
