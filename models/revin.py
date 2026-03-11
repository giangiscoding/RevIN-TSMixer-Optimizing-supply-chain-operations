import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        Algorithm 2 & 3: Reversible Instance Normalization
        """
        super(RevIN, self).__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        
        self.mean = None
        self.stdev = None

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            
            x = (x - self.mean) / self.stdev
            
            if self.affine:
                x = x * self.gamma + self.beta
                
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            x = x * self.stdev + self.mean
            
        return x