import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))            
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise ValueError("Mode chỉ có thể là 'norm' hoặc 'denorm'")
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        
        if self.affine:
            x = x * self.gamma + self.beta
            
        return x

    def _denormalize(self, x):      
        if self.affine:
            x = (x - self.beta) / self.gamma
        x = x * self.stdev + self.mean
        
        return x