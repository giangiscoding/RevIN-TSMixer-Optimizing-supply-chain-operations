import torch
import torch.nn as nn
from .revin import RevIN
from .mixer_layers import TSMixerLayer
from .temporalprojectionlayer import TemporalProjectionLayer

class RevIN_TSMixer(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, ff_dim, num_layers, dropout=0.1):
        super(RevIN_TSMixer, self).__init__()
        self.revin = RevIN(num_features)
        self.mixer_layers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.projection = TemporalProjectionLayer(seq_len, pred_len)

    def forward(self, x):
            x = self.revin(x, mode='norm')
            for layer in self.mixer_layers:
                x = layer(x)
            x = self.projection(x)
            x = self.revin(x, mode='denorm')
            
            return x