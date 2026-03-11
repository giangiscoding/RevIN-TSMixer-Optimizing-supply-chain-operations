import torch.nn as nn
from .revin import RevIN
from .mixer_layers import TSMixerLayer


class RevINTSMixer(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, n_block, ff_dim, dropout):
        """
        Algorithm 4: RevIN-TSMixer Model
        """
        super(RevINTSMixer, self).__init__()
        
        self.revin = RevIN(num_features, affine=True)
        
        self.mixers = nn.ModuleList([
            TSMixerLayer(seq_len, num_features, ff_dim, dropout) 
            for _ in range(n_block)
        ])
        
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, mode='norm')
        
        for mixer in self.mixers:
            x = mixer(x)
            
        x = x.transpose(1, 2)
        x = self.projection(x)
        x = x.transpose(1, 2)  
        
        x = self.revin(x, mode='denorm')
        
        return x