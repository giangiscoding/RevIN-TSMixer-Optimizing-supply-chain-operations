import torch.nn as nn


class TSMixerLayer(nn.Module):
    def __init__(self, seq_len, num_features, ff_dim, dropout=0.1):
        super(TSMixerLayer, self).__init__()
        # Time Mixing Layer: Hoạt động trên chiều T
        self.temporal_norm = nn.BatchNorm1d(num_features)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        # Feature Mixing Layer: Hoạt động trên chiều C
        self.feature_norm = nn.BatchNorm1d(seq_len)
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, num_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 1. Time Mixing
        res = x
        x = x.transpose(1, 2) # [B, C, T]
        x = self.temporal_mlp(x)
        x = x.transpose(1, 2) # [B, T, C]
        x = x + res # Residual connection
        
        # 2. Feature Mixing
        res = x
        x = self.feature_mlp(x)
        x = x + res # Residual connection
        
        return x