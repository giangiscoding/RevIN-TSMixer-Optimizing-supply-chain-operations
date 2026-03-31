import torch
import torch.nn as nn

class TSMixerLayer(nn.Module):
    def __init__(self, seq_len: int, num_features: int, ff_dim: int, out_features: int = None, dropout: float = 0.1):
        super(TSMixerLayer, self).__init__()
        self.out_features = out_features if out_features is not None else num_features
        
        # --- Time Mixing ---
        self.temporal_norm = nn.BatchNorm1d(seq_len)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- Feature Mixing ---
        self.feature_norm = nn.BatchNorm1d(num_features)
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, self.out_features),
            nn.Dropout(dropout)
        )
        
        # Xử lý Residual Connection nếu chiều feature bị thay đổi (F < C) theo Eq. 10
        if self.out_features != num_features:
            self.res_projection = nn.Linear(num_features, self.out_features)
        else:
            self.res_projection = nn.Identity()

    def forward(self, x: torch.Tensor):
        # Time-Mixing [B, T, C]
        res_time = x
        x_t = self.temporal_norm(x)
        x_t = x_t.transpose(1, 2)            # [B, C, T]
        x_t = self.temporal_mlp(x_t)         # [B, C, T]
        x_t = x_t.transpose(1, 2)            # [B, T, C]
        x = x_t + res_time
        
        # Feature-Mixing [B, T, C]
        res_feature = self.res_projection(x) # Chuyển đổi res nếu cần (Eq. 10)
        x_f = x.transpose(1, 2)              # [B, C, T]
        x_f = self.feature_norm(x_f)
        x_f = x_f.transpose(1, 2)            # [B, T, C]
        x_f = self.feature_mlp(x_f)          # [B, T, F]
        x = x_f + res_feature
        
        return x