import torch
import torch.nn as nn

class TemporalProjectionLayer(nn.Module):
    def __init__(self, seq_len: int, pred_len: int):
        super(TemporalProjectionLayer, self).__init__()
        self.projection = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor):
        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        # Chiếu từ T sang N (pred_len) theo Eq. 12
        x = self.projection(x)
        # [B, F, N] -> [B, N, F] theo Eq. 13
        x = x.transpose(1, 2)
        return x