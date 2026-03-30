import torch
import torch.nn as nn
import math
from torch.distributions import Normal

class Inventory_model(nn.Module):
    def __init__(self, h=2, L=2, o=50000, cs_steps=100):
        super(Inventory_model, self).__init__()
        self.h = h
        self.L = L
        self.o = o
        self.cs = torch.linspace(0.01, 10.0, cs_steps).view(-1, 1) # Giữ chuẩn 0.1 đến 10
        self.normal = Normal(0.0, 1.0)

    def forward(self, preds, trues):
            # 1. Tính Phần dư (Residuals) và Độ lệch chuẩn của chúng (sigma_e)
            residuals = trues - preds
            
            # unbiased=False để tính độ lệch chuẩn theo quần thể (giống cách tính RMSE)
            # Cộng thêm 1e-5 để tránh lỗi chia cho 0 hoặc log(0) trong quá trình tính toán
            sigma_d = torch.std(residuals, dim=1, unbiased=False) + 1e-5
            sigma_d = sigma_d.view(1, -1) 
            
            mu_D = torch.mean(preds, dim=1)
            mu_D = torch.clamp(mu_D, min=1e-4).view(1, -1)
            cs_device = self.cs.to(preds.device)
            
            q_star = torch.sqrt((2 * mu_D * self.o) / self.h)
            
            alpha_raw = 1.0 - (self.h * q_star) / (cs_device * mu_D)
            
            valid_mask = alpha_raw > 0.001
            
            alpha = torch.clamp(alpha_raw, min=1e-4, max=0.9999) 
            
            z_alpha = self.normal.icdf(alpha)
            
            # 2. Áp dụng sigma_e vào công thức tính Tồn kho an toàn (SS)
            ss = z_alpha * sigma_d * math.sqrt(self.L)
            
            phi_z = torch.exp(-0.5 * z_alpha**2) / math.sqrt(2 * math.pi)
            PHI_z = self.normal.cdf(z_alpha)
            lz = phi_z - z_alpha * (1.0 - PHI_z)
            
            # 3. Áp dụng sigma_e vào công thức Kỳ vọng thiếu hụt (E_s)
            e_s = lz * sigma_d * math.sqrt(self.L)
            
            E_cs = (cs_device * e_s * mu_D) / q_star
            c_o = (mu_D / q_star) * self.o
            c_h = (q_star / 2.0 + torch.relu(ss)) * self.h
            
            tc = c_o + c_h + E_cs
            
            tc = torch.where(valid_mask, tc, torch.tensor(float('inf'), device=tc.device))
            
            best_tc, best_idx = torch.min(tc, dim=0)
            best_cs_vals = cs_device[best_idx].squeeze()
            
            return torch.mean(best_tc), torch.mean(best_cs_vals).item()