import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random
import scipy.stats as stats

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_sequences(data, seq_len, pred_len, target_idx=0):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len, target_idx]) 
    return np.array(X), np.array(y)

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(MAPELoss, self).__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.abs((target - pred) / (torch.abs(target) + self.eps))) * 100

def calculate_tc_min(preds, trues, h=2, L=2, o=50000):
    rmse = np.sqrt(np.mean((trues - preds)**2))
    mu_D = np.mean(preds)
    q_star = np.sqrt((2 * mu_D * o) / h)
    
    best_tc = float('inf')
    # ĐIỀU CHỈNH 1: Quét cs từ 10 đến 100 để ép alpha > 0.5
    for cs in np.linspace(10.0, 100.0, 100): 
        alpha = 1 - (h * q_star) / (cs * mu_D)
        # Đảm bảo alpha nằm trong vùng an toàn, không quá thấp
        alpha = np.clip(alpha, 0.5001, 0.9999) 
        
        z_alpha = stats.norm.ppf(alpha)
        # ĐIỀU CHỈNH 2: Cắt SS âm (Clip SS)
        ss = max(0, z_alpha * rmse * np.sqrt(L)) 
        
        lz = stats.norm.pdf(z_alpha) - z_alpha * (1 - stats.norm.cdf(z_alpha))
        expected_shortage = lz * rmse * np.sqrt(L)
        
        tc = ((cs * expected_shortage * mu_D) / q_star) + ((mu_D / q_star) * o) + ((q_star / 2 + ss) * h)
        if tc < best_tc:
            best_tc = tc
    return best_tc

def train_model(model, train_loader, val_loader, epochs, lr, device='cpu', scenario=1):
    criterion = MAPELoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # ĐIỀU CHỈNH 3: Tăng patience cho Scenario 2
    patience_limit = 15 if scenario == 1 else 35 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    model.to(device)
    best_val_score = float('inf') 
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output[:, :, 0], batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                v_out = model(vx.to(device))[:, :, 0]
                val_preds.append(v_out.cpu().numpy())
                val_trues.append(vy.numpy())

        val_preds = np.concatenate(val_preds).flatten()
        val_trues = np.concatenate(val_trues).flatten()
        
        avg_val_mape = np.mean(np.abs((val_trues - val_preds) / (val_trues + 1e-5))) * 100
        val_tc = calculate_tc_min(val_preds, val_trues)
        
        current_score = avg_val_mape if scenario == 1 else val_tc
        
        if current_score < best_val_score:
            best_val_score = current_score
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            
        scheduler.step(current_score)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Val MAPE: {avg_val_mape:.2f}% | Val TC: {val_tc:,.0f}")
        
        if no_improve >= patience_limit:
            print(f"Early Stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    return model