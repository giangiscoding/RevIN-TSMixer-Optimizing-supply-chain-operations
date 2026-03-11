import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_sequences(data, seq_len, pred_len, target_idx=0):
    """ Tạo chuỗi dữ liệu đầu vào (Sliding Window) """
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pred_len, target_idx]) 
    return np.array(X), np.array(y)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

def train_model(model, train_loader, val_loader, epochs, lr, device='cpu'):
    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    model.to(device)
    best_val_mse = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output[:, :, 0], batch_y)
            loss.backward()
            optimizer.step()

        # Đánh giá trên Validation
        model.eval()
        val_mse, val_mae, val_mape = 0, 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)[:, :, 0]
                val_mse += criterion(v_out, vy).item()
                val_mae += mae_criterion(v_out, vy).item()
                val_mape += torch.mean(torch.abs((vy - v_out) / (vy + 1e-5))).item() * 100

        avg_mse = val_mse / len(val_loader)
        scheduler.step(avg_mse)

        if avg_mse < best_val_mse:
            best_val_mse = avg_mse
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch+1:03d} | MSE: {avg_mse:,.0f} | MAE: {val_mae/len(val_loader):,.0f} | MAPE: {val_mape/len(val_loader):.2f}%")

    model.load_state_dict(best_model_wts)
    return model