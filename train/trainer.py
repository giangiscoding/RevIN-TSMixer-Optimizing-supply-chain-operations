import torch
import torch.nn as nn
import numpy as np
import copy
import random
import math
from models.inventory_model import Inventory_model

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

def train_model(model, train_loader, val_loader, test_loader,
                epochs, lr, device, target_idx, num_features,
                h=2, L=2, o=50000, cs_steps=100):
    
    # --- 1. TÍNH TOÁN THÔNG SỐ CHUẨN HÓA CỐ ĐỊNH ---
    # Ta cần một scale chuẩn để ép MSE về khoảng nhỏ, giúp gradient ổn định.
    all_y = []
    for _, batch_y in train_loader:
        all_y.append(batch_y)
    all_y = torch.cat(all_y, dim=0)
    
    # Tính mean và std của target trên tập train
    t_mean = all_y.mean().to(device)
    t_std  = (all_y.std() + 1e-6).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    patience_limit = 15
    model.to(device)

    tc_calculate = Inventory_model(h=h, L=L, o=o, cs_steps=cs_steps).to(device)

    best_val_mape = float('inf')
    best_wts_s1   = copy.deepcopy(model.state_dict())
    best_val_tc   = float('inf')
    best_wts_s2   = copy.deepcopy(model.state_dict())
    
    best_val_loss = float('inf')
    no_improve_val = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Output này đã được RevIN denormalize về scale gốc (giá trị lớn)
            output = model(batch_x)[:, :, target_idx]
            
            # --- CHUẨN HÓA LOSS TẠI ĐÂY ---
            # Ép giá trị về scale chuẩn (Standard Score) trước khi tính MSE
            output_norm = (output - t_mean) / t_std
            y_norm      = (batch_y - t_mean) / t_std
            
            loss = criterion(output_norm, y_norm)
            
            optimizer.zero_grad()
            loss.backward()
            # Clip gradient để tránh bùng nổ trọng số
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss_std = total_train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_preds, val_trues = [], []
        total_val_loss_std = 0
        
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)[:, :, target_idx]
                
                # Tính Val Loss theo cùng thước đo chuẩn hóa với Train
                v_out_norm = (v_out - t_mean) / t_std
                vy_norm    = (vy - t_mean) / t_std
                v_loss     = criterion(v_out_norm, vy_norm)
                
                total_val_loss_std += v_loss.item()
                val_preds.append(v_out)
                val_trues.append(vy)

        avg_val_loss_std = total_val_loss_std / len(val_loader)
        val_preds_tensor = torch.clamp(torch.cat(val_preds, dim=0), min=0.0)
        val_trues_tensor = torch.cat(val_trues, dim=0)

        # Đánh giá Metrics thực tế (MAPE & Total Cost dùng giá trị gốc)
        val_mape = torch.mean(torch.abs((val_trues_tensor.flatten() - val_preds_tensor.flatten()) / 
                   torch.clamp(val_trues_tensor.flatten(), min=1e-5))).item() * 100
        
        val_tc_tensor, _ = tc_calculate(val_preds_tensor, val_trues_tensor)
        val_tc = val_tc_tensor.item()

        # In log chi tiết để theo dõi hội tụ
        print(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss(Std): {avg_train_loss_std:.6f} | "
              f"Val Loss(Std): {avg_val_loss_std:.6f} | Val MAPE: {val_mape:.2f}%")

        # Lưu weight tốt nhất theo từng Scenario
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_wts_s1 = copy.deepcopy(model.state_dict())

        if val_tc < best_val_tc:
            best_val_tc = val_tc
            best_wts_s2 = copy.deepcopy(model.state_dict())

        # Early Stopping dựa trên Standardized Validation Loss
        if avg_val_loss_std < best_val_loss:
            best_val_loss = avg_val_loss_std
            no_improve_val = 0
        else:
            no_improve_val += 1

        if no_improve_val >= patience_limit:
            print(f"      -> Early stopping: Val Loss không cải thiện sau {patience_limit} epoch.")
            break

    # ================================================================
    # ĐÁNH GIÁ TẬP TEST (Sử dụng bộ trọng số lưu trữ)
    # ================================================================
    test_trues = []
    with torch.no_grad():
        for _, ty in test_loader:
            test_trues.append(ty.to(device))
    test_trues_tensor = torch.cat(test_trues, dim=0)

    # Scenario 1: Best MAPE
    model.load_state_dict(best_wts_s1)
    model.eval()
    test_preds_s1 = []
    with torch.no_grad():
        for tx, _ in test_loader:
            test_preds_s1.append(model(tx.to(device))[:, :, target_idx])
    p1_tensor = torch.clamp(torch.cat(test_preds_s1, dim=0), min=0.0)
    test_mape_s1 = torch.mean(torch.abs((test_trues_tensor.flatten() - p1_tensor.flatten()) / 
                   torch.clamp(test_trues_tensor.flatten(), min=1e-5))).item() * 100

    # Scenario 2: Best Total Cost
    model.load_state_dict(best_wts_s2)
    model.eval()
    test_preds_s2 = []
    with torch.no_grad():
        for tx, _ in test_loader:
            test_preds_s2.append(model(tx.to(device))[:, :, target_idx])
    p2_tensor = torch.clamp(torch.cat(test_preds_s2, dim=0), min=0.0)
    test_tc_tensor_s2, _ = tc_calculate(p2_tensor, test_trues_tensor)
    test_tc_s2 = test_tc_tensor_s2.item()

    return best_val_mape, test_mape_s1, best_val_tc, test_tc_s2