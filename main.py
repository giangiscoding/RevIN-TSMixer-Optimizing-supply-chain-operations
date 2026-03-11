import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.revin_tsmixer import RevINTSMixer
from models.inventory_model import ProbabilisticInventoryModel
from training.traning import set_seed, create_sequences, train_model

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data (Dữ liệu thô - Algorithm 4)
    df = pd.read_csv('data/data_TSI_v2.csv')
    important_features = ['Quantity', 'Imports', 'IPI', 'DisbursedFDI', 'CompetitorQuantity', 'PromotionAmount', 'RetailSales']
    selected_cols = ['Quantity'] + [c for c in important_features if c in df.columns and c != 'Quantity']
    data_df = df[selected_cols].interpolate().ffill().bfill()
    
    # 2. Setup
    SEQ_LEN, PRED_LEN, N_BLOCK, FF_DIM = 9, 3, 2, 128
    BATCH_SIZE, EPOCHS, LR, DROPOUT = 4, 30, 1e-4, 0.3

    X, y = create_sequences(data_df.values, SEQ_LEN, PRED_LEN, target_idx=0)
    train_idx, val_idx = int(len(X) * 0.8), int(len(X) * 0.9)

    # Chia dữ liệu
    train_loader = DataLoader(TensorDataset(torch.Tensor(X[:train_idx]), torch.Tensor(y[:train_idx])), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(X[train_idx:val_idx]), torch.Tensor(y[train_idx:val_idx])), batch_size=BATCH_SIZE)
    X_test_tensor = torch.Tensor(X[val_idx:]).to(device)
    y_test_tensor = torch.Tensor(y[val_idx:]).to(device)

    # 3. Train
    model = RevINTSMixer(SEQ_LEN, PRED_LEN, len(selected_cols), N_BLOCK, FF_DIM, DROPOUT)
    model = train_model(model, train_loader, val_loader, EPOCHS, LR, device)

    # 4. Đánh giá tập Test (Tính 4 chỉ số sai số)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)[:, :, 0] # Lấy dự báo thực tế (đã denorm)
        
        mse = torch.mean((test_preds - y_test_tensor)**2).item()
        mae = torch.mean(torch.abs(test_preds - y_test_tensor)).item()
        rmse = np.sqrt(mse)
        mape = torch.mean(torch.abs((y_test_tensor - test_preds) / (y_test_tensor + 1e-5))).item() * 100

    # 5. Tối ưu tồn kho (Tính TCmin)
    inv_model = ProbabilisticInventoryModel(holding_cost=2, ordering_cost=50000, shortage_cost=1.52, lead_time=2)
    # Dự báo cho 3 tháng tới dựa trên mẫu cuối cùng
    last_forecast = test_preds[-1].cpu().numpy() 
    metrics = inv_model.calculate_metrics(last_forecast, rmse)

    print("\n" + "="*80)
    print(f"{'Model':<20} | {'MSE':<15} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'TC_min':<10}")
    print("-"*80)
    print(f"{'RevIN-TSMixer':<20} | {mse:15,.0f} | {mae:10,.0f} | {rmse:10,.0f} | {mape:10.2f} | {metrics['Total Cost (TC)']:,.2f}")
    print("="*80)
    
    print("\nCHI TIẾT TỒN KHO:")
    print(f" • Safety Stock (SS): {metrics['Safety Stock (SS)']}")
    print(f" • Service Level (alpha): {metrics['Service Level (alpha)']}")
    print(f" • Reorder Point (r): {metrics['Reorder Point (r)']}")

if __name__ == "__main__":
    main()