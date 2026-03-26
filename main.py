import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.revin_tsmixer import RevINTSMixer
from models.inventory_model import ProbabilisticInventoryModel
from training.traning import set_seed, create_sequences, train_model

def run_experiment(scenario, data_df, selected_cols, device):
    set_seed(42)
    
    # ĐIỀU CHỈNH 3: Tăng Epochs lên 300 cho Scenario 2
    if scenario == 1:
        SEQ_LEN, PRED_LEN, N_BLOCK, FF_DIM = 6, 3, 1, 64
        BATCH_SIZE, EPOCHS, LR, DROPOUT = 4, 150, 1e-4, 0.1
    else:
        # Scenario 2 cần học sâu hơn và lâu hơn
        SEQ_LEN, PRED_LEN, N_BLOCK, FF_DIM = 9, 3, 2, 128
        BATCH_SIZE, EPOCHS, LR, DROPOUT = 2, 60, 5e-5, 0.2 

    X, y = create_sequences(data_df.values, SEQ_LEN, PRED_LEN, target_idx=0)
    train_idx, val_idx = int(len(X) * 0.8), int(len(X) * 0.9)

    train_loader = DataLoader(TensorDataset(torch.Tensor(X[:train_idx]), torch.Tensor(y[:train_idx])), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(X[train_idx:val_idx]), torch.Tensor(y[train_idx:val_idx])), batch_size=BATCH_SIZE)
    
    X_test_tensor = torch.Tensor(X[val_idx:]).to(device)
    y_test_tensor = torch.Tensor(y[val_idx:]).to(device)

    # ĐIỀU CHỈNH 4: RevIN đã được tích hợp sẵn trong lớp RevINTSMixer. 
    # Nếu kết quả vẫn bẹt, hãy kiểm tra file revin_tsmixer.py xem lớp RevIN có bị comment hay không.
    model = RevINTSMixer(SEQ_LEN, PRED_LEN, len(selected_cols), N_BLOCK, FF_DIM, DROPOUT)
    model = train_model(model, train_loader, val_loader, EPOCHS, LR, device, scenario=scenario)

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)[:, :, 0]
        mse = torch.mean((test_preds - y_test_tensor)**2).item()
        rmse = np.sqrt(mse)
        mape = torch.mean(torch.abs((y_test_tensor - test_preds) / (y_test_tensor + 1e-5))).item() * 100

    # Đánh giá cuối cùng với cs = 80 (để đảm bảo SS luôn dương và có ý nghĩa kinh tế)
    inv_model = ProbabilisticInventoryModel(holding_cost=2, ordering_cost=50000, shortage_cost=1.54, lead_time=2)
    last_forecast = test_preds[-1].cpu().numpy() 
    metrics = inv_model.calculate_metrics(forecasted_demands=last_forecast, demand_std=rmse)

    return {"MAPE": mape, "RMSE": rmse, "TC": metrics['Total Cost (TC)'], "SS": metrics['Safety Stock (SS)']}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('data/data_TSI_v2.csv')
    
    # Theo bài báo: lấy đúng 6 features quan trọng nhất
    important_features = ['Quantity', 'Imports', 'IPI', 'DisbursedFDI', 'CompetitorQuantity', 'PromotionAmount']
    data_df = df[important_features].interpolate().ffill().bfill()
    
    res1 = run_experiment(1, data_df, important_features, device)
    res2 = run_experiment(2, data_df, important_features, device)

    print("\n" + "="*90)
    print(f"{'Kịch bản (Scenario)':<25} | {'MAPE (%)':<10} | {'RMSE':<12} | {'TC_min':<15} | {'SS':<10}")
    print("-" * 90)
    print(f"{'1: Tối ưu Sai số (MAPE)':<25} | {res1['MAPE']:10.2f} | {res1['RMSE']:12,.0f} | {res1['TC']:15,.2f} | {res1['SS']:10.1f}")
    print(f"{'2: Tối ưu Chi phí (TC)':<25} | {res2['MAPE']:10.2f} | {res2['RMSE']:12,.0f} | {res2['TC']:15,.2f} | {res2['SS']:10.1f}")
    print("="*90)

if __name__ == "__main__":
    main()