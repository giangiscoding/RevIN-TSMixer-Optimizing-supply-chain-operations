import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import itertools
from train.trainer import train_model, set_seed, create_sequences
from models.revin_tsmixer import RevIN_TSMixer

def get_dataloaders(seq_len, batch_size, real_data, pred_len, target_idx=0):
    X, y = create_sequences(real_data, seq_len, pred_len, target_idx=target_idx)
    train_end = int(len(X) * 0.8)
    val_end   = int(len(X) * 0.9)

    X_train = torch.tensor(X[:train_end], dtype=torch.float32)
    y_train = torch.tensor(y[:train_end], dtype=torch.float32)
    X_val = torch.tensor(X[train_end:val_end], dtype=torch.float32)
    y_val = torch.tensor(y[train_end:val_end], dtype=torch.float32)
    X_test = torch.tensor(X[val_end:], dtype=torch.float32)
    y_test = torch.tensor(y[val_end:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main():
    GLOBAL_SEED = 42
    set_seed(GLOBAL_SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang chạy trên thiết bị: {device}")

    csv_path = 'data/data_TSI_v2.csv'
    df = pd.read_csv(csv_path)

    selected_columns = ['Imports', 'IPI', 'DisbursedFDI', 'CompetitorQuantity', 'PromotionAmount', 'Quantity']
    df = df[selected_columns]

    target_idx = df.columns.get_loc('Quantity')

    raw_data = df.values.astype(np.float32)
    pred_len = 3 
    num_features = raw_data.shape[1]

    param_grid = {
        'seq_len': [6, 9],
        'n_block': [1, 2],
        'dropout': [0.1, 0.3],
        'batch_size': [2, 4],
        'ff_dim': [64, 128],
        'learning_rate': [1e-4, 1e-5],
    }

    keys, values  = zip(*param_grid.items())
    combinations  = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_s1_params, best_global_val_mape, best_global_test_mape = None, float('inf'), float('inf')
    best_s2_params, best_global_val_tc, best_global_test_tc = None, float('inf'), float('inf')

    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] Đang thử nghiệm: {params}")

        set_seed(GLOBAL_SEED)

        train_loader, val_loader, test_loader = get_dataloaders(
            seq_len=params['seq_len'],
            batch_size=params['batch_size'],
            real_data=raw_data,
            pred_len=pred_len,
            target_idx=target_idx,
        )

        model = RevIN_TSMixer(
            seq_len=params['seq_len'],
            pred_len=pred_len,
            num_features=num_features,
            ff_dim=params['ff_dim'],
            num_layers=params['n_block'],
            dropout=params['dropout'],
        )

        try:
            val_mape, test_mape, val_tc, test_tc = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=100,
                lr=params['learning_rate'],
                device=device,
                target_idx=target_idx,
                num_features=num_features,
            )

            print(f"  -> [S1 - MAPE] Val: {val_mape:.2f}% | Test: {test_mape:.2f}%")
            print(f"  -> [S2 - Cost] Val: {val_tc:,.0f} | Test: {test_tc:,.0f}")

            # Chọn config tốt nhất theo val metric (không dùng test để tránh data leakage)
            if val_mape < best_global_val_mape:
                best_global_val_mape = val_mape
                best_global_test_mape = test_mape
                best_s1_params = params

            if val_tc < best_global_val_tc:
                best_global_val_tc = val_tc
                best_global_test_tc = test_tc
                best_s2_params = params

        except Exception as e:
            print(f"  -> [LỖI] Cấu hình thất bại: {e}")
            continue

    # ================================================================
    # In kết quả cuối
    # ================================================================
    print("\n" + "=" * 55)
    print("KỊCH BẢN 1: TỐI ƯU HÓA SAI SỐ (MIN MAPE)")
    print(f"Cấu hình tối ưu : {best_s1_params}")
    print(f"Val MAPE : {best_global_val_mape:.2f}%")
    print(f"Test MAPE : {best_global_test_mape:.2f}%")
    print("-" * 55)
    print("KỊCH BẢN 2: TỐI ƯU HÓA CHI PHÍ (MIN TOTAL COST)")
    print(f"Cấu hình tối ưu : {best_s2_params}")
    print(f"Val Cost : {best_global_val_tc:,.0f}")
    print(f"Test Cost : {best_global_test_tc:,.0f}")
    print("=" * 55)


if __name__ == "__main__":
    main()