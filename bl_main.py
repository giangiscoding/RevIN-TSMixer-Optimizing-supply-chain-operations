"""
run_baselines.py - Chạy tất cả baseline trên rolling-origin evaluation.

Pipeline chuẩn:
  - Rolling-origin, 3 folds, expanding window
  - Cùng seq_len, pred_len, cost parameters cho tất cả model
  - Hai scenario: S1 (min MAPE) và S2 (min Total Cost)
  - In bảng so sánh đầy đủ: MSE, MAE, RMSE, MAPE, TC, cs*

Baseline models:
  1. RevIN-TSMixer  (bài báo đề xuất)
  2. TSMixer        (không có RevIN)
  3. DLinear        (decomposition linear)
  4. NLinear        (normalized linear)
  5. N-BEATS        (neural basis expansion)
  6. N-HiTS         (hierarchical interpolation)

Chạy:
    python run_baselines.py
"""

import torch
import numpy as np
import pandas as pd

from train.trainer import set_seed
from models.revin_tsmixer import RevIN_TSMixer
from models.models import (
    TSMixer, DLinear, NLinear,
    NBEATSBaseline, NHiTSBaseline
)
from evaluation.rolling_origin import rolling_origin_evaluate

# ===========================================================================
# Cấu hình cố định – KHÔNG thay đổi khi so sánh baseline
# ===========================================================================
CONFIG = {
    # Data
    'csv_path'    : 'data/data_TSI_v2.csv',
    'features'    : ['Imports', 'IPI', 'DisbursedFDI',
                     'CompetitorQuantity', 'PromotionAmount', 'Quantity'],
    'target'      : 'Quantity',

    # Forecasting protocol (Section 3.2.1 bài báo)
    'seq_len'     : 9,       # lookback window
    'pred_len'    : 3,       # forecast horizon
    'epochs'      : 50,
    'batch_size'  : 4,
    'lr'          : 1e-4,

    # Rolling-origin protocol
    'n_folds'     : 3,
    'val_size'    : 11,      # sequences dùng làm val mỗi fold
    'seed'        : 42,

    # Inventory parameters (Section 4.8 bài báo – cố định cho tất cả model)
    'h'           : 2,       # holding cost per unit per month
    'L'           : 2,       # lead time (months)
    'o'           : 50000,   # ordering cost per order
    'cs_steps'    : 100,     # số điểm duyệt shortage cost
}


def build_model_registry(seq_len, pred_len, num_features):
    """
    Trả về dict: tên model → factory function (callable không tham số).
    Factory được gọi lại mỗi fold để tạo model mới từ đầu.
    """
    def revin_tsmixer():
        return RevIN_TSMixer(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            ff_dim=128, num_layers=2, dropout=0.1,
        )

    def tsmixer():
        return TSMixer(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            ff_dim=128, num_layers=2, dropout=0.1,
        )

    def dlinear():
        return DLinear(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            kernel_size=min(25, seq_len - 1) if seq_len > 2 else 1,
        )

    def nlinear():
        return NLinear(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
        )

    def nbeats():
        return NBEATSBaseline(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            units=64, n_blocks=3, n_layers=4,
        )

    def nhits():
        return NHiTSBaseline(
            seq_len=seq_len, pred_len=pred_len, num_features=num_features,
            units=64, pool_sizes=[1, 2, 4], n_layers=2,
        )

    return {
        'RevIN-TSMixer' : revin_tsmixer,
        'TSMixer'       : tsmixer,
        'DLinear'       : dlinear,
        'NLinear'       : nlinear,
        'N-BEATS'       : nbeats,
        'N-HiTS'        : nhits,
    }


def print_results_table(all_results: dict, scenario: str):
    """In bảng kết quả so sánh dạng markdown-style."""
    s_key  = 's1' if scenario == '1' else 's2'
    title  = "Scenario 1: Min MAPE" if scenario == '1' else "Scenario 2: Min Total Cost"

    print(f"\n{'='*80}")
    print(f"  {title}  (Rolling-Origin, {CONFIG['n_folds']} folds, mean)")
    print(f"{'='*80}")
    print(f"{'Model':<18} {'MSE':>14} {'MAE':>10} {'RMSE':>10} "
          f"{'MAPE%':>8} {'TC':>12} {'cs*':>6}")
    print(f"{'-'*18} {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*6}")

    # Sắp xếp theo MAPE cho S1, theo TC cho S2
    sort_key = 'MAPE' if scenario == '1' else 'TC'
    rows = []
    for name, result in all_results.items():
        m = result[s_key]['mean_metrics']
        rows.append((name, m))
    rows.sort(key=lambda r: r[1][sort_key])

    for name, m in rows:
        print(f"{name:<18} "
              f"{m['MSE']:>14,.0f} "
              f"{m['MAE']:>10,.0f} "
              f"{m['RMSE']:>10,.0f} "
              f"{m['MAPE']:>7.2f}% "
              f"{m['TC']:>12,.0f} "
              f"{m['cs*']:>6.2f}")

    print(f"{'='*80}")


def print_fold_detail(all_results: dict, model_name: str):
    """In chi tiết từng fold cho một model để debug."""
    result = all_results.get(model_name)
    if result is None:
        print(f"Model '{model_name}' không tìm thấy.")
        return

    print(f"\n--- Chi tiết từng fold: {model_name} ---")
    for s_key, s_label in [('s1', 'S1-MAPE'), ('s2', 'S2-Cost')]:
        print(f"\n  {s_label}:")
        for i, m in enumerate(result[s_key]['fold_metrics']):
            print(f"    Fold {i+1}: MAPE={m['MAPE']:.2f}%  "
                  f"TC={m['TC']:,.0f}  MAE={m['MAE']:,.0f}")


def analyze_error_source(all_results: dict):
    """
    Phân tích: sai số nằm ở forecast hay inventory mapping?

    Logic:
    - Nếu model có MAPE thấp nhưng TC cao → inventory mapping là bottleneck
    - Nếu model có MAPE cao nhưng TC thấp → forecast không cần quá chính xác
    - Gap giữa S1 và S2 cho thấy lợi ích của tích hợp inventory vào training
    """
    print(f"\n{'='*80}")
    print("  PHÂN TÍCH NGUỒN SAI SỐ: FORECAST vs INVENTORY MAPPING")
    print(f"{'='*80}")
    print(f"{'Model':<18} {'S1-MAPE':>8} {'S2-MAPE':>8} "
          f"{'S1-TC':>12} {'S2-TC':>12} "
          f"{'ΔTC(S1→S2)':>12} {'Lợi ích tích hợp':>18}")
    print(f"{'-'*18} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*18}")

    for name, result in all_results.items():
        s1 = result['s1']['mean_metrics']
        s2 = result['s2']['mean_metrics']
        delta_tc   = s1['TC'] - s2['TC']         # TC giảm bao nhiêu khi tích hợp
        delta_mape = s2['MAPE'] - s1['MAPE']     # MAPE tăng bao nhiêu khi ưu tiên TC

        # Đánh giá định tính
        if delta_tc > 0 and delta_mape > 0:
            note = "↑MAPE nhưng ↓TC → inventory win"
        elif delta_tc <= 0 and delta_mape <= 0:
            note = "S1 tốt hơn cả hai"
        elif delta_tc > 0 and delta_mape <= 0:
            note = "S2 tốt hơn cả hai"
        else:
            note = "S1 tốt hơn"

        print(f"{name:<18} "
              f"{s1['MAPE']:>7.2f}% "
              f"{s2['MAPE']:>7.2f}% "
              f"{s1['TC']:>12,.0f} "
              f"{s2['TC']:>12,.0f} "
              f"{delta_tc:>+12,.0f} "
              f"  {note}")

    print(f"\nGhi chú:")
    print(f"  ΔTC > 0 : Scenario 2 giảm được TC so với Scenario 1")
    print(f"  ΔTC < 0 : Tích hợp inventory vào training không giúp giảm TC")
    print(f"  Nếu ΔTC lớn → sai số chủ yếu ở inventory mapping, không phải forecast")
    print(f"  Nếu ΔTC nhỏ → forecast accuracy là bottleneck chính")
    print(f"{'='*80}")


def main():
    set_seed(CONFIG['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Config: seq_len={CONFIG['seq_len']}, pred_len={CONFIG['pred_len']}, "
          f"n_folds={CONFIG['n_folds']}, epochs={CONFIG['epochs']}")

    # ---- Load data ----
    df = pd.read_csv(CONFIG['csv_path'])
    df = df[CONFIG['features']]
    target_idx   = df.columns.get_loc(CONFIG['target'])
    raw_data     = df.values.astype(np.float32)
    num_features = raw_data.shape[1]

    print(f"Dataset: {len(raw_data)} điểm, {num_features} features, "
          f"target='{CONFIG['target']}' (idx={target_idx})")

    # ---- Build model registry ----
    registry = build_model_registry(
        seq_len=CONFIG['seq_len'],
        pred_len=CONFIG['pred_len'],
        num_features=num_features,
    )

    # ---- Chạy rolling-origin cho từng model ----
    all_results = {}
    for model_name, model_fn in registry.items():
        print(f"\n{'─'*60}")
        print(f"  Đang đánh giá: {model_name}")
        print(f"{'─'*60}")

        result = rolling_origin_evaluate(
            model_fn    = model_fn,
            raw_data    = raw_data,
            seq_len     = CONFIG['seq_len'],
            pred_len    = CONFIG['pred_len'],
            target_idx  = target_idx,
            lr          = CONFIG['lr'],
            epochs      = CONFIG['epochs'],
            device      = device,
            batch_size  = CONFIG['batch_size'],
            val_size    = CONFIG['val_size'],
            n_folds     = CONFIG['n_folds'],
            seed        = CONFIG['seed'],
            h           = CONFIG['h'],
            L           = CONFIG['L'],
            o           = CONFIG['o'],
            cs_steps    = CONFIG['cs_steps'],
        )
        all_results[model_name] = result

    # ---- In bảng kết quả ----
    print_results_table(all_results, scenario='1')
    print_results_table(all_results, scenario='2')

    # ---- Phân tích nguồn sai số ----
    analyze_error_source(all_results)

    # ---- Chi tiết từng fold cho RevIN-TSMixer ----
    print_fold_detail(all_results, 'RevIN-TSMixer')


if __name__ == '__main__':
    main()