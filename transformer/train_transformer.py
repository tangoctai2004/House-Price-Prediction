"""Train a Transformer model for Vietnamese house-price prediction.

Run from project root:
    python transformer/train_transformer.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from transformer_model import HousePriceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ánh xạ Quận/Huyện về Tỉnh/Thành phố tương ứng
DISTRICT_TO_PROVINCE = {
    # Hà Nội
    "Cầu Giấy": "Hà Nội", "Nam Từ Liêm": "Hà Nội", "Bắc Từ Liêm": "Hà Nội", 
    "Tây Hồ": "Hà Nội", "Thanh Xuân": "Hà Nội", "Hà Đông": "Hà Nội", 
    "Đống Đa": "Hà Nội", "Hoàng Mai": "Hà Nội", "Long Biên": "Hà Nội", 
    "Ba Đình": "Hà Nội", "Hai Bà Trưng": "Hà Nội", "Gia Lâm": "Hà Nội", 
    "Đông Anh": "Hà Nội", "Thanh Trì": "Hà Nội", "Hoài Đức": "Hà Nội",
    
    # TP. Hồ Chí Minh
    "Quận 1": "TP. Hồ Chí Minh", "Quận 2": "TP. Hồ Chí Minh", "Quận 3": "TP. Hồ Chí Minh", 
    "Quận 4": "TP. Hồ Chí Minh", "Quận 5": "TP. Hồ Chí Minh", "Quận 6": "TP. Hồ Chí Minh", 
    "Quận 7": "TP. Hồ Chí Minh", "Quận 8": "TP. Hồ Chí Minh", "Quận 9": "TP. Hồ Chí Minh", 
    "Quận 10": "TP. Hồ Chí Minh", "Quận 11": "TP. Hồ Chí Minh", "Quận 12": "TP. Hồ Chí Minh", 
    "Bình Tân": "TP. Hồ Chí Minh", "Bình Thạnh": "TP. Hồ Chí Minh", "Tân Phú": "TP. Hồ Chí Minh", 
    "Bình Chánh": "TP. Hồ Chí Minh", "Thủ Đức": "TP. Hồ Chí Minh", "Tân Bình": "TP. Hồ Chí Minh", 
    "Phú Nhuận": "TP. Hồ Chí Minh", "Nhà Bè": "TP. Hồ Chí Minh", "Hóc Môn": "TP. Hồ Chí Minh", 
    "Gò Vấp": "TP. Hồ Chí Minh",
    
    # Đà Nẵng
    "Ngũ Hành Sơn": "Đà Nẵng", "Sơn Trà": "Đà Nẵng", "Cẩm Lệ": "Đà Nẵng", 
    "Liên Chiểu": "Đà Nẵng", "Hải Châu": "Đà Nẵng"
}

# BƯỚC 3: Chọn các cột đầu vào (Features) và cột cần dự đoán (Target - giá nhà)
FEATURES = [
    "area_m2",          # diện tích
    "bedrooms_num",     # số phòng ngủ
    "district",         # quận
    "tinh_thanh",       # tỉnh/thành phố
    "direction",        # hướng
    "furniture_std",    # nội thất
    "legal_std",        # pháp lý
    "floors_num",       # số tầng
    "frontage_m",       # mặt tiền
    "road_width_m",     # đường rộng
    "loai_bds",         # loại bất động sản
]
TARGET = "price_billion" # Giá nhà (tỷ VNĐ)
NUMERIC_FEATURES = ["area_m2", "bedrooms_num", "floors_num", "frontage_m", "road_width_m"]
CATEGORICAL_FEATURES = ["district", "tinh_thanh", "direction", "furniture_std", "legal_std", "loai_bds"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_clean_data() -> pd.DataFrame:
    # BƯỚC 1: Dùng lại 2 file CSV cũ (chung cư và nhà đất), thêm cột loại bất động sản rồi gộp thành một bộ dữ liệu chung.
    cc_path = PROJECT_ROOT / "data" / "processed" / "cleaned_chung_cu.csv"
    nd_path = PROJECT_ROOT / "data" / "processed" / "cleaned_nha_dat.csv"
    df_cc = pd.read_csv(cc_path)
    df_nd = pd.read_csv(nd_path)

    df_cc["loai_bds"] = "chung_cu"
    df_nd["loai_bds"] = "nha_dat"
    if "balcony_direction" in df_cc.columns:
        df_cc = df_cc.drop(columns=["balcony_direction"])

    for col in ["floors_num", "frontage_m", "road_width_m"]:
        df_cc[col] = 0

    df_cc["tinh_thanh"] = df_cc["district"].map(DISTRICT_TO_PROVINCE).fillna("Khác")
    df_nd["tinh_thanh"] = df_nd["district"].map(DISTRICT_TO_PROVINCE).fillna("Khác")

    all_cols = [TARGET] + FEATURES
    df = pd.concat([df_cc[all_cols], df_nd[all_cols]], ignore_index=True)

    # BƯỚC 2: Dọn dữ liệu lỗi hoặc quá bất thường, ví dụ giá quá cao/thấp hoặc diện tích không hợp lý.
    df = df.dropna(subset=[TARGET, "area_m2"])
    df = df[(df[TARGET] >= 1) & (df[TARGET] <= 200)] # Lọc giá bất thường: chỉ lấy từ 1 đến 200 tỷ
    df = df[(df["area_m2"] >= 10) & (df["area_m2"] <= 1000)] # Lọc diện tích không hợp lý: từ 10m2 đến 1000m2

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Không rõ").astype(str)
    return df.reset_index(drop=True)


def make_category_mappings(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    # BƯỚC 5 (Phần 1): Đổi các cột chữ thành mã số nguyên (mapping)
    mappings: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_FEATURES:
        values = sorted(train_df[col].astype(str).unique())
        mappings[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return mappings


def encode_categories(df: pd.DataFrame, mappings: dict[str, dict[str, int]]) -> np.ndarray:
    encoded = []
    for col in CATEGORICAL_FEATURES:
        mapping = mappings[col]
        encoded.append(df[col].astype(str).map(mapping).fillna(0).astype("int64").to_numpy())
    return np.stack(encoded, axis=1)


def make_loader(
    numeric_array: np.ndarray,
    categorical_array: np.ndarray,
    target_array: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(numeric_array, dtype=torch.float32),
        torch.tensor(categorical_array, dtype=torch.long),
        torch.tensor(target_array, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds = []
    actuals = []
    with torch.no_grad():
        for numeric_values, categorical_ids, y in loader:
            numeric_values = numeric_values.to(device)
            categorical_ids = categorical_ids.to(device)
            y = y.to(device)
            output = model(numeric_values, categorical_ids)
            loss = criterion(output, y)
            losses.append(float(loss.item()) * len(y))
            preds.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    return sum(losses) / len(y_true), y_pred, y_true


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def save_plots(history: list[dict[str, float]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_mse"] for item in history]
    val_loss = [item["val_mse"] for item in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="Train MSE")
    plt.plot(epochs, val_loss, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Transformer Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.35, s=14)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color="red", linewidth=1)
    plt.xlabel("Giá thật (tỷ VNĐ)")
    plt.ylabel("Giá dự đoán (tỷ VNĐ)")
    plt.title("Transformer: Giá thật vs Giá dự đoán")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prediction_scatter.png", dpi=150)
    plt.close()

    errors = y_pred - y_true
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=40, edgecolor="black")
    plt.xlabel("Sai số dự đoán (tỷ VNĐ)")
    plt.ylabel("Số lượng")
    plt.title("Transformer Error Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_distribution.png", dpi=150)
    plt.close()


def save_xgboost_comparison(transformer_metrics: dict[str, float]) -> None:
    meta_path = PROJECT_ROOT / "models" / "model_meta.pkl"
    if not meta_path.exists():
        return
    with open(meta_path, "rb") as f:
        xgb_meta = pickle.load(f)
    xgb_metrics = xgb_meta.get("metrics", {})
    rows = [
        {"model": xgb_meta.get("best_model_name", "XGBoost"), **xgb_metrics},
        {"model": "Transformer", **transformer_metrics},
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison_with_xgboost.csv", index=False)

    labels = [row["model"] for row in rows]
    metrics = ["R2", "MAE", "RMSE", "MAPE"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        values = [row.get(metric, 0) for row in rows]
        ax.bar(labels, values, color=["#2f80ed", "#27ae60"], edgecolor="black")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transformer_vs_xgboost.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df = load_clean_data()

    # BƯỚC 4: Chia dữ liệu thành 3 phần: train (70%), validation (15%), test (15%)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=args.seed)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=args.seed)

    # BƯỚC 6: Với cột số: chuẩn hóa về cùng thang đo bằng công thức (x - mean) / std sử dụng StandardScaler
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[NUMERIC_FEATURES])
    val_num = scaler.transform(val_df[NUMERIC_FEATURES])
    test_num = scaler.transform(test_df[NUMERIC_FEATURES])

    # BƯỚC 5 (Phần 2): Áp dụng mã số đã map vào các tập dữ liệu chữ để chuẩn bị đưa qua Embedding
    mappings = make_category_mappings(train_df)
    train_cat = encode_categories(train_df, mappings)
    val_cat = encode_categories(val_df, mappings)
    test_cat = encode_categories(test_df, mappings)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET].to_numpy(dtype=np.float32)

    train_loader = make_loader(train_num, train_cat, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(val_num, val_cat, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(test_num, test_cat, y_test, args.batch_size, shuffle=False)

    categorical_cardinalities = [len(mappings[col]) + 1 for col in CATEGORICAL_FEATURES]
    model = HousePriceTransformer(
        num_numeric=len(NUMERIC_FEATURES),
        categorical_cardinalities=categorical_cardinalities,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.d_model * 2,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    history: list[dict[str, float]] = []
    best_state_path = OUTPUT_DIR / "transformer_model.pt"

    print(f"Rows: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print(f"Device: {device}")
    for epoch in range(1, args.epochs + 1):
        # BƯỚC 8: Trong quá trình train, model dùng tập train để dự đoán giá, so với giá thật, tính MSE, rồi sửa trọng số
        model.train()
        total_loss = 0.0
        seen = 0
        for numeric_values, categorical_ids, y in train_loader:
            numeric_values = numeric_values.to(device)
            categorical_ids = categorical_ids.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            # Gọi hàm forward (Bước 7) để lấy giá dự đoán tạm thời
            output = model(numeric_values, categorical_ids)
            loss = criterion(output, y) # Tính toán lỗi MSE

            loss.backward() # Lan truyền ngược sai số để tính đạo hàm
            optimizer.step() # Sửa các trọng số bên trong model để lần sau đoán tốt hơn

            total_loss += float(loss.item()) * len(y)
            seen += len(y)

        train_mse = total_loss / seen

        # BƯỚC 9: Sau mỗi epoch, model được kiểm tra trên tập validation.
        # Nếu validation MSE không giảm trong nhiều epoch liên tiếp thì dừng sớm và lấy model ở epoch tốt nhất.
        val_mse, _, _ = evaluate(model, val_loader, device, criterion)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        print(f"Epoch {epoch:03d} | train MSE={train_mse:.4f} | validation MSE={val_mse:.4f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_params": {
                        "num_numeric": len(NUMERIC_FEATURES),
                        "categorical_cardinalities": categorical_cardinalities,
                        "d_model": args.d_model,
                        "nhead": args.heads,
                        "num_layers": args.layers,
                        "dim_feedforward": args.d_model * 2,
                        "dropout": args.dropout,
                    },
                    "features": FEATURES,
                    "numeric_features": NUMERIC_FEATURES,
                    "categorical_features": CATEGORICAL_FEATURES,
                    "target": TARGET,
                },
                best_state_path,
            )
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping: validation MSE did not improve for {args.patience} epochs.")
                break

    # Tải lại trọng số tốt nhất được chọn từ Bước 9
    checkpoint = torch.load(best_state_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    # BƯỚC 10: Cuối cùng dùng tập test để tính MAE, RMSE, R², MAPE, rồi so sánh Transformer với XGBoost.
    test_mse, test_pred, test_true = evaluate(model, test_loader, device, criterion)
    train_mse, train_pred, train_true = evaluate(model, train_loader, device, criterion)

    test_metrics = regression_metrics(test_true, test_pred)
    train_metrics = regression_metrics(train_true, train_pred)
    overfit_gap = train_metrics["R2"] - test_metrics["R2"]

    preprocessing = {
        "scaler": scaler,
        "category_mappings": mappings,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "target": TARGET,
    }
    with open(OUTPUT_DIR / "preprocessing.pkl", "wb") as f:
        pickle.dump(preprocessing, f)

    meta = {
        "model_name": "Tabular Transformer",
        "best_epoch": best_epoch,
        "best_validation_mse": float(best_val_loss),
        "train_metrics": train_metrics,
        "metrics": test_metrics,
        "overfit_gap": float(overfit_gap),
        "history": history,
        "hyperparameters": vars(args),
        "rows": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
    }
    with open(OUTPUT_DIR / "transformer_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(OUTPUT_DIR / "transformer_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_plots(history, test_true, test_pred)
    save_xgboost_comparison(test_metrics)

    print("\nFinal test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"  overfit_gap: {overfit_gap:.4f}")
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
