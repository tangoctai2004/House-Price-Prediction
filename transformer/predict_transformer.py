"""Predict one house price with the trained Transformer model.

Run from project root after training:
    python transformer/predict_transformer.py --area-m2 80 --district "Cầu Giấy"
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformer_model import HousePriceTransformer


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area-m2", type=float, default=80)
    parser.add_argument("--bedrooms-num", type=float, default=3)
    parser.add_argument("--district", type=str, default="Cầu Giấy")
    parser.add_argument("--direction", type=str, default="Không rõ")
    parser.add_argument("--furniture-std", type=str, default="Không rõ")
    parser.add_argument("--legal-std", type=str, default="Sổ đỏ/Sổ hồng")
    parser.add_argument("--floors-num", type=float, default=4)
    parser.add_argument("--frontage-m", type=float, default=5)
    parser.add_argument("--road-width-m", type=float, default=6)
    parser.add_argument("--loai-bds", type=str, default="nha_dat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = OUTPUT_DIR / "transformer_model.pt"
    preprocessing_path = OUTPUT_DIR / "preprocessing.pkl"
    if not checkpoint_path.exists() or not preprocessing_path.exists():
        raise FileNotFoundError("Chưa có model. Hãy chạy: python transformer/train_transformer.py")

    with open(preprocessing_path, "rb") as f:
        preprocessing = pickle.load(f)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = HousePriceTransformer(**checkpoint["model_params"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    row = {
        "area_m2": args.area_m2,
        "bedrooms_num": args.bedrooms_num,
        "district": args.district,
        "direction": args.direction,
        "furniture_std": args.furniture_std,
        "legal_std": args.legal_std,
        "floors_num": args.floors_num,
        "frontage_m": args.frontage_m,
        "road_width_m": args.road_width_m,
        "loai_bds": args.loai_bds,
    }
    df = pd.DataFrame([row])
    numeric = preprocessing["scaler"].transform(df[preprocessing["numeric_features"]])

    cat_ids = []
    for col in preprocessing["categorical_features"]:
        mapping = preprocessing["category_mappings"][col]
        cat_ids.append(mapping.get(str(df.at[0, col]), 0))
    cat_ids = np.array([cat_ids], dtype=np.int64)

    with torch.no_grad():
        prediction = model(
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(cat_ids, dtype=torch.long),
        )

    print(f"Giá dự đoán bằng Transformer: {float(prediction.item()):.3f} tỷ VNĐ")


if __name__ == "__main__":
    main()
