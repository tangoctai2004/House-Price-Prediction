"""
Training riêng cho bài toán dự đoán giá BĐS Hà Nội.

Ý tưởng theo hướng của Thái:
- Không tạo CSV mới.
- Không sửa file train/model cũ.
- Dùng dữ liệu tổng hiện có, lọc các dòng có district thuộc Hà Nội ngay trong lúc train.

Chạy từ root project:
    python experiments/hanoi_only/run_training_hanoi.py

Output được lưu riêng tại:
    experiments/hanoi_only/models_hanoi/
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Chỉ tắt các warning không quan trọng, giữ lại lỗi thật khi có.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

BASE_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / 'models_hanoi'

DATA_CC = BASE_DIR / 'data' / 'processed' / 'cleaned_chung_cu.csv'
DATA_ND = BASE_DIR / 'data' / 'processed' / 'cleaned_nha_dat.csv'

# Danh sách đầy đủ quận/huyện/thị xã Hà Nội.
# Dataset hiện tại chỉ có một phần trong danh sách này; quận nào không có dữ liệu thì tự không match.
HANOI_DISTRICTS = [
    'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Long Biên', 'Cầu Giấy',
    'Đống Đa', 'Hai Bà Trưng', 'Hoàng Mai', 'Thanh Xuân',
    'Hà Đông', 'Bắc Từ Liêm', 'Nam Từ Liêm',
    'Sơn Tây', 'Ba Vì', 'Chương Mỹ', 'Đan Phượng', 'Đông Anh',
    'Gia Lâm', 'Hoài Đức', 'Mê Linh', 'Mỹ Đức', 'Phú Xuyên',
    'Phúc Thọ', 'Quốc Oai', 'Sóc Sơn', 'Thạch Thất', 'Thanh Oai',
    'Thanh Trì', 'Thường Tín', 'Ứng Hòa',
]

FEATURES = [
    'area_m2', 'bedrooms_num', 'district', 'direction', 'furniture_std', 'legal_std',
    'floors_num', 'frontage_m', 'road_width_m', 'loai_bds'
]
TARGET = 'price_billion'

NUMERIC_FEATURES = ['area_m2', 'bedrooms_num', 'floors_num', 'frontage_m', 'road_width_m']
CATEGORICAL_FEATURES = ['district', 'direction', 'furniture_std', 'legal_std', 'loai_bds']


def load_and_prepare_data():
    print("\n[1/6] Đang tải dữ liệu local...")
    df_cc = pd.read_csv(DATA_CC)
    df_nd = pd.read_csv(DATA_ND)

    df_cc['loai_bds'] = 'chung_cu'
    df_nd['loai_bds'] = 'nha_dat'

    print(f"  Chung cư gốc: {len(df_cc)} bản ghi")
    print(f"  Nhà đất gốc : {len(df_nd)} bản ghi")

    print("\n[2/6] Tiền xử lý và lọc riêng Hà Nội...")

    # Chung cư không có các feature nhà đất, giữ cách xử lý như run_training.py cũ.
    if 'balcony_direction' in df_cc.columns:
        df_cc = df_cc.drop(columns=['balcony_direction'])
    for col in ['floors_num', 'frontage_m', 'road_width_m']:
        df_cc[col] = 0

    all_cols = [TARGET, 'area_m2', 'bedrooms_num', 'district', 'direction',
                'furniture_std', 'legal_std', 'floors_num', 'frontage_m',
                'road_width_m', 'loai_bds']

    df_all = pd.concat([df_cc[all_cols], df_nd[all_cols]], ignore_index=True)
    print(f"  Dataset tổng trước lọc: {len(df_all)} bản ghi")

    df_all = df_all.dropna(subset=[TARGET, 'area_m2'])
    df_all = df_all[(df_all[TARGET] >= 1) & (df_all[TARGET] <= 200)]
    df_all = df_all[(df_all['area_m2'] >= 10) & (df_all['area_m2'] <= 1000)]
    print(f"  Sau lọc giá/diện tích: {len(df_all)} bản ghi")

    # Điểm khác chính so với script cũ: chỉ giữ quận/huyện Hà Nội.
    df_all['district'] = df_all['district'].astype(str).str.strip()
    df_hanoi = df_all[df_all['district'].isin(HANOI_DISTRICTS)].copy()

    print(f"  Dataset Hà Nội sau lọc: {len(df_hanoi)} bản ghi")
    print("  Theo loại BĐS:")
    print(df_hanoi['loai_bds'].value_counts().to_string())
    print("  Các quận/huyện có trong dữ liệu:")
    print(", ".join(df_hanoi['district'].value_counts().index.tolist()))

    if df_hanoi.empty:
        raise ValueError('Không còn dữ liệu Hà Nội sau khi lọc district. Kiểm tra lại danh sách HANOI_DISTRICTS.')

    for col in CATEGORICAL_FEATURES:
        df_hanoi[col] = df_hanoi[col].astype(str)

    return df_hanoi


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def train_eval(name, base_model, preprocessor, X_train, X_test, y_train, y_test, results):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model),
    ])
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    overfit_gap = r2_train - r2_test

    results[name] = {
        'RMSE': rmse_test,
        'MAE': mae_test,
        'R2': r2_test,
        'MAPE': mape_test,
        'R2_train': r2_train,
        'RMSE_train': rmse_train,
        'overfit_gap': overfit_gap,
        'pipeline': pipeline,
        'y_pred': y_pred_test,
    }

    status = '✅' if r2_test > 0.7 and overfit_gap < 0.15 else '⚠️'
    print(
        f"  {status} {name:<20} "
        f"R²_train={r2_train:.4f}  R²_test={r2_test:.4f}  "
        f"Gap={overfit_gap:.4f}  RMSE={rmse_test:.3f}  "
        f"MAE={mae_test:.3f}  MAPE={mape_test:.1%}"
    )


def get_feature_importance(best_pipeline):
    base_model = best_pipeline.named_steps['model']
    ohe_cols = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERIC_FEATURES + list(ohe_cols)

    if hasattr(base_model, 'feature_importances_'):
        fi = base_model.feature_importances_
    elif hasattr(base_model, 'coef_'):
        fi = np.abs(base_model.coef_)
    else:
        fi = np.zeros(len(all_feature_names))

    fi = np.asarray(fi, dtype=float)
    fi_dict = dict(zip(all_feature_names, [float(v) for v in fi]))
    total_importance = float(fi.sum()) if float(fi.sum()) > 0 else 1.0
    return fi_dict, total_importance, all_feature_names


def save_charts(results, cv_results, best_name):
    print("\n[8/8] Tạo biểu đồ...")
    names = list(results.keys())
    colors = ['#3498db', '#f39c12', '#2ecc71', '#e74c3c']

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, metric, title, best_fn in zip(
        axes,
        ['RMSE', 'MAE', 'R2', 'MAPE'],
        ['RMSE↓', 'MAE↓', 'R²↑', 'MAPE↓'],
        [min, min, max, min],
    ):
        vals = [results[n][metric] for n in names]
        best_val = best_fn(vals)
        bar_colors = ['gold' if v == best_val else c for v, c in zip(vals, colors)]
        bars = ax.bar(names, vals, color=bar_colors, edgecolor='black', linewidth=0.7)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
        fmt = '.1%' if metric == 'MAPE' else '.3f'
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{val:{fmt}}', ha='center', va='bottom', fontsize=8)
    plt.suptitle('So sánh 4 thuật toán ML — Dữ liệu Hà Nội', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Đã lưu: {OUTPUT_DIR / 'model_comparison.png'}")

    fig_ov, ax_ov = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(names))
    width = 0.35
    train_r2 = [results[n]['R2_train'] for n in names]
    test_r2 = [results[n]['R2'] for n in names]
    ax_ov.bar(x_pos - width / 2, train_r2, width, label='R² Train', color='#3498db', edgecolor='black', linewidth=0.5)
    ax_ov.bar(x_pos + width / 2, test_r2, width, label='R² Test', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax_ov.set_ylabel('R² Score')
    ax_ov.set_title('Phân tích Overfitting — Hà Nội', fontweight='bold')
    ax_ov.set_xticks(x_pos)
    ax_ov.set_xticklabels(names, fontsize=9)
    ax_ov.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ov)
    print(f"  Đã lưu: {OUTPUT_DIR / 'overfitting_analysis.png'}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    for ax, model_name in [(axes2[0], 'Random Forest'), (axes2[1], 'XGBoost')]:
        pipeline = results[model_name]['pipeline']
        base_model = pipeline.named_steps['model']
        ohe_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
        all_features = NUMERIC_FEATURES + list(ohe_cols)
        fi = base_model.feature_importances_
        fi_df = pd.DataFrame({'feature': all_features, 'importance': fi}).sort_values('importance', ascending=False).head(15)
        fi_df = fi_df.sort_values('importance', ascending=True)
        ax.barh(fi_df['feature'], fi_df['importance'], color='#3498db')
        ax.set_title(f'Top 15 Features — {model_name}', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Đã lưu: {OUTPUT_DIR / 'feature_importance.png'}")


def test_predictions(best_pipeline):
    """
    Test dự đoán nhanh — tương đương Section 11 của 03_training.ipynb.
    Sử dụng Pipeline nên không cần encode/scale thủ công.
    """
    def predict_price(area_m2, bedrooms, district_name, furniture, legal, loai='chung_cu',
                      floors=0, frontage=0, road_width=0, direction='Không rõ'):
        sample = pd.DataFrame([{
            'area_m2': area_m2, 'bedrooms_num': bedrooms,
            'district': district_name, 'direction': direction,
            'furniture_std': furniture, 'legal_std': legal,
            'floors_num': floors, 'frontage_m': frontage,
            'road_width_m': road_width, 'loai_bds': loai
        }])
        return best_pipeline.predict(sample)[0]

    test_cases = [
        ('Chung cư 75m², 2PN, Cầu Giấy, Đầy đủ, Sổ đỏ',
         dict(area_m2=75, bedrooms=2, district_name='Cầu Giấy',
              furniture='Đầy đủ', legal='Sổ đỏ/Sổ hồng', loai='chung_cu')),
        ('Chung cư 50m², 2PN, Hà Đông, Cơ bản, HĐMB',
         dict(area_m2=50, bedrooms=2, district_name='Hà Đông',
              furniture='Cơ bản', legal='HĐMB', loai='chung_cu')),
        ('Nhà đất 45m², 4PN, Thanh Xuân, Đầy đủ, Sổ đỏ, 5 tầng',
         dict(area_m2=45, bedrooms=4, district_name='Thanh Xuân',
              furniture='Đầy đủ', legal='Sổ đỏ/Sổ hồng', loai='nha_dat',
              floors=5, frontage=4.2, road_width=4.0)),
        ('Nhà đất 60m², 3PN, Ba Đình, Đầy đủ, Sổ đỏ, 4 tầng',
         dict(area_m2=60, bedrooms=3, district_name='Ba Đình',
              furniture='Đầy đủ', legal='Sổ đỏ/Sổ hồng', loai='nha_dat',
              floors=4, frontage=4.0, road_width=5.0)),
        ('Nhà đất 40m², 3PN, Đống Đa, Cơ bản, Sổ đỏ, 3 tầng',
         dict(area_m2=40, bedrooms=3, district_name='Đống Đa',
              furniture='Cơ bản', legal='Sổ đỏ/Sổ hồng', loai='nha_dat',
              floors=3, frontage=3.5, road_width=3.0)),
    ]

    print('\n🔮 THỬ NGHIỆM DỰ ĐOÁN GIÁ (Hà Nội):')
    print('=' * 60)
    for desc, params in test_cases:
        price = predict_price(**params)
        print(f'  📍 {desc}')
        print(f'     → Giá dự đoán: {price:.2f} tỷ VNĐ (~{price * 1000:.0f} triệu)\n')


def main():
    print("=" * 70)
    print("TRAINING RIÊNG CHO DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN HÀ NỘI")
    print("=" * 70)

    df_hanoi = load_and_prepare_data()

    print("\n[3/6] Thiết lập Pipeline tiền xử lý & chia dữ liệu...")
    X = df_hanoi[FEATURES]
    y = df_hanoi[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    preprocessor = build_preprocessor()
    results = {}

    print("\n[5/8] Huấn luyện 4 thuật toán ML trên dữ liệu Hà Nội...")
    train_eval('Linear Regression', LinearRegression(), preprocessor, X_train, X_test, y_train, y_test, results)
    train_eval('Decision Tree', DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42), preprocessor, X_train, X_test, y_train, y_test, results)
    train_eval('Random Forest', RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1), preprocessor, X_train, X_test, y_train, y_test, results)
    train_eval('XGBoost', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0), preprocessor, X_train, X_test, y_train, y_test, results)

    best_name = max(results, key=lambda name: results[name]['R2'])
    print(f"\nModel tốt nhất trên test set Hà Nội: {best_name}  (R² test = {results[best_name]['R2']:.4f}, Gap = {results[best_name]['overfit_gap']:.4f})")

    # Bảng so sánh kết quả — tương đương Section 8 notebook
    print('\n' + '=' * 65)
    print('📊 BẢNG SO SÁNH KẾT QUẢ 4 MÔ HÌNH (HÀ NỘI)')
    print('=' * 65)
    compare_df = pd.DataFrame({
        name: {'RMSE (tỷ)': v['RMSE'], 'MAE (tỷ)': v['MAE'], 'R² Score': v['R2']}
        for name, v in results.items()
    }).T.round(4)
    print(compare_df.to_string())
    print(f'\n🏆 MODEL TỐT NHẤT: {best_name}  (R² = {results[best_name]["R2"]:.4f})')

    print("\n[6/8] Cross-Validation 5-Fold trên dữ liệu Hà Nội...")
    cv_results = {}
    for name, result in results.items():
        scores = cross_val_score(result['pipeline'], X, y, cv=5, scoring='r2', n_jobs=-1)
        cv_results[name] = {
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
            'cv_scores': [float(score) for score in scores],
        }
        status = '✅' if scores.mean() > 0.7 else '⚠️'
        print(f"  {status} {name:<20} CV R² = {scores.mean():.4f} ± {scores.std():.4f}  ({', '.join(f'{s:.3f}' for s in scores)})")

    best_cv_name = max(cv_results, key=lambda name: cv_results[name]['cv_mean'])
    print(f"\n  📊 Best CV Hà Nội: {best_cv_name}  (CV R² = {cv_results[best_cv_name]['cv_mean']:.4f})")

    print("\n[7/8] Xuất model và metadata riêng cho Hà Nội...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_pipeline = results[best_name]['pipeline']

    with open(OUTPUT_DIR / 'best_model_pipeline.pkl', 'wb') as f:
        pickle.dump(best_pipeline, f)

    fi_dict, total_importance, _ = get_feature_importance(best_pipeline)
    model_meta = {
        'scope': 'hanoi_only',
        'filter_note': 'Chỉ train các dòng có district thuộc danh sách quận/huyện Hà Nội.',
        'hanoi_districts': HANOI_DISTRICTS,
        'dataset_size': int(len(df_hanoi)),
        'best_model_name': best_name,
        'features': FEATURES,
        'target': TARGET,
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'metrics': {k: v for k, v in results[best_name].items() if k in ['RMSE', 'MAE', 'R2', 'MAPE']},
        'all_results': {
            name: {k: v for k, v in data.items() if k in ['RMSE', 'MAE', 'R2', 'MAPE', 'R2_train', 'RMSE_train', 'overfit_gap']}
            for name, data in results.items()
        },
        'cv_results': cv_results,
        'feature_importance': fi_dict,
        'total_importance': total_importance,
    }
    with open(OUTPUT_DIR / 'model_meta.pkl', 'wb') as f:
        pickle.dump(model_meta, f)

    print(f"  Đã lưu: {OUTPUT_DIR / 'best_model_pipeline.pkl'}")
    print(f"  Đã lưu: {OUTPUT_DIR / 'model_meta.pkl'}")

    save_charts(results, cv_results, best_name)

    # Test dự đoán nhanh — tương đương Section 11 notebook
    test_predictions(best_pipeline)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TRAINING RIÊNG CHO HÀ NỘI")
    print("=" * 70)
    print("\nKết quả cuối cùng:")
    for name, result in results.items():
        tag = " ← TỐT NHẤT TEST" if name == best_name else ""
        cv = cv_results[name]
        print(
            f"  {name:<20} R²_train={result['R2_train']:.4f}  "
            f"R²_test={result['R2']:.4f}  Gap={result['overfit_gap']:.4f}  "
            f"MAE={result['MAE']:.4f}  MAPE={result['MAPE']:.1%}  "
            f"CV_R²={cv['cv_mean']:.4f}±{cv['cv_std']:.4f}{tag}"
        )
    print(f"\nCác file đã xuất trong thư mục: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
