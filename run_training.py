"""
Script chạy thẳng toàn bộ pipeline training (tương đương 03_training.ipynb)
Chạy bằng: python run_training.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

print("="*60)
print("DỰ ÁN DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN VIỆT NAM")
print("="*60)

# ---- 1. Load dữ liệu ----
print("\n[1/6] Đang tải dữ liệu...")
URL_CC = 'https://raw.githubusercontent.com/tangoctai2004/House-Price-Prediction/refs/heads/main/data/processed/cleaned_chung_cu.csv'
URL_ND = 'https://raw.githubusercontent.com/tangoctai2004/House-Price-Prediction/refs/heads/main/data/processed/cleaned_nha_dat.csv'

df_cc = pd.read_csv(URL_CC)
df_nd = pd.read_csv(URL_ND)
df_cc['loai_bds'] = 'chung_cu'
df_nd['loai_bds'] = 'nha_dat'
print(f" Chung cư: {len(df_cc)} bản ghi")
print(f" Nhà đất : {len(df_nd)} bản ghi")

# ---- 2. Kết hợp & làm sạch ----
print("\n[2/6] Tiền xử lý dữ liệu...")
if 'balcony_direction' in df_cc.columns:
    df_cc = df_cc.drop(columns=['balcony_direction'])
for c in ['floors_num', 'frontage_m', 'road_width_m']:
    df_cc[c] = 0

ALL_COLS = ['price_billion','area_m2','bedrooms_num','district','direction',
            'furniture_std','legal_std','floors_num','frontage_m','road_width_m','loai_bds']
df_all = pd.concat([df_cc[ALL_COLS], df_nd[ALL_COLS]], ignore_index=True)
df_all = df_all.dropna(subset=['price_billion','area_m2'])
df_all = df_all[(df_all['price_billion']>=1) & (df_all['price_billion']<=200)]
df_all = df_all[(df_all['area_m2']>=10) & (df_all['area_m2']<=1000)]
print(f"  Dataset tổng hợp sau lọc: {len(df_all)} bản ghi")

# ---- 3. Encoding ----
print("\n[3/6] Label Encoding...")
cat_cols = ['district','direction','furniture_std','legal_std','loai_bds']
label_encoders = {}
df_m = df_all.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_m[col] = le.fit_transform(df_m[col].astype(str))
    label_encoders[col] = le
print("Encoding hoàn tất!")

# ---- 4. Train/Test split ----
print("\n[4/6] Chia dữ liệu 80/20...")
FEATURES = ['area_m2','bedrooms_num','district','direction','furniture_std','legal_std',
            'floors_num','frontage_m','road_width_m','loai_bds']
TARGET = 'price_billion'
X, y = df_m[FEATURES], df_m[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- 5. Train 4 mô hình ----
print("\n[5/6] Huấn luyện 4 thuật toán ML...")

results = {}

def train_eval(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(yte, yp))
    mae  = mean_absolute_error(yte, yp)
    r2   = r2_score(yte, yp)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model': model, 'y_pred': yp}
    print(f"  {'✅' if r2 > 0.7 else '⚠️'} {name:<20} RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

train_eval('Linear Regression', LinearRegression(), X_train_sc, X_test_sc, y_train, y_test)
train_eval('Decision Tree',     DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42), X_train, X_test, y_train, y_test)
train_eval('Random Forest',     RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1), X_train, X_test, y_train, y_test)
train_eval('XGBoost',           XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0), X_train, X_test, y_train, y_test)

best_name = max(results, key=lambda x: results[x]['R2'])
print(f"\nModel tốt nhất: {best_name}  (R² = {results[best_name]['R2']:.4f})")

# ---- 6. Xuất file ----
print("\n[6/6] Xuất model và kết quả...")
os.makedirs('models', exist_ok=True)
best_model = results[best_name]['model']

with open('models/best_model.pkl', 'wb') as f: pickle.dump(best_model, f)
with open('models/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('models/label_encoders.pkl', 'wb') as f: pickle.dump(label_encoders, f)

model_meta = {
    'best_model_name': best_name,
    'features': FEATURES,
    'target': TARGET,
    'metrics': {k: v for k, v in results[best_name].items() if k in ['RMSE','MAE','R2']},
    'all_results': {n: {k: v for k, v in d.items() if k in ['RMSE','MAE','R2']} for n, d in results.items()}
}
with open('models/model_meta.pkl', 'wb') as f: pickle.dump(model_meta, f)
print(" Đã lưu: models/best_model.pkl")
print(" Đã lưu: models/scaler.pkl")
print(" Đã lưu: models/label_encoders.pkl")
print(" Đã lưu: models/model_meta.pkl")

# Biểu đồ so sánh
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
names = list(results.keys())
colors = ['#3498db','#f39c12','#2ecc71','#e74c3c']
for ax, metric, title, best_fn in zip(axes, ['RMSE','MAE','R2'],
    ['RMSE↓','MAE↓','R²↑'], [min, min, max]):
    vals = [results[n][metric] for n in names]
    best_val = best_fn(vals)
    bar_colors = ['gold' if v == best_val else c for v, c in zip(vals, colors)]
    bars = ax.bar(names, vals, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.suptitle('So sánh 4 Thuật toán ML — Giá BĐS Hà Nội', fontweight='bold')
plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
print(" Đã lưu: models/model_comparison.png")

# Feature importance
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
for ax, model_name in [(axes2[0],'Random Forest'), (axes2[1],'XGBoost')]:
    fi = results[model_name]['model'].feature_importances_
    fi_df = pd.DataFrame({'feature':FEATURES,'importance':fi}).sort_values('importance')
    ax.barh(fi_df['feature'], fi_df['importance'], color='#3498db')
    ax.set_title(f'Feature Importance — {model_name}', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
print(" Đã lưu: models/feature_importance.png")

print("\n" + "="*60)
print(" HOÀN THÀNH BƯỚC: HUẤN LUYỆN MÔ HÌNH AI!")
print("="*60)
print(f"\nKẾT QUẢ CUỐI CÙNG:")
for name, r in results.items():
    tag = " ← TỐT NHẤT" if name == best_name else ""
    print(f"  {name:<20}  RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  R²={r['R2']:.4f}{tag}")
print(f"\nCác file đã xuất trong thư mục: models/")
