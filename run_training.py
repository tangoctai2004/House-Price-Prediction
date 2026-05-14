"""
Script chạy thẳng toàn bộ pipeline training (tương đương 03_training.ipynb)
Chạy bằng: python run_training.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, os, warnings
# Chỉ tắt các warning không quan trọng, giữ lại ConvergenceWarning / DataConversionWarning
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
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

# ---- 3 & 4. Tiền xử lý & Chia dữ liệu ----
print("\n[3/6] Thiết lập Pipeline tiền xử lý & Chia dữ liệu...")
FEATURES = ['area_m2','bedrooms_num','district','direction','furniture_std','legal_std',
            'floors_num','frontage_m','road_width_m','loai_bds']
TARGET = 'price_billion'

df_m = df_all.copy()
# Đảm bảo các cột categorical là kiểu string
cat_cols = ['district','direction','furniture_std','legal_std','loai_bds']
for c in cat_cols:
    df_m[c] = df_m[c].astype(str)

X, y = df_m[FEATURES], df_m[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['area_m2', 'bedrooms_num', 'floors_num', 'frontage_m', 'road_width_m']
categorical_features = ['district', 'direction', 'furniture_std', 'legal_std', 'loai_bds']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# ---- 5. Train 4 mô hình ----
print("\n[5/8] Huấn luyện 4 thuật toán ML...")

results = {}

def train_eval(name, base_model, Xtr, Xte, ytr, yte):
    """Train model bằng Pipeline và đánh giá trên cả train + test set."""
    # Wrap model in a Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    pipeline.fit(Xtr, ytr)

    # Predict trên CẢ train và test để đánh giá overfitting
    yp_train = pipeline.predict(Xtr)
    yp_test  = pipeline.predict(Xte)

    # Metrics trên test set
    rmse_test = np.sqrt(mean_squared_error(yte, yp_test))
    mae_test  = mean_absolute_error(yte, yp_test)
    r2_test   = r2_score(yte, yp_test)
    mape_test = mean_absolute_percentage_error(yte, yp_test)

    # Metrics trên train set
    r2_train   = r2_score(ytr, yp_train)
    rmse_train = np.sqrt(mean_squared_error(ytr, yp_train))

    # Overfitting gap: R² train - R² test (càng nhỏ càng tốt)
    overfit_gap = r2_train - r2_test

    results[name] = {
        'RMSE': rmse_test, 'MAE': mae_test, 'R2': r2_test, 'MAPE': mape_test,
        'R2_train': r2_train, 'RMSE_train': rmse_train,
        'overfit_gap': overfit_gap,
        'pipeline': pipeline, 'y_pred': yp_test
    }

    status = '✅' if r2_test > 0.7 and overfit_gap < 0.15 else '⚠️'
    print(f"  {status} {name:<20} R²_train={r2_train:.4f}  R²_test={r2_test:.4f}  Gap={overfit_gap:.4f}  RMSE={rmse_test:.3f}  MAE={mae_test:.3f}  MAPE={mape_test:.1%}")

train_eval('Linear Regression', LinearRegression(), X_train, X_test, y_train, y_test)
train_eval('Decision Tree',     DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42), X_train, X_test, y_train, y_test)
train_eval('Random Forest',     RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1), X_train, X_test, y_train, y_test)
train_eval('XGBoost',           XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0), X_train, X_test, y_train, y_test)

best_name = max(results, key=lambda x: results[x]['R2'])
print(f"\nModel tốt nhất: {best_name}  (R² test = {results[best_name]['R2']:.4f}, Overfit gap = {results[best_name]['overfit_gap']:.4f})")

# ---- 6. Cross-Validation (5-Fold) để kiểm chứng độ ổn định ----
print("\n[6/8] Cross-Validation (5-Fold)...")
cv_results = {}
for name, r in results.items():
    cv_pipeline = r['pipeline']
    scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    cv_results[name] = {'cv_mean': float(scores.mean()), 'cv_std': float(scores.std()), 'cv_scores': [float(s) for s in scores]}
    status = '✅' if scores.mean() > 0.7 else '⚠️'
    print(f"  {status} {name:<20} CV R² = {scores.mean():.4f} ± {scores.std():.4f}  ({', '.join(f'{s:.3f}' for s in scores)})")

print(f"\n  📊 Best CV: {max(cv_results, key=lambda x: cv_results[x]['cv_mean'])}  (CV R² = {max(cv_results[x]['cv_mean'] for x in cv_results):.4f})")

# ---- 7. Xuất model và kết quả ----
print("\n[7/8] Xuất model và kết quả...")
os.makedirs('models', exist_ok=True)
best_pipeline = results[best_name]['pipeline']

with open('models/best_model_pipeline.pkl', 'wb') as f: pickle.dump(best_pipeline, f)

# Lấy Feature Importance để dùng cho XAI (Giải thích AI) trên Web
base_model = best_pipeline.named_steps['model']
ohe_cols = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_features_names = numeric_features + list(ohe_cols)
fi = base_model.feature_importances_
fi_dict = dict(zip(all_features_names, [float(v) for v in fi]))

# Tính tổng importance để dùng cho XAI (tỷ lệ % thực sự)
total_importance = float(sum(fi))

model_meta = {
    'best_model_name': best_name,
    'features': FEATURES,
    'target': TARGET,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    # Metrics test set
    'metrics': {k: v for k, v in results[best_name].items() if k in ['RMSE','MAE','R2','MAPE']},
    # Metrics đầy đủ (train + test + gap) cho mỗi model
    'all_results': {
        n: {k: v for k, v in d.items() if k in ['RMSE','MAE','R2','MAPE','R2_train','RMSE_train','overfit_gap']}
        for n, d in results.items()
    },
    # Cross-Validation scores
    'cv_results': cv_results,
    # Feature Importance
    'feature_importance': fi_dict,
    'total_importance': total_importance,
}
with open('models/model_meta.pkl', 'wb') as f: pickle.dump(model_meta, f)
print(" Đã lưu: models/best_model_pipeline.pkl")
print(" Đã lưu: models/model_meta.pkl")

# ---- 8. Biểu đồ so sánh ----
print("\n[8/8] Tạo biểu đồ...")

# 8a. Biểu đồ so sánh 4 metrics
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
names = list(results.keys())
colors = ['#3498db','#f39c12','#2ecc71','#e74c3c']
for ax, metric, title, best_fn in zip(axes, ['RMSE','MAE','R2','MAPE'],
    ['RMSE↓','MAE↓','R²↑','MAPE↓'], [min, min, max, min]):
    vals = [results[n][metric] for n in names]
    best_val = best_fn(vals)
    bar_colors = ['gold' if v == best_val else c for v, c in zip(vals, colors)]
    bars = ax.bar(names, vals, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    fmt = '.1%' if metric == 'MAPE' else '.3f'
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:{fmt}}', ha='center', va='bottom', fontsize=8)
plt.suptitle('So sánh 4 Thuật toán ML — Giá BĐS Việt Nam', fontweight='bold')
plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
print(" Đã lưu: models/model_comparison.png")

# 8b. Biểu đồ Overfitting Analysis: R² Train vs Test
fig_ov, ax_ov = plt.subplots(figsize=(8, 4))
x_pos = np.arange(len(names))
width = 0.35
train_r2 = [results[n]['R2_train'] for n in names]
test_r2  = [results[n]['R2']       for n in names]
bars1 = ax_ov.bar(x_pos - width/2, train_r2, width, label='R² Train', color='#3498db', edgecolor='black', linewidth=0.5)
bars2 = ax_ov.bar(x_pos + width/2, test_r2,  width, label='R² Test',  color='#e74c3c', edgecolor='black', linewidth=0.5)
ax_ov.set_ylabel('R² Score')
ax_ov.set_title('Phân tích Overfitting — R² Train vs Test', fontweight='bold')
ax_ov.set_xticks(x_pos)
ax_ov.set_xticklabels(names, fontsize=9)
ax_ov.legend()
for bar, val in zip(bars1, train_r2):
    ax_ov.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=8)
for bar, val in zip(bars2, test_r2):
    ax_ov.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig('models/overfitting_analysis.png', dpi=150, bbox_inches='tight')
print(" Đã lưu: models/overfitting_analysis.png")

# 8c. Feature importance
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
for ax, model_name in [(axes2[0],'Random Forest'), (axes2[1],'XGBoost')]:
    pipeline = results[model_name]['pipeline']
    base_model = pipeline.named_steps['model']
    
    # Lấy feature names sau khi One-Hot Encoding
    ohe_cols = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = numeric_features + list(ohe_cols)
    
    fi = base_model.feature_importances_
    
    # Chỉ lấy top 15 features quan trọng nhất
    fi_df = pd.DataFrame({'feature':all_features,'importance':fi}).sort_values('importance', ascending=False).head(15)
    fi_df = fi_df.sort_values('importance', ascending=True)
    
    ax.barh(fi_df['feature'], fi_df['importance'], color='#3498db')
    ax.set_title(f'Top 15 Features — {model_name}', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
print(" Đã lưu: models/feature_importance.png")

print("\n" + "="*60)
print(" HOÀN THÀNH: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH AI!")
print("="*60)
print(f"\n📊 KẾT QUẢ CUỐI CÙNG (Train → Test → Overfitting Gap):")
for name, r in results.items():
    tag = " ← TỐT NHẤT" if name == best_name else ""
    cv = cv_results[name]
    print(f"  {name:<20}  R²_train={r['R2_train']:.4f}  R²_test={r['R2']:.4f}  Gap={r['overfit_gap']:.4f}  MAE={r['MAE']:.4f}  MAPE={r['MAPE']:.1%}  CV_R²={cv['cv_mean']:.4f}±{cv['cv_std']:.4f}{tag}")
print(f"\nCác file đã xuất trong thư mục: models/")
