"""
So sánh model gốc và model Hà Nội-only.

Mục tiêu:
- Không sửa app cũ.
- Không sửa model cũ trong models/.
- Đọc metadata của 2 model và sinh report/biểu đồ so sánh trong thư mục experiment.

Chạy từ root project:
    python experiments/hanoi_only/compare_models.py

Output:
    experiments/hanoi_only/comparison_report.md
    experiments/hanoi_only/comparison_hanoi_vs_all.png
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

BASE_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
HANOI_MODEL_DIR = EXPERIMENT_DIR / 'models_hanoi'

ORIGINAL_META_PATH = BASE_DIR / 'models' / 'model_meta.pkl'
HANOI_META_PATH = HANOI_MODEL_DIR / 'model_meta.pkl'

DATA_CC = BASE_DIR / 'data' / 'processed' / 'cleaned_chung_cu.csv'
DATA_ND = BASE_DIR / 'data' / 'processed' / 'cleaned_nha_dat.csv'

REPORT_PATH = EXPERIMENT_DIR / 'comparison_report.md'
CHART_PATH = EXPERIMENT_DIR / 'comparison_hanoi_vs_all.png'

HANOI_DISTRICTS = [
    'Ba Đình', 'Hoàn Kiếm', 'Tây Hồ', 'Long Biên', 'Cầu Giấy',
    'Đống Đa', 'Hai Bà Trưng', 'Hoàng Mai', 'Thanh Xuân',
    'Hà Đông', 'Bắc Từ Liêm', 'Nam Từ Liêm',
    'Sơn Tây', 'Ba Vì', 'Chương Mỹ', 'Đan Phượng', 'Đông Anh',
    'Gia Lâm', 'Hoài Đức', 'Mê Linh', 'Mỹ Đức', 'Phú Xuyên',
    'Phúc Thọ', 'Quốc Oai', 'Sóc Sơn', 'Thạch Thất', 'Thanh Oai',
    'Thanh Trì', 'Thường Tín', 'Ứng Hòa',
]

FEATURE_COLS = [
    'price_billion', 'area_m2', 'bedrooms_num', 'district', 'direction',
    'furniture_std', 'legal_std', 'floors_num', 'frontage_m', 'road_width_m', 'loai_bds'
]


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f'Không tìm thấy file: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def scalar(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def fmt(value, digits=4):
    value = scalar(value)
    if value is None:
        return 'N/A'
    return f'{float(value):.{digits}f}'


def fmt_percent(value, digits=1):
    value = scalar(value)
    if value is None:
        return 'N/A'
    return f'{float(value) * 100:.{digits}f}%'


def best_cv_name(meta):
    cv_results = meta.get('cv_results', {})
    if not cv_results:
        return 'N/A', None, None
    name = max(cv_results, key=lambda n: cv_results[n].get('cv_mean', float('-inf')))
    result = cv_results[name]
    return name, result.get('cv_mean'), result.get('cv_std')


def load_training_like_dataset():
    df_cc = pd.read_csv(DATA_CC)
    df_nd = pd.read_csv(DATA_ND)

    df_cc['loai_bds'] = 'chung_cu'
    df_nd['loai_bds'] = 'nha_dat'

    if 'balcony_direction' in df_cc.columns:
        df_cc = df_cc.drop(columns=['balcony_direction'])
    for col in ['floors_num', 'frontage_m', 'road_width_m']:
        df_cc[col] = 0

    df_all = pd.concat([df_cc[FEATURE_COLS], df_nd[FEATURE_COLS]], ignore_index=True)
    df_all = df_all.dropna(subset=['price_billion', 'area_m2'])
    df_all = df_all[(df_all['price_billion'] >= 1) & (df_all['price_billion'] <= 200)]
    df_all = df_all[(df_all['area_m2'] >= 10) & (df_all['area_m2'] <= 1000)]
    df_all['district'] = df_all['district'].astype(str).str.strip()
    return df_all


def dataset_summary():
    df_all = load_training_like_dataset()
    hanoi_mask = df_all['district'].isin(HANOI_DISTRICTS)
    df_hanoi = df_all[hanoi_mask].copy()

    return {
        'all_count': int(len(df_all)),
        'hanoi_count': int(len(df_hanoi)),
        'hanoi_ratio': float(len(df_hanoi) / len(df_all)) if len(df_all) else 0,
        'all_by_type': df_all['loai_bds'].value_counts().to_dict(),
        'hanoi_by_type': df_hanoi['loai_bds'].value_counts().to_dict(),
        'hanoi_districts_present': df_hanoi['district'].value_counts().index.tolist(),
    }


def meta_summary(label, scope, meta, dataset_size):
    metrics = meta.get('metrics', {})
    best_name = meta.get('best_model_name', 'N/A')
    cv_name, cv_mean, cv_std = best_cv_name(meta)

    return {
        'label': label,
        'scope': scope,
        'dataset_size': dataset_size,
        'best_model_test': best_name,
        'r2': scalar(metrics.get('R2')),
        'rmse': scalar(metrics.get('RMSE')),
        'mae': scalar(metrics.get('MAE')),
        'mape': scalar(metrics.get('MAPE')),
        'best_model_cv': cv_name,
        'cv_mean': scalar(cv_mean),
        'cv_std': scalar(cv_std),
    }


def create_chart(rows):
    labels = [row['label'] for row in rows]
    r2_values = [row['r2'] for row in rows]
    mae_values = [row['mae'] for row in rows]
    mape_values = [row['mape'] * 100 for row in rows]
    cv_values = [row['cv_mean'] for row in rows]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('So sánh model gốc vs model Hà Nội-only', fontweight='bold')

    chart_specs = [
        ('R² test ↑', r2_values, '#3498db', '.3f'),
        ('MAE test ↓ (tỷ)', mae_values, '#e67e22', '.2f'),
        ('MAPE test ↓ (%)', mape_values, '#9b59b6', '.1f'),
        ('CV R² mean ↑', cv_values, '#2ecc71', '.3f'),
    ]

    for ax, (title, values, color, number_format) in zip(axes, chart_specs):
        bars = ax.bar(labels, values, color=color, edgecolor='black', linewidth=0.7)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=12, ha='right')
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01 if bar.get_height() >= 0 else bar.get_height() * 0.99,
                format(value, number_format),
                ha='center',
                va='bottom',
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_report(rows, data_info):
    original, hanoi = rows
    hanoi_by_type = data_info['hanoi_by_type']
    all_by_type = data_info['all_by_type']

    report = f"""# So sánh model gốc và model Hà Nội-only

## Mục tiêu

Experiment này được tạo để làm đúng hướng: **không tạo CSV mới, không sửa app cũ, không đụng model cũ**.
Thay vào đó, script Hà Nội-only dùng lại dữ liệu tổng hiện có rồi lọc `district` thuộc danh sách quận/huyện Hà Nội ngay trong quá trình training.

## File liên quan

| Nhóm | Đường dẫn |
|---|---|
| Training gốc | `run_training.py` |
| Model gốc | `models/best_model_pipeline.pkl` |
| Metadata gốc | `models/model_meta.pkl` |
| Training Hà Nội-only | `experiments/hanoi_only/run_training_hanoi.py` |
| Model Hà Nội-only | `experiments/hanoi_only/models_hanoi/best_model_pipeline.pkl` |
| Metadata Hà Nội-only | `experiments/hanoi_only/models_hanoi/model_meta.pkl` |
| Biểu đồ so sánh | `experiments/hanoi_only/comparison_hanoi_vs_all.png` |

## Dataset sau filter training

| Dataset | Số mẫu | Chung cư | Nhà đất | Ghi chú |
|---|---:|---:|---:|---|
| Model gốc | {data_info['all_count']:,} | {int(all_by_type.get('chung_cu', 0)):,} | {int(all_by_type.get('nha_dat', 0)):,} | Hà Nội + TP.HCM + Đà Nẵng |
| Model Hà Nội-only | {data_info['hanoi_count']:,} | {int(hanoi_by_type.get('chung_cu', 0)):,} | {int(hanoi_by_type.get('nha_dat', 0)):,} | Chỉ giữ district thuộc Hà Nội |

Tỷ lệ dữ liệu Hà Nội so với dataset sau filter training: **{data_info['hanoi_ratio'] * 100:.1f}%**.

Các quận/huyện Hà Nội đang xuất hiện trong dataset:

```text
{', '.join(data_info['hanoi_districts_present'])}
```

## Bảng so sánh kết quả

| Tiêu chí | Model gốc | Model Hà Nội-only |
|---|---:|---:|
| Phạm vi train | {original['scope']} | {hanoi['scope']} |
| Số mẫu train/eval source | {original['dataset_size']:,} | {hanoi['dataset_size']:,} |
| Best model theo test | {original['best_model_test']} | {hanoi['best_model_test']} |
| R² test | {fmt(original['r2'])} | {fmt(hanoi['r2'])} |
| RMSE test | {fmt(original['rmse'])} tỷ | {fmt(hanoi['rmse'])} tỷ |
| MAE test | {fmt(original['mae'])} tỷ | {fmt(hanoi['mae'])} tỷ |
| MAPE test | {fmt_percent(original['mape'])} | {fmt_percent(hanoi['mape'])} |
| Best model theo CV | {original['best_model_cv']} | {hanoi['best_model_cv']} |
| CV R² mean | {fmt(original['cv_mean'])} ± {fmt(original['cv_std'])} | {fmt(hanoi['cv_mean'])} ± {fmt(hanoi['cv_std'])} |

## Nhận xét

1. **Model gốc** có R² test cao hơn vì train trên toàn bộ dữ liệu nhiều thành phố, số mẫu lớn hơn.
2. **Model Hà Nội-only** đúng hơn với mục tiêu mới: dự đoán giá nhà trong Hà Nội, vì không học mặt bằng giá của TP.HCM/Đà Nẵng.
3. Sau khi thu hẹp phạm vi địa lý, kết quả test/CV thay đổi: model tốt nhất theo test là **{hanoi['best_model_test']}**, còn model tốt nhất theo CV là **{hanoi['best_model_cv']}**.
4. Việc đặt experiment trong `experiments/hanoi_only/` giúp giữ nguyên hệ thống cũ để demo, đồng thời có artifact rõ ràng để so sánh.

## Cách chạy lại

```bash
python experiments/hanoi_only/run_training_hanoi.py
python experiments/hanoi_only/compare_models.py
```

## Kết luận ngắn để đưa vào báo cáo/defense

> Nhóm giữ nguyên model gốc để đối chiếu, sau đó tạo experiment Hà Nội-only. Experiment này không sinh thêm CSV riêng mà lọc trực tiếp các dòng có `district` thuộc Hà Nội trong quá trình training. Kết quả cho thấy model Hà Nội-only phù hợp hơn với phạm vi bài toán mới, còn model gốc phù hợp hơn nếu muốn dự đoán tổng quát trên nhiều thành phố.
"""
    REPORT_PATH.write_text(report, encoding='utf-8')


def main():
    print('Đang đọc metadata model...')
    original_meta = load_pickle(ORIGINAL_META_PATH)
    hanoi_meta = load_pickle(HANOI_META_PATH)

    print('Đang tính thống kê dataset...')
    data_info = dataset_summary()

    rows = [
        meta_summary('Model gốc', 'HN + HCM + Đà Nẵng', original_meta, data_info['all_count']),
        meta_summary('Hà Nội-only', 'Chỉ Hà Nội', hanoi_meta, hanoi_meta.get('dataset_size', data_info['hanoi_count'])),
    ]

    print('Đang tạo biểu đồ so sánh...')
    create_chart(rows)

    print('Đang tạo report markdown...')
    create_report(rows, data_info)

    print('\nHoàn thành.')
    print(f'  Report : {REPORT_PATH}')
    print(f'  Chart  : {CHART_PATH}')


if __name__ == '__main__':
    main()
