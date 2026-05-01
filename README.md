# 🤖 BƯỚC 3: HUẤN LUYỆN MÔ HÌNH AI

> **Mục tiêu:** Train 4 thuật toán ML trên dữ liệu BĐS Hà Nội, chọn model tốt nhất, xuất file `.pkl`

---

## 📋 Yêu cầu

- Python **3.9+**
- Kết nối Internet (để tải dataset từ GitHub)

---

## ⚙️ Cài đặt thư viện

Mở Terminal tại thư mục gốc dự án, chạy lệnh:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

---

## 🚀 Cách chạy

### Cách 1 — Script Python (nhanh nhất)

```powershell
# Windows — cần thêm dòng này nếu terminal báo lỗi ký tự
$env:PYTHONIOENCODING = 'utf-8'

python run_training.py
```

Kết quả mong đợi sau khi chạy xong:

```
============================================================
DU AN DU DOAN GIA BAT DONG SAN VIET NAM
============================================================

[1/6] Dang tai du lieu...
  Chung cu: 1031 ban ghi
  Nha dat : 1551 ban ghi

[2/6] Tien xu ly du lieu...
  Dataset tong hop sau loc: 2581 ban ghi

[3/6] Label Encoding...

[4/6] Chia du lieu 80/20...
  Train: 2064, Test: 517

[5/6] Huan luyen 4 thuat toan ML...
  Linear Regression    RMSE=7.514  MAE=4.678  R²=0.6853
  Decision Tree        RMSE=9.957  MAE=4.635  R²=0.4473
  Random Forest        RMSE=6.953  MAE=3.642  R²=0.7305
  XGBoost              RMSE=6.344  MAE=3.378  R²=0.7757  <- TOT NHAT

[6/6] Xuat model va ket qua...
  Da luu: models/best_model.pkl
  Da luu: models/scaler.pkl
  Da luu: models/label_encoders.pkl
  Da luu: models/model_meta.pkl
  Da luu: models/model_comparison.png
  Da luu: models/feature_importance.png
============================================================
```

---

### Cách 2 — Jupyter Notebook (xem từng bước)

**Cài Jupyter:**
```bash
pip install notebook
```

**Khởi động:**
```bash
jupyter notebook
```

Trình duyệt mở `http://localhost:8888`, chạy lần lượt:

| Notebook | Nội dung |
|---|---|
| `notebooks/03_training.ipynb` | Train 4 thuật toán, vẽ biểu đồ so sánh |
| `notebooks/04_evaluation.ipynb` | Đánh giá chi tiết, phân tích sai số |

> ⚠️ Phải chạy `03_training.ipynb` trước để tạo file `.pkl`, sau đó mới chạy `04_evaluation.ipynb`.

---

## 📊 Kết quả 4 mô hình

| Thuật toán | RMSE (tỷ) | MAE (tỷ) | R² Score |
|---|---|---|---|
| Linear Regression | 7.514 | 4.678 | 68.5% |
| Decision Tree | 9.957 | 4.635 | 44.7% |
| Random Forest | 6.953 | 3.642 | 73.1% |
| **XGBoost** ✅ | **6.344** | **3.379** | **77.6%** |

**→ XGBoost được chọn làm model chính** vì R² cao nhất (77.6%), MAE thấp nhất (~3.4 tỷ).

---

## 📁 File đầu ra

Sau khi chạy xong, thư mục `models/` sẽ có:

```
models/
├── best_model.pkl          ← Não AI (XGBoost) — dùng cho Flask API
├── scaler.pkl              ← Bộ chuẩn hóa dữ liệu
├── label_encoders.pkl      ← Bộ mã hóa nhãn (quận, hướng, nội thất...)
├── model_meta.pkl          ← Metadata: tên model, điểm số, danh sách features
├── model_comparison.png    ← Biểu đồ so sánh 4 model
└── feature_importance.png  ← Biểu đồ tầm quan trọng của từng đặc trưng
```

---

## ❌ Lỗi thường gặp

**Lỗi UnicodeEncodeError (Windows)**
```
UnicodeEncodeError: 'charmap' codec can't encode character
```
→ Thêm dòng này trước khi chạy:
```powershell
$env:PYTHONIOENCODING = 'utf-8'
```

---

**Lỗi ModuleNotFoundError**
```
ModuleNotFoundError: No module named 'xgboost'
```
→ Cài lại thư viện:
```bash
pip install xgboost scikit-learn pandas numpy matplotlib seaborn
```

---

**Lỗi FileNotFoundError khi mở notebook `04_evaluation.ipynb`**
```
FileNotFoundError: models/best_model.pkl not found
```
→ Chưa chạy bước training. Chạy script trước:
```bash
python run_training.py
```
