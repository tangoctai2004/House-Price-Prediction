# TÀI LIỆU BẢO VỆ — PHẦN 3: CÂU HỎI & YÊU CẦU KHI DEMO TRỰC TIẾP
# "Giảng viên yêu cầu làm gì đó ngay trước mặt"

> ⚠️ Đây là phần NGUY HIỂM NHẤT — giảng viên sẽ yêu cầu bạn thao tác trực tiếp
> trên code/terminal để kiểm tra bạn có thực sự hiểu hệ thống hay không.

---

## NHÓM 1: YÊU CẦU CHẠY & THAO TÁC TRỰC TIẾP

### Q1: "Chạy lại training cho tôi xem. Giải thích từng dòng output."

**Cách làm:**
```bash
cd house-price-prediction
python run_training.py
```

**Output sẽ hiện và cách giải thích:**
```
[1/6] Đang tải dữ liệu...
  Chung cư: ~6000 bản ghi      ← Load 2 file CSV từ GitHub
  Nhà đất : ~6000 bản ghi

[2/6] Tiền xử lý dữ liệu...
  Dataset tổng hợp sau lọc: ~11800 bản ghi  ← Gộp 2 loại, lọc outlier (giá 1-200 tỷ, DT 10-1000m²)

[3/6] Thiết lập Pipeline...
  Train: ~9440, Test: ~2360     ← Split 80/20, random_state=42 để reproducible

[5/8] Huấn luyện 4 thuật toán ML...
  ✅ Linear Regression  R²_train=0.xx  R²_test=0.xx  Gap=0.xx   ← Gap nhỏ = không overfit
  ⚠️ Decision Tree      R²_train=0.xx  R²_test=0.xx  Gap=0.xx   ← Gap lớn = overfit!
  ✅ Random Forest       R²_train=0.xx  R²_test=0.xx  Gap=0.xx
  ✅ XGBoost            R²_train=0.xx  R²_test=0.xx  Gap=0.xx   ← R² cao nhất → BEST

[6/8] Cross-Validation (5-Fold)...
  ← Chia data 5 phần, train 5 lần, lấy TB → kiểm tra model ổn định

[7/8] Xuất model...
  ← Lưu pipeline (preprocessor + model) thành 1 file .pkl

[8/8] Tạo biểu đồ...
  ← 3 biểu đồ: so sánh metrics, overfitting analysis, feature importance
```

**Câu nói hay:** "R² Score của XGBoost trên test set cao nhất và Overfit Gap nhỏ, chứng tỏ model vừa chính xác vừa tổng quát tốt."

---

### Q2: "Mở file model ra xem bên trong có gì. Nó lưu cái gì?"

**Cách làm (mở Python REPL):**
```python
import pickle
with open('models/best_model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

print(type(pipeline))          # → sklearn.pipeline.Pipeline
print(pipeline.named_steps)    # → {'preprocessor': ColumnTransformer, 'model': XGBRegressor}

# Xem preprocessor
preprocessor = pipeline.named_steps['preprocessor']
print(preprocessor.transformers)
# → [('num', StandardScaler(), [...5 cột số...]),
#    ('cat', OneHotEncoder(), [...5 cột categorical...])]

# Xem model
model = pipeline.named_steps['model']
print(model.get_params())      # → n_estimators=300, max_depth=7, learning_rate=0.05...
```

**Giải thích:** "File .pkl chứa toàn bộ Pipeline đã train, bao gồm cả bộ tiền xử lý (StandardScaler đã fit mean/std, OneHotEncoder đã biết danh sách categories) VÀ model XGBoost (300 cây quyết định). Khi predict, Pipeline tự động chạy preprocessing rồi mới đưa vào model."

---

### Q3: "Thay đổi 1 tham số rồi train lại xem kết quả thay đổi thế nào."

**Cách làm:** Mở `run_training.py`, sửa dòng 128:
```python
# TRƯỚC:
XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, ...)
# SAU (giảm số cây):
XGBRegressor(n_estimators=50, learning_rate=0.05, max_depth=7, ...)
```
Chạy lại `python run_training.py` → R² sẽ giảm.

**Giải thích:** "Giảm n_estimators từ 300 xuống 50 nghĩa là XGBoost chỉ dùng 50 cây thay vì 300. Ít cây hơn → underfitting → R² giảm. Ngược lại, nếu tăng quá nhiều cây (5000) mà không giảm learning_rate thì sẽ overfit."

---

### Q4: "Predict thử 1 căn nhà bằng code Python, không qua web."

```python
import pickle, pandas as pd

# Load model
with open('models/best_model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Tạo input (phải đúng tên cột lúc train)
test_input = pd.DataFrame([{
    'area_m2': 65,
    'bedrooms_num': 2,
    'district': 'Cầu Giấy',
    'direction': 'Đông - Nam',
    'furniture_std': 'Đầy đủ',
    'legal_std': 'Sổ đỏ',
    'floors_num': 0,
    'frontage_m': 0,
    'road_width_m': 0,
    'loai_bds': 'chung_cu'
}])

# Predict
price = pipeline.predict(test_input)[0]
print(f"Giá dự đoán: {price:.2f} tỷ VNĐ")
```

**Giải thích:** "Pipeline tự động xử lý: StandardScaler chuẩn hóa 5 cột số, OneHotEncoder mã hóa 5 cột categorical, rồi XGBoost predict. Chỉ cần 1 dòng `pipeline.predict()` là xong."

---

### Q5: "Cho tôi xem Feature Importance. Giải thích tại sao feature X lại quan trọng nhất."

```python
import pickle
with open('models/model_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

fi = meta['feature_importance']
# Sắp xếp giảm dần
sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, imp in sorted_fi:
    print(f"  {feat}: {imp:.4f}")
```

**Kết quả thường thấy:** area_m2 là feature quan trọng nhất.

**Giải thích:** "area_m2 (diện tích) quan trọng nhất vì giá nhà tỷ lệ thuận mạnh nhất với diện tích — đây là quy luật kinh tế cơ bản. Tiếp theo là district (vị trí) vì cùng diện tích nhưng Ba Đình đắt hơn Gia Lâm gấp 3-5 lần."

---

### Q6: "Thêm 1 feature mới vào model. Ví dụ thêm 'khoảng cách đến trung tâm'."

**Trả lời:** "Để thêm feature mới, cần sửa 3 chỗ:
1. **Data**: Thêm cột `distance_to_center` vào CSV (có thể tính từ tọa độ quận)
2. **Training** (`run_training.py`): Thêm `'distance_to_center'` vào `FEATURES` list và `numeric_features` list
3. **Web** (`app.py`): Thêm field tương ứng trong `input_dict` của API `/predict`, thêm input vào form HTML

Sau đó chạy lại `python run_training.py` để train model mới. Pipeline sẽ tự điều chỉnh."

---

## NHÓM 2: CÂU HỎI VỀ CODE CỤ THỂ KHI DEMO

### Q7: "Dòng `pipeline.fit(Xtr, ytr)` này làm gì bên trong?"

**Trả lời:** "Nó thực hiện 2 bước tuần tự:
1. `preprocessor.fit_transform(Xtr)` — StandardScaler tính mean/std từ Xtr rồi chuẩn hóa; OneHotEncoder học danh sách categories rồi mã hóa → ra ma trận số
2. `model.fit(X_transformed, ytr)` — XGBoost nhận ma trận số và train 300 cây boosted

Khi gọi `pipeline.predict(Xte)`, nó chỉ `transform` (không fit lại) rồi predict → đảm bảo test data được xử lý giống train."

---

### Q8: "Mở app.py, chỉ cho tôi luồng từ lúc user bấm nút đến lúc hiện kết quả."

**Trả lời (chỉ trực tiếp trong code):**
1. **Frontend** (`script.js` dòng 128-170): User submit form → JS thu thập data → `fetch('/predict', {body: JSON})`
2. **Backend** (`app.py` dòng 427-644): Flask nhận request → validate input → tạo DataFrame → `model.predict(input_data)` → tính XAI contributions → tìm BĐS tương tự → return JSON
3. **Frontend** (`script.js` dòng 174-290): Nhận JSON → animate giá (count up) → render XAI bars → render similar properties → hiện badge so sánh khu vực

**Thời gian xử lý:** < 100ms (model predict rất nhanh, bottleneck là tìm BĐS tương tự trong CSV)

---

### Q9: "Giảng viên nhập giá trị bất thường: diện tích = -5 hoặc 99999. Xử lý thế nào?"

**Trả lời (chỉ code):**
- `app.py` dòng 438-457: Validation server-side
```python
if area < 0 or area > 50000:
    return jsonify({'success': False, 'error': 'Diện tích không hợp lệ!'}), 400
```
- `script.js` dòng 140-152: Validation client-side
```javascript
if (!areaInput.value || parseFloat(areaInput.value) <= 0) {
    areaInput.classList.add('input-error');
    hasError = true;
}
```

"Hệ thống validate 2 lớp: frontend chặn trước khi gửi, backend chặn lần nữa để bảo mật (phòng trường hợp gọi API trực tiếp)."

---

### Q10: "Tại sao dùng Flask Session chứ không dùng JWT Token?"

**Trả lời:** "Flask Session lưu server-side (mã hóa bằng secret_key), phù hợp với server-rendered app (Jinja2 template). JWT phù hợp hơn cho SPA (React/Vue) hoặc mobile app vì stateless. Dự án này là server-rendered nên Session là lựa chọn tự nhiên. Nếu chuyển sang SPA thì nên đổi sang JWT."

---

## NHÓM 3: YÊU CẦU GIẢI THÍCH THUẬT TOÁN TRỰC TIẾP

### Q11: "Vẽ cho tôi cách XGBoost hoạt động trên bảng trắng."

**Cách vẽ:**
```
Dữ liệu gốc X, y
    ↓
Cây 1: predict → y_pred_1 → tính residual_1 = y - y_pred_1
    ↓
Cây 2: học từ residual_1 → predict → y_pred_2 → residual_2 = residual_1 - lr*y_pred_2
    ↓
Cây 3: học từ residual_2 → ...
    ↓
... (300 cây)
    ↓
Kết quả cuối = y_pred_1 + lr*y_pred_2 + lr*y_pred_3 + ... + lr*y_pred_300
```

**Giải thích:** "XGBoost = eXtreme Gradient Boosting. Mỗi cây mới CỐ GẮNG SỬA LỖI của cây trước (học từ residual/gradient). learning_rate=0.05 nghĩa là mỗi cây chỉ đóng góp 5% → cần nhiều cây (300) nhưng ổn định hơn. Khác với Random Forest (300 cây SONG SONG, lấy trung bình)."

---

### Q12: "Giải thích sự khác biệt Bagging vs Boosting bằng ví dụ thực tế."

**Trả lời:**
- **Bagging (Random Forest):** Giống như hỏi 200 chuyên gia BĐS ĐỒNG THỜI, mỗi người xem 1 phần data khác nhau, rồi lấy trung bình → giảm variance, tránh 1 ý kiến cực đoan chi phối
- **Boosting (XGBoost):** Giống như 1 chuyên gia định giá lần 1, rồi 1 chuyên gia khác xem lại CÁC CĂN ĐỊNH GIÁ SAI → sửa lỗi → rồi chuyên gia thứ 3 lại sửa tiếp → 300 vòng sửa lỗi liên tiếp → giảm bias

"Boosting thường cho kết quả tốt hơn Bagging trên tabular data, nhưng dễ overfit hơn nếu không tune."

---

### Q13: "Tại sao train_test_split dùng random_state=42?"

**Trả lời:** "random_state=42 là seed cho bộ sinh số ngẫu nhiên. Nó đảm bảo **reproducibility** — mỗi lần chạy đều split data giống nhau → kết quả nhất quán để so sánh. Số 42 không có ý nghĩa đặc biệt (convention trong cộng đồng ML, lấy từ 'The Hitchhiker's Guide to the Galaxy'). Có thể dùng bất kỳ số nào."

---

## NHÓM 4: YÊU CẦU SỬA LỖI / DEBUG TRỰC TIẾP

### Q14: "Nếu tôi xóa file model .pkl, hệ thống xử lý thế nào?"

**Trả lời (chỉ code `app.py` dòng 124-135):**
```python
try:
    with open(model_path, 'rb') as f:
        models[name] = pickle.load(f)
    print(f"[OK] Da load thanh cong {name}")
except FileNotFoundError:
    print(f"[ERROR] Chua tim thay model {name}...")
    models[name] = None  # ← Model = None
```

Khi predict (`app.py` dòng 490-497):
```python
model = models.get('best_model_pipeline')
if model:
    prediction_billion = float(model.predict(input_data)[0])
else:
    prediction_billion = float(data.get('area', 0)) * 0.1  # ← Fallback đơn giản
```

"Hệ thống có fallback: nếu không có model, dùng công thức đơn giản (giá = diện tích × 0.1). Không crash nhưng kết quả sẽ không chính xác."

---

### Q15: "Server đang chạy mà sửa code app.py, có cần restart không?"

**Trả lời:** "Mặc định không tự reload vì `use_reloader=False` (dòng 857). Nếu muốn auto-reload khi develop, set `FLASK_DEBUG=1` hoặc sửa code thành `use_reloader=True`. Trong production, KHÔNG BAO GIỜ bật debug mode vì nó expose debugger có thể thực thi code tùy ý."

---

### Q16: "Tại sao dùng `pd.DataFrame([input_dict])` mà không truyền dict thẳng vào model?"

**Trả lời:** "scikit-learn Pipeline yêu cầu input là 2D array hoặc DataFrame. Dict là 1D → không đúng format. `pd.DataFrame([input_dict])` tạo DataFrame 1 dòng với tên cột khớp lúc train. Nếu truyền dict thẳng, Pipeline sẽ raise `ValueError: Expected 2D array`."

---

## NHÓM 5: CÂU HỎI VỀ DỮ LIỆU KHI DEMO

### Q17: "Mở file CSV cho tôi xem. Dữ liệu trông như thế nào?"

```python
import pandas as pd
df = pd.read_csv('data/processed/cleaned_chung_cu.csv')
print(df.shape)          # → (6xxx, 12)
print(df.columns.tolist()) # → ['price_billion','area_m2','bedrooms_num','district',...]
print(df.head())
print(df.describe())     # → Thống kê: mean, std, min, max
print(df['district'].value_counts().head())  # → Top quận nhiều nhất
print(df.isnull().sum())  # → Số missing values mỗi cột
```

---

### Q18: "Dữ liệu có bao nhiêu quận? Liệt kê ra."

```python
districts = df['district'].unique()
print(f"Tổng: {len(districts)} quận")
print(sorted(districts))
```

"Có khoảng 43 quận/huyện từ Hà Nội, TP.HCM, Đà Nẵng. Model OneHotEncoder tạo 43 cột binary tương ứng."

---

### Q19: "Nếu người dùng nhập 1 quận KHÔNG CÓ trong data training (ví dụ 'Quận ABC'), chuyện gì xảy ra?"

**Trả lời:** "OneHotEncoder có tham số `handle_unknown='ignore'` (dòng 79 run_training.py). Khi gặp category chưa biết, nó tạo vector toàn 0 cho cột district → model coi như không có thông tin vị trí → dự đoán dựa trên các feature còn lại. Kết quả sẽ kém chính xác nhưng KHÔNG crash."

---

### Q20: "Demo trực tiếp API bằng curl/Postman cho tôi xem."

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "property_type": "chung_cu",
    "area": 65,
    "bedrooms": 2,
    "district": "Cầu Giấy",
    "direction": "Đông - Nam",
    "furniture": "Đầy đủ",
    "legal": "Sổ đỏ"
  }'
```

**Response:**
```json
{
  "success": true,
  "price_billion": 3.45,
  "price_low": 2.60,
  "price_high": 4.30,
  "price_per_m2": 53.1,
  "contributions": [...],
  "similar_properties": [...]
}
```

---

## NHÓM 6: CÂU HỎI BẪY (GIẢNG VIÊN CỐ TÌNH GÀI)

### Q21: "Model này có dùng Neural Network không?"

**Trả lời:** "KHÔNG. XGBoost là tree-based model (cây quyết định), KHÔNG phải Neural Network. Nó dùng Gradient Boosting trên Decision Trees. Với dữ liệu tabular ~12K mẫu, tree-based models (XGBoost, LightGBM) thường tốt hơn Neural Networks."

---

### Q22: "R² = 0.85 nghĩa là model đúng 85% phải không?"

**Trả lời:** "KHÔNG ĐÚNG. R² = 0.85 nghĩa là model giải thích được 85% PHƯƠNG SAI (variance) của giá nhà, KHÔNG phải đúng 85% trường hợp. Để đo tỷ lệ sai lệch, ta dùng MAPE (Mean Absolute Percentage Error). Ví dụ MAPE = 20% nghĩa là trung bình mỗi dự đoán lệch khoảng 20% so với giá thực."

---

### Q23: "Nếu tất cả nhà đều có giá 5 tỷ, model sẽ thế nào?"

**Trả lời:** "R² = 0, vì R² = 1 - SS_res/SS_tot. Khi tất cả y bằng nhau → SS_tot = 0 → R² undefined (hoặc 0). Model không học được gì vì không có variance để giải thích. Nó sẽ chỉ predict trung bình = 5 tỷ cho mọi input."

---

### Q24: "Tại sao không tự viết thuật toán mà dùng thư viện?"

**Trả lời:** "Vì: (1) scikit-learn/XGBoost được phát triển bởi hàng trăm kỹ sư, đã test trên hàng triệu bài toán, (2) XGBoost có optimizations C++ cấp thấp (SIMD, cache-aware, histogram-based) mà tự viết Python không thể đạt được, (3) Mục tiêu môn học là XÂY DỰNG HỆ THỐNG THÔNG MINH — biết chọn đúng công cụ, tune đúng tham số, đánh giá đúng kết quả quan trọng hơn việc implement lại thuật toán từ đầu."

---

## CHECKLIST TRƯỚC KHI DEMO

- [ ] Server chạy OK: `cd app && python app.py`
- [ ] Mở sẵn browser tại `http://127.0.0.1:5000`
- [ ] Mở sẵn terminal thứ 2 với Python REPL (để demo code trực tiếp)
- [ ] Mở sẵn `run_training.py` và `app.py` trong editor
- [ ] Đã chạy `python run_training.py` ít nhất 1 lần (model .pkl tồn tại)
- [ ] Thuộc lòng: 10 features, 4 thuật toán, R²/MAE/MAPE nghĩa là gì
- [ ] Biết chỉ dòng code cụ thể khi được hỏi (Pipeline dòng 76-80, predict dòng 493)
