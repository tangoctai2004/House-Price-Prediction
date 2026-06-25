# CHUẨN BỊ BẢO VỆ — MÔN PHÁT TRIỂN HỆ THỐNG THÔNG MINH

## TÔI CẦN HỌC GÌ?

**CÓ, bạn cần hiểu code và thuật toán** — nhưng không phải thuộc lòng từng dòng. GV môn này quan tâm đến **tư duy** đằng sau mỗi quyết định kỹ thuật, không phải cú pháp Python.

Cụ thể, bạn cần hiểu **3 tầng**:
1. **Tại sao** — Lý do đưa ra quyết định (quan trọng nhất)
2. **Như thế nào** — Thuật toán hoạt động ra sao (quan trọng)
3. **Ở đâu trong code** — Chỉ được vào đúng chỗ khi GV hỏi (biết thì tốt)

---

## PHẦN 1: NHỮNG GÌ GV SẼ KHOÉT SÂU

Môn "Phát triển Hệ thống Thông minh" → GV sẽ tập trung vào phần **"thông minh"** của hệ thống. Tức là:

### 🔴 MỨC ĐỘ 1 — CHẮC CHẮN BỊ HỎI (Phải trả lời được)

#### 1. XGBoost hoạt động thế nào? Tại sao nó thắng?

Đây là câu hỏi số 1. GV sẽ không hỏi "XGBoost là gì" mà hỏi **"giải thích cho tôi tại sao nó tốt hơn 3 cái kia"**.

**Cách trả lời:**

> Linear Regression giả định quan hệ tuyến tính — nhưng giá nhà phụ thuộc phi tuyến (căn 50m² ở Ba Đình khác hoàn toàn 50m² ở Gia Lâm). Nên R² chỉ đạt ~0.56.
>
> Decision Tree học tốt nhưng **overfit nặng** — R² train = 0.93 nhưng test chỉ 0.65 (gap 0.28). Nó "thuộc bài" training data.
>
> Random Forest giảm overfitting bằng Bagging (200 cây, mỗi cây train trên bootstrap sample khác nhau, lấy trung bình) → R² ~0.80, gap nhỏ.
>
> **XGBoost thắng vì Boosting**: mỗi cây mới được xây để **sửa lỗi** cây trước (học từ phần dư residual). Kết hợp với Regularization L1+L2 để kiểm soát overfitting → R² ~0.841, gap chỉ ~0.04.

**Nếu GV hỏi sâu hơn** — "Boosting khác Bagging chỗ nào?":
- **Bagging** (Random Forest): 200 cây xây **song song, độc lập** → lấy trung bình → giảm variance
- **Boosting** (XGBoost): 300 cây xây **tuần tự** → cây sau sửa lỗi cây trước → giảm cả bias lẫn variance

#### 2. Pipeline sklearn là gì? Tại sao quan trọng?

GV sẽ hỏi **Data Leakage** — đây là khái niệm cốt lõi.

**Cách trả lời:**

> Nếu em fit StandardScaler trên toàn bộ dữ liệu (train + test), scaler sẽ biết mean và std của test set → model "nhìn trước" dữ liệu chưa gặp → R² tăng ảo → ra thực tế predict sai.
>
> Pipeline đóng gói ColumnTransformer + Model vào 1 object duy nhất. Khi gọi `pipeline.fit(X_train, y_train)`, OHE và Scaler chỉ fit trên train. Khi predict, pipeline tự transform input mới bằng các tham số đã fit → không bao giờ leak.
>
> Lúc deploy lên web, em chỉ cần load 1 file `.pkl` và gọi `model.predict(raw_dataframe)` — pipeline tự xử lý hết bên trong.

#### 3. Đánh giá model — R², RMSE, MAE nghĩa là gì?

GV sẽ hỏi: **"R² = 0.841 có nghĩa là gì? Tốt hay chưa tốt?"**

**Cách trả lời:**

> R² = 0.841 nghĩa là model giải thích được 84.1% sự biến động của giá nhà. 15.9% còn lại là do các yếu tố model không capture được: cảm xúc người bán, quy hoạch tương lai, phong thủy...
>
> Với bài toán giá BĐS, R² từ 0.78–0.85 được coi là tốt. R² > 0.90 trên test set cần kiểm tra kỹ overfitting.
>
> RMSE phạt nặng sai số lớn (bình phương), MAE đo sai số trung bình tuyệt đối. Trong BĐS có outlier (biệt thự 100+ tỷ), MAE đáng tin hơn RMSE.

#### 4. Dữ liệu — Quá trình thu thập và xử lý

GV sẽ hỏi: **"Dữ liệu lấy từ đâu? Có vấn đề gì không? Xử lý thế nào?"**

**Cách trả lời:**

> Tự crawl 14.015 bản ghi từ batdongsan.com.vn bằng curl_cffi (bypass Cloudflare). Sau tiền xử lý còn 11.831 bản ghi sạch.
>
> Vấn đề lớn nhất: **370 dòng bị đảo cột** — scraper parse nhầm giá tổng vào cột giá/m² và ngược lại. Em phát hiện bằng cách kiểm tra nếu cột price chứa chuỗi "triệu/m" thì swap lại.
>
> Ngoài ra: 1.893 tin không có giá (loại bỏ), quận < 30 mẫu gom vào "Khác", fill NaN phòng ngủ bằng median theo quận.

---

### 🟠 MỨC ĐỘ 2 — CÓ THỂ BỊ HỎI (Nên chuẩn bị)

#### 5. Feature Engineering — Tại sao chọn những features này?

> **10 features**: area_m2, bedrooms_num, district, direction, furniture_std, legal_std, floors_num, frontage_m, road_width_m, loai_bds
>
> EDA cho thấy `area_m2` có tương quan cao nhất với giá (r=0.667 cho CC). `floors_num` và `frontage_m` chỉ có ý nghĩa với Nhà đất (r=0.386 và 0.202) → CC gán = 0.
>
> Dùng OHE cho categorical (district, direction...) vì XGBoost dựa trên tree-based splitting — OHE cho phép split trên từng quận riêng biệt.

#### 6. Overfitting — Nhận biết và xử lý thế nào?

> Decision Tree: R² train = 0.93, test = 0.65 → gap = 0.28 → **overfit nặng**. Cây học thuộc training data.
>
> XGBoost kiểm soát overfitting bằng 4 cơ chế:
> 1. `learning_rate=0.05` — mỗi cây chỉ đóng góp 5%, cần nhiều cây mới hội tụ
> 2. `max_depth=7` — giới hạn độ sâu, tránh cây quá phức tạp
> 3. `subsample=0.8` — mỗi cây chỉ dùng 80% dữ liệu ngẫu nhiên
> 4. Regularization L1+L2 — phạt mô hình phức tạp

#### 7. EDA dẫn đến quyết định kiến trúc gì?

> 5 biểu đồ cho thấy CC và NĐ khác nhau hoàn toàn:
> - Boxplot: phân phối giá khác nhau (CC median ~7.2 tỷ vs NĐ ~12.95 tỷ)
> - Heatmap: bộ features quan trọng khác nhau (NĐ có floors/frontage, CC không có)
>
> → Quyết định: dùng cột `loai_bds` để model tự phân biệt, gán features không tồn tại = 0

#### 8. Singleton Pattern — Load model vào web thế nào?

> Load model từ disk tốn 2-5 giây. Nếu mỗi request đều load lại → response 3-5 giây.
>
> Giải pháp: biến toàn cục `models = {}`, gọi `load_model()` 1 lần khi Flask khởi động. Mọi request sau lấy model từ RAM → response < 0.3 giây.

---

### 🟡 MỨC ĐỘ 3 — ÍT KHI HỎI NHƯNG NẾU HỎI THÌ GHI ĐIỂM

#### 9. Tại sao không log transform?

> Phân phối giá lệch phải (skewness CC=2.75, NĐ=3.98). Thông thường nên log transform. Nhưng thử nghiệm cho thấy XGBoost xử lý tốt phân phối lệch nhờ Boosting — R² không cải thiện đáng kể khi dùng log → giữ nguyên đơn vị tỷ VNĐ cho đơn giản khi deploy.

#### 10. Hạn chế lớn nhất? Hướng phát triển?

> - **In-memory storage**: Dữ liệu user/history mất khi restart → cần SQLAlchemy + database
> - **Chưa cross-validation**: Chỉ 1 lần train_test_split → cần 5-fold CV
> - **XAI đơn giản**: Dùng Feature Importance × prediction, không phải SHAP → cần SHAP TreeExplainer
> - **Không retrain tự động**: Giá BĐS biến động → cần Celery scheduler crawl + retrain định kỳ

---

## PHẦN 2: CÓ CẦN THUỘC CODE KHÔNG?

### Không cần thuộc, nhưng cần HIỂU 5 đoạn code sau:

**Đoạn 1 — Pipeline tiền xử lý** (`run_training.py` dòng 74-78):
```python
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['area_m2', 'bedrooms_num', 'floors_num', 'frontage_m', 'road_width_m']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['district', 'direction', 'furniture_std', 'legal_std', 'loai_bds'])
])
```
→ Hiểu: Số được chuẩn hóa (mean=0, std=1). Phân loại được mã hóa nhị phân. `handle_unknown='ignore'` để khi gặp quận mới không crash.

**Đoạn 2 — Wrap Pipeline + Model** (`run_training.py` dòng 89-92):
```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', base_model)
])
```
→ Hiểu: 1 object duy nhất xử lý từ raw data đến prediction. Fit trên train, predict trên bất kỳ data mới nào.

**Đoạn 3 — Load model 1 lần** (`app.py` dòng 122-138):
```python
models = {}
def load_model(name):
    with open(model_path, 'rb') as f:
        models[name] = pickle.load(f)
load_model('best_model_pipeline')
```
→ Hiểu: Singleton Pattern. Load khi Flask khởi động, mọi request sau đều dùng chung object trong RAM.

**Đoạn 4 — Predict** (`app.py` dòng 462-493):
```python
input_dict = {'area_m2': area, 'bedrooms_num': bedrooms, ..., 'loai_bds': property_type}
if property_type == 'nha_dat':
    input_dict['floors_num'] = floors   # Chỉ NĐ mới có
# CC giữ nguyên floors_num=0, frontage_m=0, road_width_m=0

input_data = pd.DataFrame([input_dict])
prediction_billion = model.predict(input_data)[0]
```
→ Hiểu: 1 dict → 1 DataFrame 1 dòng → Pipeline tự OHE + Scale → XGBoost predict → ra số tỷ VNĐ.

**Đoạn 5 — Tìm BĐS tương tự** (`app.py` dòng 567-581):
```python
filtered = df[(df['price_billion'] >= prediction * 0.8) & (df['price_billion'] <= prediction * 1.2)]
same_dist = filtered[filtered['district'] == user_dist]
# Ưu tiên cùng quận, fallback lấy quận khác
```
→ Hiểu: Lọc ±20% giá dự đoán, ưu tiên cùng quận → gợi ý 12 BĐS tương tự.

---

## PHẦN 3: CÁC CON SỐ QUAN TRỌNG NHẤT

Chỉ cần nhớ những con số này:

| Cần nhớ | Số |
|---------|-----|
| Bản ghi sau xử lý | **11.831** (5.452 CC + 6.379 NĐ) |
| R² XGBoost | **~0.841** |
| Overfit gap DT | **~0.28** (train 0.93, test 0.65) |
| Overfit gap XGB | **~0.04** (ổn định) |
| XGBoost params | **300 cây, lr=0.05, depth=7, subsample=0.8** |
| Response time | **< 0.3 giây** |
| Số features | **10** + 1 target (price_billion) |

---

## PHẦN 4: THỨ TỰ ƯU TIÊN HỌC TỐI NAY

1. **Đọc kỹ phần 1 mức đỏ** (câu 1-4) — 30 phút — đây là thứ GV chắc chắn hỏi
2. **Hiểu 5 đoạn code phần 2** — 20 phút — không cần thuộc, chỉ cần giải thích được
3. **Nhớ con số phần 3** — 5 phút
4. **Đọc phần 1 mức cam** (câu 5-8) — 15 phút — nếu còn thời gian
5. **Demo thử 1 lần** trên máy — 10 phút — đảm bảo web chạy được

**Tổng: ~1.5 tiếng là đủ.**
